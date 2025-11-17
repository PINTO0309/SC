#!/usr/bin/env python3
"""Estimate feature importance for sc_c_32x24.onnx by ablating intermediate tensors."""

from __future__ import annotations

import argparse
import copy
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnx import TensorProto, helper, shape_inference  # noqa: E402
from PIL import Image  # noqa: E402


@dataclass(frozen=True)
class FeatureTarget:
    tensor_name: str
    node_name: str
    op_type: str
    default_selected: bool


@dataclass(frozen=True)
class GateSpec:
    tensor_name: str
    gate_name: str
    node_name: str
    op_type: str


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_val: float = math.inf
    max_val: float = -math.inf

    def update(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    def std(self) -> float:
        if self.count <= 1:
            return float("nan")
        mean = self.mean()
        variance = max((self.total_sq / self.count) - mean * mean, 0.0)
        return math.sqrt(variance)


@dataclass
class GateStats:
    prob: RunningStats = field(default_factory=RunningStats)
    delta: RunningStats = field(default_factory=RunningStats)


def _gather_image_paths(image: str | None, image_dir: str | None, max_images: int) -> List[Path]:
    if image and image_dir:
        raise ValueError("Use either --image or --image-dir, not both.")
    if not image and not image_dir:
        raise ValueError("Provide --image or --image-dir when running ablations.")
    if image:
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return [path]
    root = Path(image_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Image directory not found: {root}")
    allowed_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    candidates = sorted(p for p in root.rglob("*") if p.suffix.lower() in allowed_ext)
    if not candidates:
        raise RuntimeError(f"No images found under {root}")
    max_use = max(1, max_images)
    return candidates[:max_use]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert scalar gates after selected ONNX tensors and measure prob_sitting drops when zeroed."
    )
    parser.add_argument("--model", default="sc_c_32x24.onnx", help="Path to ONNX model.")
    parser.add_argument("--image", help="Path to a single RGB image.")
    parser.add_argument(
        "--image-dir",
        help="Directory containing RGB images; up to --max-images will be used (sorted lexicographically).",
    )
    parser.add_argument("--max-images", type=int, default=100, help="Maximum number of samples loaded from --image-dir.")
    parser.add_argument(
        "--input-size",
        type=_parse_input_size,
        default=(32, 24),
        help="Input size as HxW (e.g. 32x24). Single integers apply to both height and width.",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(0.0, 0.0, 0.0),
        help="Per-channel normalization mean in [0,1].",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(1.0, 1.0, 1.0),
        help="Per-channel normalization std in [0,1].",
    )
    parser.add_argument("--providers", nargs="*", help="Optional ONNX Runtime providers override.")
    parser.add_argument("--limit", type=int, help="Maximum number of tensors to gate.")
    parser.add_argument("--name-contains", nargs="*", help="Select tensors whose node or tensor name contains any term.")
    parser.add_argument("--op-type", nargs="*", help="Select tensors produced by these operator types.")
    parser.add_argument("--regex", nargs="*", help="Regex patterns applied to tensor names for selection.")
    parser.add_argument("--include-all", action="store_true", help="Disable default (Sigmoid->Mul only) filtering.")
    parser.add_argument("--list-targets", action="store_true", help="List matching tensors and exit.")
    parser.add_argument(
        "--ablation-value",
        type=float,
        default=0.0,
        help="Scalar gate value applied during ablation (default zeros the tensor).",
    )
    parser.add_argument(
        "--topk-heatmaps",
        type=int,
        default=6,
        help="Automatically render heatmaps for the top-K tensors after ranking (set 0 to disable).",
    )
    parser.add_argument(
        "--topk-image",
        help="Optional override image used when rendering top-K heatmaps (defaults to the first processed image).",
    )
    parser.add_argument(
        "--topk-output-dir",
        default="ablation_heatmaps",
        help="Directory where auto-rendered top-K heatmaps and composite PNG are stored.",
    )
    parser.add_argument(
        "--topk-composite-name",
        default="ablation_top_features.png",
        help="Filename for the stitched 3x2 PNG generated from the top-K heatmaps.",
    )
    default_heatmap_script = Path(__file__).with_name("10_visualize_sc_heatmaps.py")
    parser.add_argument(
        "--heatmap-script",
        default=str(default_heatmap_script),
        help="Path to 10_visualize_sc_heatmaps.py used for auto-rendering top-K heatmaps.",
    )
    return parser.parse_args()


def _load_model(path: Path) -> onnx.ModelProto:
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    return onnx.load(path.as_posix())


def _build_producers(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
    producers: Dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            if out:
                producers[out] = node
    return producers


def _collect_targets(model: onnx.ModelProto) -> List[FeatureTarget]:
    producers = _build_producers(model)
    targets: List[FeatureTarget] = []
    for node in model.graph.node:
        has_sigmoid_input = False
        if node.op_type == "Mul":
            for inp in node.input:
                producer = producers.get(inp)
                if producer is not None and producer.op_type == "Sigmoid":
                    has_sigmoid_input = True
                    break
        for out in node.output:
            if not out:
                continue
            targets.append(
                FeatureTarget(
                    tensor_name=out,
                    node_name=node.name or out,
                    op_type=node.op_type,
                    default_selected=has_sigmoid_input,
                )
            )
    return targets


def _filter_targets(
    targets: Sequence[FeatureTarget],
    name_terms: Sequence[str] | None,
    op_types: Sequence[str] | None,
    regex_patterns: Sequence[str] | None,
    include_all: bool,
    limit: int | None,
) -> List[FeatureTarget]:
    filtered: Iterable[FeatureTarget]
    if name_terms or op_types or regex_patterns or include_all:
        filtered = targets
    else:
        filtered = [t for t in targets if t.default_selected]

    if name_terms:
        lowered = [term.lower() for term in name_terms]
        filtered = [
            t for t in filtered if any(term in t.tensor_name.lower() or term in (t.node_name or "").lower() for term in lowered)
        ]
    if op_types:
        allowed = {op.upper() for op in op_types}
        filtered = [t for t in filtered if t.op_type.upper() in allowed]
    if regex_patterns:
        patterns = [re.compile(p) for p in regex_patterns]
        filtered = [t for t in filtered if any(p.search(t.tensor_name) for p in patterns)]

    filtered_list = list(filtered)
    if limit is not None:
        filtered_list = filtered_list[: max(0, limit)]
    return filtered_list


def _infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return shape_inference.infer_shapes(model, strict_mode=False)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Shape inference failed: {exc}. Proceeding with original graph.")
        return model


def _value_info_map(model: onnx.ModelProto) -> Dict[str, onnx.ValueInfoProto]:
    mapping: Dict[str, onnx.ValueInfoProto] = {}
    for info in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        mapping[info.name] = info
    return mapping


def _make_shape_from_value_info(value_info) -> List[int | str] | None:
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims if any(d is not None for d in dims) else None


def _slugify(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_") or "tensor"


def _insert_gates(model: onnx.ModelProto, targets: Sequence[FeatureTarget]) -> tuple[onnx.ModelProto, List[GateSpec]]:
    gated = copy.deepcopy(model)
    value_infos = _value_info_map(gated)
    target_map = {t.tensor_name: t for t in targets}
    target_order = {t.tensor_name: idx for idx, t in enumerate(targets)}
    gate_specs: List[GateSpec] = []
    insertions: List[tuple[int, onnx.NodeProto]] = []

    for node_idx, node in enumerate(gated.graph.node):
        for out_idx, output_name in enumerate(node.output):
            if output_name not in target_map:
                continue
            target = target_map[output_name]
            info = value_infos.get(output_name)
            if info is None:
                raise ValueError(f"No value info available for tensor {output_name}. Run shape inference first.")

            gate_name = f"gate_{_slugify(output_name)}"
            pre_gate_name = f"{output_name}__pre_gate"

            node.output[out_idx] = pre_gate_name
            new_value_info = helper.make_tensor_value_info(
                pre_gate_name,
                info.type.tensor_type.elem_type,
                _make_shape_from_value_info(info),
            )
            gated.graph.value_info.append(new_value_info)

            gate_input = helper.make_tensor_value_info(gate_name, TensorProto.FLOAT, [])
            gated.graph.input.append(gate_input)

            mul_node = helper.make_node(
                "Mul",
                inputs=[pre_gate_name, gate_name],
                outputs=[output_name],
                name=f"Gate::{_slugify(output_name)}",
            )
            insertions.append((node_idx + 1, mul_node))
            gate_specs.append(
                GateSpec(
                    tensor_name=output_name,
                    gate_name=gate_name,
                    node_name=target.node_name,
                    op_type=target.op_type,
                )
            )

    if len(gate_specs) != len(targets):
        missing = {t.tensor_name for t in targets} - {spec.tensor_name for spec in gate_specs}
        raise RuntimeError(f"Failed to insert gates for tensors: {sorted(missing)}")

    offset = 0
    for insert_idx, new_node in insertions:
        gated.graph.node.insert(insert_idx + offset, new_node)
        offset += 1

    gate_specs.sort(key=lambda spec: target_order.get(spec.tensor_name, 0))
    return gated, gate_specs


def _parse_input_size(value: str | int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise argparse.ArgumentTypeError("Input size tuple must be (height, width).")
        return value
    if isinstance(value, int):
        if value <= 0:
            raise argparse.ArgumentTypeError("Input size must be positive.")
        return value, value
    text = str(value).strip().lower()
    if not text:
        raise argparse.ArgumentTypeError("Input size cannot be empty.")
    if "x" in text:
        parts = text.split("x")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("Use HxW format (e.g. 32x24).")
        height_str, width_str = parts
    else:
        height_str = width_str = text
    try:
        height = int(height_str)
        width = int(width_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Input size must be integers in HxW format.") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("Input dimensions must be positive.")
    return height, width


def _load_image(path: Path, input_size: Tuple[int, int]) -> np.ndarray:
    height, width = input_size
    if height <= 0 or width <= 0:
        raise ValueError("Input size must be positive.")
    image = Image.open(path).convert("RGB")
    resized = image.resize((width, height), Image.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    return arr


def _prepare_tensor(image_arr: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    if not np.all(std_arr):
        raise ValueError("Std entries must be non-zero.")
    normalized = (image_arr - mean_arr) / std_arr
    tensor = normalized.transpose(2, 0, 1)[None, ...].astype(np.float32)
    return tensor


def _select_providers(requested: Sequence[str] | None) -> List[str]:
    available = ort.get_available_providers()
    if requested:
        selected = [p for p in requested if p in available]
    else:
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        selected = [p for p in preferred if p in available]
        if not selected:
            selected = list(available)
    if not selected:
        raise RuntimeError("No usable ONNX Runtime providers found.")
    return selected


def _build_session(model: onnx.ModelProto, providers: Sequence[str]) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    return ort.InferenceSession(model.SerializeToString(), sess_options=opts, providers=list(providers))


def _run_prob(session: ort.InferenceSession, feeds: Dict[str, np.ndarray], output_name: str) -> float:
    prob = session.run([output_name], feeds)[0].squeeze()
    return float(prob)


def _render_topk_heatmaps(
    model_path: Path,
    heatmap_script: Path,
    image_path: Path,
    tensor_names: Sequence[str],
    output_dir: Path,
    composite_name: str,
) -> None:
    if not tensor_names:
        print("[WARN] No tensor names provided for heatmap rendering.")
        return
    if not image_path.exists():
        print(f"[WARN] Heatmap reference image not found: {image_path}")
        return
    if not heatmap_script.exists():
        print(f"[WARN] Heatmap script not found: {heatmap_script}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    lowered_layers = [name.lower() for name in tensor_names]
    cmd = [
        sys.executable,
        str(heatmap_script),
        "--model",
        str(model_path),
        "--image",
        str(image_path),
        "--output-dir",
        str(output_dir),
        "--limit",
        str(len(lowered_layers)),
        "--composite-topk",
        str(len(lowered_layers)),
        "--composite-output",
        composite_name,
        "--composite-sort",
        "order",
        "--composite-layout",
        "col",
    ]
    cmd.extend(["--layers", *lowered_layers])
    print(f"[INFO] Rendering heatmaps for top-{len(lowered_layers)} tensors -> {output_dir}/{composite_name}")
    sys.stdout.flush()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] Heatmap script failed (exit code {exc.returncode}). Command: {' '.join(cmd)}")


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model)
    base_model = _load_model(model_path)
    targets = _collect_targets(base_model)
    selected = _filter_targets(targets, args.name_contains, args.op_type, args.regex, args.include_all, args.limit)

    if args.list_targets:
        for idx, target in enumerate(selected):
            marker = "*" if target.default_selected else " "
            print(f"[{idx:03d}] {marker} {target.op_type:<8} {target.node_name} -> {target.tensor_name}")
        if not selected:
            print("No tensors matched the current filters.")
        return

    if not selected:
        raise RuntimeError(
            "No tensors selected. Add --name-contains/--op-type/--regex or --include-all to broaden the search."
        )
    image_paths = _gather_image_paths(args.image, args.image_dir, args.max_images)

    inferred = _infer_shapes(base_model)
    gated_model, gate_specs = _insert_gates(inferred, selected)
    providers = _select_providers(args.providers)
    session = _build_session(gated_model, providers)

    input_name = session.get_inputs()[0].name
    gate_one = np.array(1.0, dtype=np.float32)
    gate_zero = np.array(args.ablation_value, dtype=np.float32)
    gate_defaults = {spec.gate_name: gate_one for spec in gate_specs}
    tensor_to_stats: Dict[str, GateStats] = {spec.tensor_name: GateStats() for spec in gate_specs}
    baseline_stats = RunningStats()

    num_images = len(image_paths)
    multi_mode = num_images > 1

    output_name = base_model.graph.output[-1].name
    for idx, image_path in enumerate(image_paths, start=1):
        image_arr = _load_image(Path(image_path), args.input_size)
        input_tensor = _prepare_tensor(image_arr, args.mean, args.std)
        feeds: Dict[str, np.ndarray] = dict(gate_defaults)
        feeds[input_name] = input_tensor

        baseline = _run_prob(session, feeds, output_name)
        baseline_stats.update(baseline)
        if multi_mode:
            print(f"[IMAGE {idx:04d}] {image_path}: baseline={baseline:.4f}")
        else:
            print(f"[BASELINE] prob_sitting={baseline:.4f} with {len(gate_specs)} gates active.")

        for spec in gate_specs:
            ablate_feeds = dict(feeds)
            ablate_feeds[spec.gate_name] = gate_zero
            prob = _run_prob(session, ablate_feeds, output_name)
            delta = baseline - prob
            stats = tensor_to_stats[spec.tensor_name]
            stats.prob.update(prob)
            stats.delta.update(delta)
            if not multi_mode:
                print(f"[ABLATE] {spec.tensor_name}: prob={prob:.4f} delta={delta:+.4f}")

    if not tensor_to_stats:
        return

    ordered_specs = sorted(
        gate_specs, key=lambda spec: tensor_to_stats[spec.tensor_name].delta.mean(), reverse=True
    )

    if multi_mode:
        print(
            f"\nProcessed {num_images} images. Baseline prob: mean={baseline_stats.mean():.4f} "
            f"std={baseline_stats.std():.4f} min={baseline_stats.min_val:.4f} max={baseline_stats.max_val:.4f}"
        )
        print("\nAggregate feature impact (sorted by mean Δ):")
        for rank, spec in enumerate(ordered_specs, start=1):
            stats = tensor_to_stats[spec.tensor_name]
            mean_delta = stats.delta.mean()
            std_delta = stats.delta.std()
            mean_prob = stats.prob.mean()
            print(
                f"{rank:02d}. {spec.op_type:<8} {spec.node_name} ({spec.tensor_name}) -> "
                f"meanΔ={mean_delta:+.4f} stdΔ={std_delta:.4f} meanProb={mean_prob:.4f}"
            )
    else:
        print("\nTop contributors (sorted by drop):")
        for rank, spec in enumerate(ordered_specs, start=1):
            stats = tensor_to_stats[spec.tensor_name]
            delta = stats.delta.mean()
            prob = stats.prob.mean()
            print(f"{rank:02d}. {spec.op_type:<8} {spec.node_name} ({spec.tensor_name}) -> Δ={delta:+.4f}, prob={prob:.4f}")

    topk = max(0, args.topk_heatmaps)
    if topk and ordered_specs:
        top_specs = ordered_specs[: min(topk, len(ordered_specs))]
        tensor_names = [spec.tensor_name for spec in top_specs]
        heatmap_script = Path(args.heatmap_script)
        ref_image_path = Path(args.topk_image) if args.topk_image else image_paths[0]
        output_dir = Path(args.topk_output_dir)
        sys.stdout.flush()
        _render_topk_heatmaps(
            model_path,
            heatmap_script,
            ref_image_path,
            tensor_names,
            output_dir,
            args.topk_composite_name,
        )


if __name__ == "__main__":
    main()
