#!/usr/bin/env python3
"""Export Sigmoid->Mul feature maps from sc_c_32x24.onnx as heatmap PNG images."""

from __future__ import annotations

import argparse
import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnx import helper, shape_inference  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

PNG_SCALE = 0.3  # final PNG dimensions will be scaled by this factor


@dataclass(frozen=True)
class FeatureTap:
    """Describes a Sigmoid->Mul activation we want to visualize."""

    sigmoid_name: str
    mul_name: str
    tensor_name: str
    producer_name: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run sc_c_32x24.onnx, capture Conv->Sigmoid->Mul feature maps, and save them as heatmap PNGs."
        )
    )
    parser.add_argument("--model", default="sc_c_32x24.onnx", help="Path to the ONNX model to inspect.")
    parser.add_argument("--image", help="Path to the RGB image used as model input.")
    parser.add_argument("--output-dir", default="feature_heatmaps", help="Directory where PNGs will be stored.")
    parser.add_argument("--layers", nargs="*", help="Optional substrings to filter which feature maps are exported.")
    parser.add_argument("--limit", type=int, help="Optional limit on the number of feature maps to export.")
    parser.add_argument(
        "--reduce",
        choices=("mean", "max", "sum"),
        default="mean",
        help="Channel reduction type applied before turning the feature map into a heatmap.",
    )
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
        help="Per-channel mean used during normalization (values are in [0,1]).",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(1.0, 1.0, 1.0),
        help="Per-channel std used during normalization (values are in [0,1]).",
    )
    parser.add_argument("--alpha", type=float, default=0.55, help="Blend factor for overlay images.")
    parser.add_argument("--cmap", default="turbo", help="Matplotlib colormap name used for heatmaps.")
    parser.add_argument(
        "--invert-cmap",
        dest="invert_cmap",
        action="store_true",
        help="Invert heatmap colours (bright regions mark high activations).",
    )
    parser.add_argument(
        "--no-invert-cmap",
        dest="invert_cmap",
        action="store_false",
        help="Disable the default colour inversion (bright = low activations).",
    )
    parser.add_argument(
        "--composite-topk",
        type=int,
        default=0,
        help="When >0, stitch the top-K heatmaps (by average intensity) into a 3x2 PNG.",
    )
    parser.add_argument(
        "--composite-output",
        default="top_features_composite.png",
        help="Filename for the stitched PNG (relative to --output-dir).",
    )
    parser.add_argument(
        "--composite-sort",
        choices=("intensity", "order"),
        default="intensity",
        help="Sort strategy for composite tiles (default: by heatmap intensity).",
    )
    parser.add_argument(
        "--composite-layout",
        choices=("row", "col"),
        default="row",
        help="Tile filling order: row-major (default) or column-major.",
    )
    parser.add_argument("--list-layers", action="store_true", help="Only list available taps and exit.")
    parser.add_argument("--providers", nargs="*", help="Optional onnxruntime providers override.")
    parser.set_defaults(invert_cmap=False)
    return parser.parse_args()


def _load_model(model_path: Path) -> onnx.ModelProto:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {model_path}")
    return onnx.load(model_path.as_posix())


def _build_consumers(model: onnx.ModelProto) -> Dict[str, List[onnx.NodeProto]]:
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
    return consumers


def _build_producers(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
    producers: Dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            producers[out] = node
    return producers


def _collect_sigmoid_mul_pairs(model: onnx.ModelProto) -> List[FeatureTap]:
    consumers = _build_consumers(model)
    producers = _build_producers(model)
    taps: List[FeatureTap] = []
    for node in model.graph.node:
        if node.op_type != "Sigmoid":
            continue
        sigmoid_out = node.output[0]
        mul_candidates = [consumer for consumer in consumers.get(sigmoid_out, []) if consumer.op_type == "Mul"]
        if not mul_candidates:
            continue
        mul_node = mul_candidates[0]
        source_tensor = node.input[0] if node.input else ""
        producer_name = producers.get(source_tensor).name if source_tensor in producers else source_tensor
        taps.append(
            FeatureTap(
                sigmoid_name=node.name or sigmoid_out,
                mul_name=mul_node.name or mul_node.output[0],
                tensor_name=mul_node.output[0],
                producer_name=producer_name,
            )
        )
    return taps


def _filter_taps(taps: Sequence[FeatureTap], filters: Sequence[str] | None, limit: int | None) -> List[FeatureTap]:
    filtered: Iterable[FeatureTap] = taps
    if filters:
        lowered = [token.lower() for token in filters]
        filtered = [
            tap
            for tap in taps
            if any(token in tap.tensor_name.lower() or token in tap.producer_name.lower() for token in lowered)
        ]
    filtered_list = list(filtered)
    if limit is not None:
        filtered_list = filtered_list[: max(limit, 0)]
    return filtered_list


def _reorder_taps_by_filters(taps: Sequence[FeatureTap], filters: Sequence[str] | None) -> List[FeatureTap]:
    if not filters:
        return list(taps)
    lowered_filters = [token.lower() for token in filters]
    ordered: List[FeatureTap] = []
    used: set[int] = set()
    def matches(token: str, tap: FeatureTap) -> bool:
        t_name = tap.tensor_name.lower()
        return token == t_name or token in t_name or token in (tap.mul_name.lower() if tap.mul_name else "")
    for token in lowered_filters:
        for idx, tap in enumerate(taps):
            if idx in used:
                continue
            if matches(token, tap):
                ordered.append(tap)
                used.add(idx)
                break
    for idx, tap in enumerate(taps):
        if idx not in used:
            ordered.append(tap)
    return ordered


def _infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return shape_inference.infer_shapes(model, strict_mode=False)
    except Exception as exc:  # pragma: no cover - inference failures are unlikely but should not stop us
        print(f"[WARN] Failed to run shape inference: {exc}. Continuing without inferred shapes.")
        return model


def _tensor_value_info_map(model: onnx.ModelProto) -> Dict[str, onnx.ValueInfoProto]:
    mapping: Dict[str, onnx.ValueInfoProto] = {}
    for value_info in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        mapping[value_info.name] = value_info
    return mapping


def _append_value_infos(model: onnx.ModelProto, tensor_names: Sequence[str]) -> onnx.ModelProto:
    augmented = copy.deepcopy(model)
    existing_outputs = {out.name for out in augmented.graph.output}
    value_infos = _tensor_value_info_map(augmented)
    for name in tensor_names:
        if name in existing_outputs:
            continue
        value_info = value_infos.get(name)
        if value_info is None:
            raise ValueError(f"Could not locate type information for tensor {name!r}.")
        tensor_type = value_info.type.tensor_type
        elem_type = tensor_type.elem_type
        dims = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value)
            elif dim.dim_param:
                dims.append(dim.dim_param)
            else:
                dims.append(None)
        shape = dims if any(d is not None for d in dims) else None
        new_value_info = helper.make_tensor_value_info(name, elem_type, shape)
        augmented.graph.output.append(new_value_info)
        existing_outputs.add(name)
    return augmented


def _parse_input_size(value: str | int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise argparse.ArgumentTypeError("Input size tuple must contain (height, width).")
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
            raise argparse.ArgumentTypeError("Use the HxW format (e.g. 32x24).")
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


def _load_image(path: Path, input_size: Tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = input_size
    if height <= 0 or width <= 0:
        raise ValueError("Input size must be positive.")
    image = Image.open(path).convert("RGB")
    resized = image.resize((width, height), Image.BILINEAR)
    resized_arr = np.asarray(resized).astype(np.float32) / 255.0
    original_arr = np.asarray(image).astype(np.float32) / 255.0
    return resized_arr, original_arr


def _prepare_tensor(image_arr: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if not np.all(std):
        raise ValueError("Std values must be non-zero.")
    normalized = (image_arr - mean) / std
    tensor = normalized.transpose(2, 0, 1)[None, ...].astype(np.float32)
    return tensor


def _select_providers(user_providers: Sequence[str] | None) -> List[str]:
    available = ort.get_available_providers()
    if user_providers:
        selected = [provider for provider in user_providers if provider in available]
    else:
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        selected = [provider for provider in preferred if provider in available]
    return selected or available


def _build_session(model: onnx.ModelProto, providers: Sequence[str]) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    serialized = model.SerializeToString()
    return ort.InferenceSession(serialized, sess_options=session_options, providers=list(providers))


def _reduce_feature_map(feature: np.ndarray, mode: str) -> np.ndarray:
    if feature.ndim == 4:
        feature = np.squeeze(feature, axis=0)
    if feature.ndim == 3:
        if mode == "mean":
            reduced = feature.mean(axis=0)
        elif mode == "max":
            reduced = feature.max(axis=0)
        else:
            reduced = feature.sum(axis=0)
    elif feature.ndim == 2:
        reduced = feature
    else:
        raise ValueError(f"Unsupported feature shape {feature.shape}.")
    return reduced


def _normalize_heatmap(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = array - array.min()
    eps = 1e-6
    denom = array.max()
    if denom < eps:
        return np.zeros_like(array)
    return array / denom


def _slugify(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return safe or "feature"


def _save_heatmap(
    output_dir: Path,
    base_image: np.ndarray,
    original_image: np.ndarray,
    heatmap_2d: np.ndarray,
    tap: FeatureTap,
    idx: int,
    cmap_name: str,
    alpha: float,
    invert: bool,
) -> Tuple[Path, float]:
    normalized = _normalize_heatmap(heatmap_2d)
    if invert:
        normalized = 1.0 - normalized
    cmap = colormaps[cmap_name]
    heat_rgb = cmap(normalized)[..., :3]
    resized_heat = _resize_like(heat_rgb, base_image)
    overlay = np.clip(alpha * resized_heat + (1.0 - alpha) * base_image, 0.0, 1.0)
    upscaled_overlay = _resize_like(overlay, original_image)
    upscaled_heat = _resize_like(resized_heat, original_image)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(upscaled_heat)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(upscaled_overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    title = f"{idx:03d} | {tap.mul_name}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    filename = f"{idx:03d}_{_slugify(tap.mul_name)}.png"
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    if PNG_SCALE != 1.0:
        with Image.open(output_path) as saved_img:
            width, height = saved_img.size
            new_w = max(1, int(width * PNG_SCALE))
            new_h = max(1, int(height * PNG_SCALE))
            resized_img = saved_img.resize((new_w, new_h), Image.LANCZOS)
            resized_img.save(output_path)

    return output_path, float(normalized.mean())


def _resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_h, ref_w = reference.shape[:2]
    clipped = np.clip(image, 0.0, 1.0)
    pil_image = Image.fromarray((clipped * 255).astype(np.uint8))
    resized = pil_image.resize((ref_w, ref_h), Image.BICUBIC)
    return np.asarray(resized).astype(np.float32) / 255.0


def _create_composite(
    records: Sequence[Tuple[Path, float]],
    output_path: Path,
    topk: int,
    cols: int = 2,
    sort_mode: str = "intensity",
    layout: str = "row",
    cmap_name: str = "turbo",
    invert: bool = True,
    legend_height: int = 50,
) -> None:
    if topk <= 0 or not records:
        return
    if sort_mode == "order":
        selected = list(records[:topk])
    else:
        selected = sorted(records, key=lambda item: item[1], reverse=True)[:topk]
    images = [Image.open(path).convert("RGB") for path, _ in selected]
    try:
        if not images:
            return
        tile_w, tile_h = images[0].size
        cols = max(1, cols)
        rows = math.ceil(len(images) / cols)
        canvas = Image.new("RGB", (tile_w * cols, tile_h * rows), color=(0, 0, 0))
        for idx, img in enumerate(images):
            if layout == "col":
                col = idx // rows
                row = idx % rows
            else:
                row, col = divmod(idx, cols)
            canvas.paste(img, (col * tile_w, row * tile_h))
        legend = _build_legend(tile_w * cols, legend_height, cmap_name, invert)
        final = Image.new(
            "RGB",
            (canvas.width, legend.height + canvas.height),
            color=(0, 0, 0),
        )
        final.paste(legend, (0, 0))
        final.paste(canvas, (0, legend.height))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final.save(output_path)
        print(f"[COMPOSITE] Saved top-{len(images)} heatmaps -> {output_path}")
    finally:
        for img in images:
            img.close()


def _build_legend(width: int, height: int, cmap_name: str, invert: bool) -> Image.Image:
    width = max(1, width)
    height = max(30, height)
    gradient = np.linspace(0.0, 1.0, width, dtype=np.float32)
    if invert:
        gradient = 1.0 - gradient
    cmap = colormaps[cmap_name]
    colors = cmap(gradient)[..., :3]
    bar = np.tile(colors, (height, 1, 1))
    legend = Image.fromarray((bar * 255).astype(np.uint8))
    draw = ImageDraw.Draw(legend)
    font = ImageFont.load_default()
    text_low = "Low impact"
    text_high = "High impact"
    text_color = (255, 255, 255)
    shadow = (0, 0, 0)
    padding = 6
    def _measure(text: str) -> Tuple[int, int]:
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        return font.getsize(text)

    low_width, low_height = _measure(text_low)
    high_width, high_height = _measure(text_high)
    low_pos = (padding, height - low_height - padding)
    high_pos = (width - high_width - padding, height - high_height - padding)
    for dx, dy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        draw.text((low_pos[0] + dx, low_pos[1] + dy), text_low, fill=shadow, font=font)
        draw.text((high_pos[0] + dx, high_pos[1] + dy), text_high, fill=shadow, font=font)
    draw.text(low_pos, text_low, fill=text_color, font=font)
    draw.text(high_pos, text_high, fill=text_color, font=font)
    return legend


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model)
    base_model = _load_model(model_path)
    taps = _collect_sigmoid_mul_pairs(base_model)
    if args.list_layers:
        for idx, tap in enumerate(taps):
            print(f"[{idx:03d}] sigmoid={tap.sigmoid_name} mul={tap.mul_name} tensor={tap.tensor_name}")
        return

    if args.image is None:
        raise ValueError("Argument --image is required unless --list-layers is specified.")
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_taps = _filter_taps(taps, args.layers, args.limit)
    selected_taps = _reorder_taps_by_filters(selected_taps, args.layers)
    if not selected_taps:
        raise RuntimeError("No feature maps matched the requested filters.")

    inferred_model = _infer_shapes(base_model)
    augmented_model = _append_value_infos(inferred_model, [tap.tensor_name for tap in selected_taps])
    providers = _select_providers(args.providers)
    session = _build_session(augmented_model, providers)

    resized_image, original_image = _load_image(image_path, args.input_size)
    input_tensor = _prepare_tensor(resized_image, args.mean, args.std)
    feeds = {session.get_inputs()[0].name: input_tensor}

    output_names = [tap.tensor_name for tap in selected_taps]
    features = session.run(output_names, feeds)
    intensity_records: List[Tuple[Path, float]] = []

    for idx, (tap, feature) in enumerate(zip(selected_taps, features), start=1):
        reduced = _reduce_feature_map(feature, args.reduce)
        saved_path, intensity = _save_heatmap(
            output_dir,
            resized_image,
            original_image,
            reduced,
            tap,
            idx,
            args.cmap,
            args.alpha,
            args.invert_cmap,
        )
        intensity_records.append((saved_path, intensity))
        print(f"[OK] Saved {tap.mul_name} -> {saved_path}")

    # Run the main model output for reference.
    final_output_name = base_model.graph.output[-1].name
    prob = session.run([final_output_name], feeds)[0]
    print(f"Model prob_sitting={prob.squeeze():.4f}")

    if args.composite_topk > 0 and intensity_records:
        _create_composite(
            intensity_records,
            output_dir / args.composite_output,
            topk=max(1, args.composite_topk),
            sort_mode=args.composite_sort,
            layout=args.composite_layout,
            cols=2,
            cmap_name=args.cmap,
            invert=args.invert_cmap,
            legend_height=60,
        )


if __name__ == "__main__":
    main()
