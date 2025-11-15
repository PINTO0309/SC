#!/usr/bin/env python3
"""Prepare training data by extracting frames and generating class labels."""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_CSV = ROOT / "ava_v2.2" / "ava_train_v2.2_seq.csv"
DEFAULT_VAL_CSV = ROOT / "ava_v2.2" / "ava_val_v2.2_seq.csv"
DEFAULT_VIDEO_DIRS = (ROOT / "data" / "trainval", ROOT / "data" / "test")
DEFAULT_IMAGE_DIR = ROOT / "data" / "images"
DEFAULT_ANNOTATION_FILE = ROOT / "data" / "annotation.txt"
DEFAULT_HISTOGRAM_FILE = ROOT / "data" / "image_size_hist.png"
DEFAULT_PIE_FILE = ROOT / "data" / "class_ratio.png"
DEFAULT_FFPROBE_BIN = "ffprobe"
DEFAULT_DETECTOR_MODEL = ROOT / "deimv2_dinov3_x_wholebody34_680query_n_batch_640x640.onnx"
SITTING_ACTION_ID = 11
MIN_DIMENSION = 6
MAX_DIMENSION = 750
IMAGES_PER_FOLDER = 1000
DETECTOR_INPUT_SIZE = 640
DETECTOR_BODY_LABEL = 0
DETECTOR_ABDOMEN_LABEL = 29
DETECTOR_HIP_LABEL = 30
DETECTOR_BODY_THRESHOLD = 0.35
DETECTOR_PART_THRESHOLD = 0.30


@dataclass
class PersonRecord:
    person_id: int
    has_sitting: bool = False
    box_sitting: tuple[float, float, float, float] | None = None
    has_other: bool = False
    box_other: tuple[float, float, float, float] | None = None

    @property
    def class_id(self) -> int:
        """Return 1 if the person has action_id 11, otherwise 0."""
        return 1 if self.has_sitting else 0

    def get_box(self) -> tuple[float, float, float, float] | None:
        """Return the bounding box that best matches the assigned class."""
        if self.class_id == 1:
            return self.box_sitting or self.box_other
        return self.box_other or self.box_sitting


@dataclass
class CropRecord:
    filename: str
    video_id: str
    timestamp: str
    person_id: int
    class_id: int
    width: int
    height: int
    path: Path | None


def read_annotation_rows(csv_paths: Iterable[Path]) -> dict[tuple[str, str], dict[int, PersonRecord]]:
    """Load annotation CSVs and aggregate per video/timestamp/person."""
    grouped: dict[tuple[str, str], dict[int, PersonRecord]] = {}
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found")
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row_no, row in enumerate(reader, start=1):
                if not row or len(row) < 8:
                    continue
                video_id = row[0].strip()
                timestamp = row[1].strip()
                try:
                    coords = tuple(float(value) for value in row[2:6])
                    action_id = int(row[6])
                    person_id = int(row[7])
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric data in {csv_path}:{row_no}") from exc

                key = (video_id, timestamp)
                bucket = grouped.setdefault(key, {})
                record = bucket.get(person_id)
                if record is None:
                    record = PersonRecord(person_id=person_id)
                    bucket[person_id] = record

                if action_id == SITTING_ACTION_ID:
                    record.has_sitting = True
                    record.box_sitting = coords
                else:
                    record.has_other = True
                    record.box_other = coords
    return grouped


def build_video_index(video_dirs: Iterable[Path]) -> dict[str, Path]:
    """Create a mapping of video_id (filename stem) to file path."""
    mapping: dict[str, Path] = {}
    for base_dir in video_dirs:
        if not base_dir.exists():
            continue
        for path in base_dir.iterdir():
            if path.is_file():
                mapping.setdefault(path.stem, path)
    return mapping


def clamp_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """Clamp normalized box coordinates to [0, 1] and ensure non-zero area."""
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    x2 = min(max(x2, 0.0), 1.0)
    y2 = min(max(y2, 0.0), 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def build_crop_filter(
    box: tuple[float, float, float, float],
) -> str:
    """Return an ffmpeg crop filter string for the normalized box."""
    x1, y1, x2, y2 = box
    crop_w = max(x2 - x1, 1e-6)
    crop_h = max(y2 - y1, 1e-6)
    return (
        f"crop=iw*{crop_w:.6f}:ih*{crop_h:.6f}:"
        f"iw*{x1:.6f}:ih*{y1:.6f}"
    )


def get_video_resolution(
    video_path: Path,
    ffprobe_bin: str,
    cache: dict[Path, tuple[int, int]],
) -> tuple[int, int] | None:
    """Return cached (width, height) for a video using ffprobe if needed."""
    if video_path in cache:
        return cache[video_path]

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"[warn] ffprobe failed for {video_path}: {result.stderr.strip()}", file=sys.stderr)
        return None

    line = result.stdout.strip()
    if "x" not in line:
        print(f"[warn] ffprobe output unexpected for {video_path}: {line}", file=sys.stderr)
        return None

    try:
        width_str, height_str = line.split("x")
        width = int(width_str)
        height = int(height_str)
    except ValueError:
        print(f"[warn] Unable to parse ffprobe output for {video_path}: {line}", file=sys.stderr)
        return None

    cache[video_path] = (width, height)
    return cache[video_path]


def estimate_pixel_dimensions(
    box: tuple[float, float, float, float],
    resolution: tuple[int, int],
) -> tuple[int, int]:
    """Estimate integer pixel width/height for a normalized box."""
    x1, y1, x2, y2 = box
    video_width, video_height = resolution
    width = max(int(round((x2 - x1) * video_width)), 1)
    height = max(int(round((y2 - y1) * video_height)), 1)
    return width, height


def load_detector_session(model_path: Path) -> tuple[ort.InferenceSession, str]:
    """Load the ONNX detector session with CUDA preference."""
    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                'trt_engine_cache_enable': True, # .engine, .profile export
                'trt_engine_cache_path': f'.',
                # 'trt_max_workspace_size': 4e9, # Maximum workspace size for TensorRT engine (1e9 ≈ 1GB)
                # onnxruntime>=1.21.0 breaking changes
                # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#data-dependant-shape-dds-ops
                # https://github.com/microsoft/onnxruntime/pull/22681/files
                # https://github.com/microsoft/onnxruntime/pull/23893/files
                'trt_op_types_to_exclude': 'NonMaxSuppression,NonZero,RoiAlign',
            },
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def detector_evaluate_crop(
    session: ort.InferenceSession,
    input_name: str,
    image_path: Path,
) -> tuple[int, bool, bool]:
    """Return detector stats (body count, abdomen present, hip present)."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Detector input missing: {image_path}")

    resized = cv2.resize(
        image,
        (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    blob = resized.transpose(2, 0, 1).astype(np.float32, copy=False)
    blob = np.expand_dims(blob, axis=0)
    detections = session.run(None, {input_name: blob})[0][0]

    body_count = 0
    abdomen_present = False
    hip_present = False
    for det in detections:
        label = int(round(det[0]))
        score = float(det[5])
        if label == DETECTOR_BODY_LABEL and score >= DETECTOR_BODY_THRESHOLD:
            body_count += 1
        elif label == DETECTOR_ABDOMEN_LABEL and score >= DETECTOR_PART_THRESHOLD:
            abdomen_present = True
        elif label == DETECTOR_HIP_LABEL and score >= DETECTOR_PART_THRESHOLD:
            hip_present = True

    return body_count, abdomen_present, hip_present


def rebalance_records(
    records: list[CropRecord],
    *,
    enabled: bool,
    seed: int | None,
) -> tuple[list[CropRecord], list[CropRecord]]:
    """Balance class distribution by downsampling the majority class."""
    if not enabled or not records:
        return records, []

    zeros = [record for record in records if record.class_id == 0]
    ones = [record for record in records if record.class_id == 1]
    if not zeros or not ones:
        return records, []

    rng = random.Random(seed)

    if len(zeros) > len(ones):
        rng.shuffle(zeros)
        kept_zero = zeros[: len(ones)]
        kept_one = ones
        dropped = zeros[len(ones) :]
    else:
        rng.shuffle(ones)
        kept_one = ones[: len(zeros)]
        kept_zero = zeros
        dropped = ones[len(zeros) :]

    balanced = kept_zero + kept_one
    balanced.sort(
        key=lambda rec: (
            rec.video_id,
            rec.timestamp,
            rec.person_id,
            rec.class_id,
        )
    )
    return balanced, dropped


def extract_cropped_frame(
    ffmpeg_bin: str,
    video_path: Path,
    timestamp: str,
    output_path: Path,
    *,
    overwrite: bool,
    box: tuple[float, float, float, float],
    min_dimension: int,
    max_dimension: int,
    skip_mmco_warning: bool,
) -> tuple[int, int] | None:
    """Extract a cropped person frame and return its (width, height) if kept."""
    seconds = float(timestamp)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            print(f"[skip] {output_path} already exists")
            return None

    crop_filter = build_crop_filter(box)
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{seconds:.3f}",
        "-i",
        str(video_path),
        "-vf",
        crop_filter,
        "-frames:v",
        "1",
        str(output_path),
    ]
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    stderr_text = result.stderr or ""
    stderr_lower = stderr_text.lower()
    if skip_mmco_warning and "mmco: unref short failure" in stderr_lower:
        output_path.unlink(missing_ok=True)
        return None

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path} at {seconds:.3f}s with {crop_filter}\n{stderr_text}"
        )

    with Image.open(output_path) as img:
        width, height = img.size
    if (
        width < min_dimension
        or height < min_dimension
        or width > max_dimension
        or height > max_dimension
    ):
        output_path.unlink(missing_ok=True)
        print(
            f"[skip] Removed {output_path} due to invalid size ({width}x{height})",
            file=sys.stderr,
        )
        return None

    return width, height


def iter_annotation_records(
    grouped_items: list[tuple[tuple[str, str], dict[int, PersonRecord]]],
    *,
    timestamp_stride: int,
) -> Iterator[tuple[str, str, PersonRecord]]:
    """Yield sorted annotation records with optional timestamp decimation."""
    stride = max(1, timestamp_stride)
    for index, ((video_id, timestamp), people) in enumerate(grouped_items):
        if stride > 1 and index % stride != 0:
            continue
        for person_id in sorted(people.keys()):
            yield video_id, timestamp, people[person_id]


def write_annotation_file(
    output_path: Path,
    records: list[tuple[str, str, str, str, str]],
) -> None:
    """Write the machine-learning annotation CSV."""
    if not records:
        print(f"[annotation] No rows to write for {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(records)
    print(f"[annotation] Wrote {output_path} ({len(records)} rows)")


def save_histogram(
    widths: list[int],
    heights: list[int],
    output_path: Path,
) -> None:
    """Plot stacked histograms for crop widths and heights."""
    sample_count = len(widths)
    if sample_count == 0:
        print("[hist] No crop dimension data available; skipping histogram")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].hist(heights, bins=50, color="#1f77b4", edgecolor="black")
    axes[0].set_title("Crop Height Distribution")
    axes[0].set_xlabel("Height (pixels)")
    axes[0].set_ylabel("Count")
    height_mean = statistics.mean(heights)
    height_median = statistics.median(heights)
    axes[0].axvline(height_mean, color="#d62728", linestyle="--", linewidth=2, label=f"Mean {height_mean:.1f}")
    axes[0].axvline(height_median, color="#2ca02c", linestyle="-.", linewidth=2, label=f"Median {height_median:.1f}")
    axes[0].legend()

    axes[1].hist(widths, bins=50, color="#ff7f0e", edgecolor="black")
    axes[1].set_title("Crop Width Distribution")
    axes[1].set_xlabel("Width (pixels)")
    axes[1].set_ylabel("Count")
    width_mean = statistics.mean(widths)
    width_median = statistics.median(widths)
    axes[1].axvline(width_mean, color="#d62728", linestyle="--", linewidth=2, label=f"Mean {width_mean:.1f}")
    axes[1].axvline(width_median, color="#2ca02c", linestyle="-.", linewidth=2, label=f"Median {width_median:.1f}")
    axes[1].legend()

    fig.suptitle("Cropped Image Dimensions")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[hist] Wrote {output_path} ({sample_count} samples)")


def save_class_ratio_chart(
    class_counts: Counter[int],
    output_path: Path,
) -> None:
    """Save a pie chart showing the classid distribution."""
    total = sum(class_counts.values())
    if total == 0:
        print("[pie] No class data available; skipping pie chart")
        return

    labels = []
    sizes = []
    for classid in sorted(class_counts.keys()):
        labels.append(f"classid={classid}")
        sizes.append(class_counts[classid])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sizes,
        labels=labels,
        labeldistance=0.6,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct * total / 100))})",
        pctdistance=0.8,
        textprops={"ha": "center", "va": "center"},
        startangle=90,
    )
    ax.set_title("Class Distribution")
    ax.axis("equal")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[pie] Wrote {output_path} ({total} samples)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-person frames from AVA videos while labeling whether "
            "action_id 11 (sitting) is present."
        )
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help="Train annotation CSV (default: ava_train_v2.2_seq.csv)",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=DEFAULT_VAL_CSV,
        help="Validation annotation CSV (default: ava_val_v2.2_seq.csv)",
    )
    parser.add_argument(
        "--video-dirs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_VIDEO_DIRS),
        help="Directories that contain the downloaded AVA videos.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Destination directory for extracted PNGs.",
    )
    parser.add_argument(
        "--images-per-folder",
        type=int,
        default=IMAGES_PER_FOLDER,
        help="Number of images to store in each sequential subfolder (default: 1000).",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=DEFAULT_ANNOTATION_FILE,
        help="Output CSV with filename,video_id,timestamp,person_id,classid.",
    )
    parser.add_argument(
        "--histogram-file",
        type=Path,
        default=DEFAULT_HISTOGRAM_FILE,
        help="PNG path for stacked height/width histograms.",
    )
    parser.add_argument(
        "--class-ratio-file",
        type=Path,
        default=DEFAULT_PIE_FILE,
        help="PNG path for a pie chart showing classid distribution.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="Path to the ffmpeg executable (default: ffmpeg).",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default=DEFAULT_FFPROBE_BIN,
        help="Path to the ffprobe executable (default: ffprobe).",
    )
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=DEFAULT_DETECTOR_MODEL,
        help="ONNX detector used to validate classid=1 crops (default: deimv2...640.onnx).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned operations without writing images or CSVs.",
    )
    parser.add_argument(
        "--min-dimension",
        type=int,
        default=MIN_DIMENSION,
        help="Minimum width/height (pixels) required to keep a crop (default: 6).",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=MAX_DIMENSION,
        help="Maximum width/height (pixels); crops larger than this are skipped (default: 750).",
    )
    parser.add_argument(
        "--skip-mmco-warning",
        action="store_true",
        default=True,
        help="Skip crops whose ffmpeg stderr contains 'mmco: unref short failure' (default: enabled).",
    )
    parser.add_argument(
        "--no-skip-mmco-warning",
        action="store_false",
        dest="skip_mmco_warning",
        help="Allow crops even if ffmpeg reports 'mmco: unref short failure'.",
    )
    parser.add_argument(
        "--timestamp-stride",
        type=int,
        default=1,
        help="Keep every Nth (video_id, timestamp) group to downsample data (default: 1, keep all).",
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        dest="balance_classes",
        help="(Deprecated) Final class balancing is enabled by default.",
    )
    parser.add_argument(
        "--no-balance-classes",
        action="store_false",
        dest="balance_classes",
        help="Disable final class balancing.",
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=None,
        help="Random seed used when --balance-classes is enabled.",
    )
    parser.set_defaults(balance_classes=True)
    args = parser.parse_args()
    if args.max_dimension < args.min_dimension:
        parser.error("--max-dimension must be greater than or equal to --min-dimension")
    return args


def main() -> None:
    args = parse_args()
    min_dimension = max(1, args.min_dimension)
    max_dimension = max(min_dimension, args.max_dimension)

    grouped = read_annotation_rows([args.train_csv, args.val_csv])
    print(f"[info] Aggregated {len(grouped)} (video_id, timestamp) groups.")
    grouped_items = sorted(grouped.items())

    video_index = build_video_index(args.video_dirs)
    if not video_index:
        print("[error] No video files found in the provided directories.", file=sys.stderr)
        sys.exit(1)

    annotation_records: list[tuple[str, str, str, str, str]] = []
    missing_videos: set[str] = set()
    collected_widths: list[int] = []
    collected_heights: list[int] = []
    class_counts: Counter[int] = Counter()
    images_per_folder = max(1, args.images_per_folder)
    generated_count = 0
    current_folder: Path | None = None
    resolution_cache: dict[Path, tuple[int, int]] = {}
    detector_session: ort.InferenceSession | None = None
    detector_input_name: str | None = None
    model_path = args.detector_model
    if not model_path.exists():
        print(f"[error] Detector model not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    try:
        detector_session, detector_input_name = load_detector_session(model_path)
        print(f"[info] Loaded detector model from {model_path}")
    except Exception as exc:
        print(f"[error] Failed to load detector model: {exc}", file=sys.stderr)
        sys.exit(1)

    stride = max(1, args.timestamp_stride)
    input_records = list(iter_annotation_records(grouped_items, timestamp_stride=stride))
    if not input_records:
        print("[warn] No annotation records remain after timestamp stride filtering.", file=sys.stderr)
        return

    progress = tqdm(
        input_records,
        total=len(input_records),
        desc="Processing",
        unit="person",
    )

    detector_skipped = 0
    processed_records: list[CropRecord] = []

    for video_id, timestamp, person_record in progress:
        video_path = video_index.get(video_id)
        if video_path is None:
            missing_videos.add(video_id)
            continue

        filename = f"{video_id}_{timestamp}_{person_record.person_id:05d}_{person_record.class_id}.png"
        raw_box = person_record.get_box()
        clamped_box = clamp_box(raw_box) if raw_box else None
        if clamped_box is None:
            print(
                f"[warn] Invalid bounding box for {video_id} {timestamp} pid={person_record.person_id}",
                file=sys.stderr,
            )
            continue

        resolution = get_video_resolution(video_path, args.ffprobe_bin, resolution_cache)
        if resolution is None:
            continue

        width_px, height_px = estimate_pixel_dimensions(clamped_box, resolution)
        if (
            width_px < min_dimension
            or height_px < min_dimension
            or width_px > max_dimension
            or height_px > max_dimension
        ):
            continue

        temp_output = False
        if args.dry_run:
            fd, temp_name = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            output_path = Path(temp_name)
            try:
                output_path.unlink()
            except FileNotFoundError:
                pass
            temp_output = True
        else:
            if current_folder is None or generated_count % images_per_folder == 0:
                folder_index = generated_count // images_per_folder + 1
                folder_name = f"{folder_index:04d}"
                current_folder = args.image_dir / folder_name
                current_folder.mkdir(parents=True, exist_ok=True)
            output_path = current_folder / filename

        try:
            dims = extract_cropped_frame(
                args.ffmpeg_bin,
                video_path,
                timestamp,
                output_path,
                overwrite=args.overwrite,
                box=clamped_box,
                min_dimension=min_dimension,
                max_dimension=max_dimension,
                skip_mmco_warning=args.skip_mmco_warning,
            )
        except Exception as exc:
            print(f"[warn] Failed to process {video_id} {timestamp} pid={person_record.person_id}: {exc}", file=sys.stderr)
            if temp_output:
                output_path.unlink(missing_ok=True)
            continue
        if dims is None:
            if temp_output:
                output_path.unlink(missing_ok=True)
            continue

        assert detector_session is not None and detector_input_name is not None
        try:
            body_count, abdomen_present, hip_present = detector_evaluate_crop(
                detector_session,
                detector_input_name,
                output_path,
            )
        except Exception as exc:
            print(
                f"[warn] Detector check failed for {output_path}: {exc}",
                file=sys.stderr,
            )
            output_path.unlink(missing_ok=True)
            continue

        if person_record.class_id == 1:
            keep_crop = body_count == 1 and (abdomen_present or hip_present)
        else:
            keep_crop = body_count <= 1

        if not keep_crop:
            output_path.unlink(missing_ok=True)
            detector_skipped += 1
            continue

        width_px, height_px = dims
        processed_records.append(
            CropRecord(
                filename=filename,
                video_id=video_id,
                timestamp=timestamp,
                person_id=person_record.person_id,
                class_id=person_record.class_id,
                width=width_px,
                height=height_px,
                path=None if args.dry_run else output_path,
            )
        )

        if temp_output:
            output_path.unlink(missing_ok=True)

        if args.dry_run:
            continue

        generated_count += 1

    if missing_videos:
        sample = ", ".join(sorted(missing_videos)[:5])
        print(
            f"[warn] Missing {len(missing_videos)} videos referenced by annotations. Examples: {sample}",
            file=sys.stderr,
        )

    balanced_records, dropped_records = rebalance_records(
        processed_records, enabled=args.balance_classes, seed=args.balance_seed
    )

    if dropped_records and not args.dry_run:
        for record in dropped_records:
            if record.path:
                record.path.unlink(missing_ok=True)

    collected_widths = [record.width for record in balanced_records]
    collected_heights = [record.height for record in balanced_records]
    class_counts = Counter(record.class_id for record in balanced_records)

    if not args.dry_run:
        annotation_records = [
            (record.filename, record.video_id, record.timestamp, str(record.person_id), str(record.class_id))
            for record in balanced_records
        ]
        write_annotation_file(args.annotation_file, annotation_records)
    else:
        print(f"[annotation] Dry run: skipping write to {args.annotation_file}")

    save_histogram(collected_widths, collected_heights, args.histogram_file)
    save_class_ratio_chart(class_counts, args.class_ratio_file)
    if detector_skipped:
        print(f"[detector] Skipped {detector_skipped} crops after detector filtering.")
    if dropped_records:
        print(f"[balance] Removed {len(dropped_records)} crops to rebalance classes.")


if __name__ == "__main__":
    main()
