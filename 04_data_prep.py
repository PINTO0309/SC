#!/usr/bin/env python3
"""Prepare training data by extracting frames and generating class labels."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_CSV = ROOT / "ava_v2.2" / "ava_train_v2.2_seq.csv"
DEFAULT_VAL_CSV = ROOT / "ava_v2.2" / "ava_val_v2.2_seq.csv"
DEFAULT_VIDEO_DIRS = (ROOT / "data" / "trainval", ROOT / "data" / "test")
DEFAULT_IMAGE_DIR = ROOT / "data" / "images"
IMAGES_PER_FOLDER = 1000
DEFAULT_ANNOTATION_FILE = ROOT / "data" / "annotation.txt"
DEFAULT_HISTOGRAM_FILE = ROOT / "data" / "image_size_hist.png"
DEFAULT_PIE_FILE = ROOT / "data" / "class_ratio.png"
SITTING_ACTION_ID = 11
MIN_DIMENSION = 6


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


def extract_cropped_frame(
    ffmpeg_bin: str,
    video_path: Path,
    timestamp: str,
    output_path: Path,
    *,
    overwrite: bool,
    dry_run: bool,
    box: tuple[float, float, float, float],
    min_dimension: int,
    skip_mmco_warning: bool,
) -> tuple[int, int] | None:
    """Extract a cropped person frame and return its (width, height) if kept."""
    seconds = float(timestamp)
    if dry_run:
        print(f"[ffmpeg] {video_path} @ {seconds:.3f}s crop {box} -> {output_path}")
        return (0, 0)

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
    if width < min_dimension or height < min_dimension:
        output_path.unlink(missing_ok=True)
        print(
            f"[skip] Removed {output_path} due to small size ({width}x{height})",
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
    *,
    dry_run: bool,
) -> None:
    """Write the machine-learning annotation CSV."""
    if dry_run:
        print(f"[annotation] Would write {output_path} with {len(records)} rows")
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
    *,
    dry_run: bool,
) -> None:
    """Plot stacked histograms for crop widths and heights."""
    sample_count = len(widths)
    if sample_count == 0:
        print("[hist] No crop dimension data available; skipping histogram")
        return

    if dry_run:
        print(f"[hist] Would write {output_path} (samples: {sample_count})")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].hist(heights, bins=50, color="#1f77b4", edgecolor="black")
    axes[0].set_title("Crop Height Distribution")
    axes[0].set_xlabel("Height (pixels)")
    axes[0].set_ylabel("Count")

    axes[1].hist(widths, bins=50, color="#ff7f0e", edgecolor="black")
    axes[1].set_title("Crop Width Distribution")
    axes[1].set_xlabel("Width (pixels)")
    axes[1].set_ylabel("Count")

    fig.suptitle("Cropped Image Dimensions")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[hist] Wrote {output_path} ({sample_count} samples)")


def save_class_ratio_chart(
    class_counts: Counter[int],
    output_path: Path,
    *,
    dry_run: bool,
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

    if dry_run:
        label_info = ", ".join(f"{label}:{count}" for label, count in zip(labels, sizes))
        print(f"[pie] Would write {output_path} ({label_info})")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct * total / 100))})",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    min_dimension = max(1, args.min_dimension)

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

    stride = max(1, args.timestamp_stride)
    total_records = sum(
        len(bucket)
        for index, (_, bucket) in enumerate(grouped_items)
        if stride == 1 or index % stride == 0
    )
    if total_records == 0:
        print("[warn] No annotation records remain after timestamp stride filtering.", file=sys.stderr)
        return

    progress = tqdm(
        iter_annotation_records(grouped_items, timestamp_stride=stride),
        total=total_records,
        desc="Processing",
        unit="person",
    )

    for video_id, timestamp, person_record in progress:
        video_path = video_index.get(video_id)
        if video_path is None:
            missing_videos.add(video_id)
            continue

        if current_folder is None or generated_count % images_per_folder == 0:
            folder_index = generated_count // images_per_folder + 1
            folder_name = f"{folder_index:04d}"
            current_folder = args.image_dir / folder_name
            if not args.dry_run:
                current_folder.mkdir(parents=True, exist_ok=True)

        filename = f"{video_id}_{timestamp}_{person_record.person_id:05d}_{person_record.class_id}.png"
        output_path = current_folder / filename
        raw_box = person_record.get_box()
        clamped_box = clamp_box(raw_box) if raw_box else None
        if clamped_box is None:
            print(
                f"[warn] Invalid bounding box for {video_id} {timestamp} pid={person_record.person_id}",
                file=sys.stderr,
            )
            continue

        try:
            dims = extract_cropped_frame(
                args.ffmpeg_bin,
                video_path,
                timestamp,
                output_path,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                box=clamped_box,
                min_dimension=min_dimension,
                skip_mmco_warning=args.skip_mmco_warning,
            )
        except Exception as exc:
            print(f"[warn] Failed to process {video_id} {timestamp} pid={person_record.person_id}: {exc}", file=sys.stderr)
            continue
        if dims is None:
            continue

        if not args.dry_run:
            width, height = dims
            collected_widths.append(width)
            collected_heights.append(height)

        annotation_records.append(
            (filename, video_id, timestamp, str(person_record.person_id), str(person_record.class_id))
        )
        class_counts[person_record.class_id] += 1
        generated_count += 1

    if missing_videos:
        sample = ", ".join(sorted(missing_videos)[:5])
        print(
            f"[warn] Missing {len(missing_videos)} videos referenced by annotations. Examples: {sample}",
            file=sys.stderr,
        )

    write_annotation_file(args.annotation_file, annotation_records, dry_run=args.dry_run)
    save_histogram(collected_widths, collected_heights, args.histogram_file, dry_run=args.dry_run)
    save_class_ratio_chart(class_counts, args.class_ratio_file, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
