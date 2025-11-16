#!/usr/bin/env python3
"""Merge multiple parquet files from the ./data directory into a single parquet dataset."""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "merged_dataset.parquet"
DEFAULT_HISTOGRAM_FILE = DEFAULT_DATA_DIR / "image_size_hist_merged.png"
DEFAULT_CLASS_RATIO_FILE = DEFAULT_DATA_DIR / "class_ratio_merged.png"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate multiple parquet files placed under the data directory."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Parquet files to merge. Relative paths are resolved from the data directory.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing parquet files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination parquet file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--histogram-file",
        type=Path,
        default=DEFAULT_HISTOGRAM_FILE,
        help=f"Output PNG for merged image size histogram (default: {DEFAULT_HISTOGRAM_FILE})",
    )
    parser.add_argument(
        "--class-ratio-file",
        type=Path,
        default=DEFAULT_CLASS_RATIO_FILE,
        help=f"Output PNG for merged class ratio pie chart (default: {DEFAULT_CLASS_RATIO_FILE})",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Base directory used to resolve relative image paths (default: --base-dir).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output parquet file.",
    )
    return parser.parse_args()


def _resolve_path(candidate: str, base_dir: Path) -> Path:
    path = Path(candidate)
    search_paths: list[Path] = []
    if path.exists():
        search_paths.append(path.resolve())
    if not path.is_absolute():
        candidate_path = (base_dir / path).resolve()
        if candidate_path.exists():
            search_paths.append(candidate_path)
    if not search_paths:
        raise FileNotFoundError(f"Input parquet not found: {path}")
    resolved = search_paths[0]
    if resolved.suffix.lower() != ".parquet":
        raise ValueError(f"Expected a parquet file, but got: {resolved}")
    return resolved


def merge_parquet_files(input_paths: list[Path]) -> pd.DataFrame:
    if not input_paths:
        raise ValueError("At least one parquet file must be provided.")
    frames: list[pd.DataFrame] = []
    for path in input_paths:
        df = pd.read_parquet(path)
        if df.empty:
            # Keep empty frames to preserve schema, but warn so the user knows.
            print(f"[warn] {path} is empty; keeping schema only.")
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True, copy=False)
    return merged


def _resolve_output_path(path: Path, base_dir: Path) -> Path:
    resolved = path if path.is_absolute() else base_dir / path
    return resolved.resolve()


def _image_size_from_bytes(raw: object) -> tuple[int, int] | None:
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        return None
    data = bytes(raw)
    if not data:
        return None
    try:
        with Image.open(BytesIO(data)) as img:
            return img.width, img.height
    except (UnidentifiedImageError, OSError):
        return None


def _image_size_from_path(path_value: object, image_root: Path) -> tuple[int, int] | None:
    if path_value is None:
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = image_root / path
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except (UnidentifiedImageError, OSError):
        return None


def collect_image_dimensions(df: pd.DataFrame, image_root: Path) -> tuple[list[int], list[int]]:
    widths: list[int] = []
    heights: list[int] = []
    has_bytes = "image_bytes" in df.columns
    has_paths = "image_path" in df.columns
    if not (has_bytes or has_paths):
        return widths, heights

    for row in df.itertuples(index=False):
        size: tuple[int, int] | None = None
        if has_bytes:
            size = _image_size_from_bytes(getattr(row, "image_bytes"))
        if size is None and has_paths:
            size = _image_size_from_path(getattr(row, "image_path"), image_root)
        if size is None:
            continue
        width, height = size
        widths.append(width)
        heights.append(height)
    return widths, heights


def save_histogram(widths: list[int], heights: list[int], output_path: Path) -> None:
    if not widths or not heights:
        print("[hist] Skipping histogram; no image dimensions available.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    def _create_stat_annotator(axis):
        def annotate(x_value: float, label: str) -> None:
            ymin, ymax = axis.get_ylim()
            y = ymin + (ymax - ymin) * 0.95
            axis.text(
                x_value,
                y,
                label,
                rotation=90,
                va="top",
                ha="center",
                fontsize=9,
                color="black",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )

        return annotate

    axes[0].hist(heights, bins=50, color="#1f77b4", edgecolor="black")
    axes[0].set_title("Image Height Distribution")
    axes[0].set_xlabel("Height (pixels)")
    axes[0].set_ylabel("Count")
    height_mean = statistics.mean(heights)
    height_median = statistics.median(heights)
    axes[0].axvline(height_mean, color="#d62728", linestyle="--", linewidth=2, label="Mean")
    axes[0].axvline(height_median, color="#2ca02c", linestyle="-.", linewidth=2, label="Median")
    annotate_height = _create_stat_annotator(axes[0])
    annotate_height(height_mean, f"Mean {height_mean:.1f}")
    annotate_height(height_median, f"Median {height_median:.1f}")
    axes[0].legend()

    axes[1].hist(widths, bins=50, color="#ff7f0e", edgecolor="black")
    axes[1].set_title("Image Width Distribution")
    axes[1].set_xlabel("Width (pixels)")
    axes[1].set_ylabel("Count")
    width_mean = statistics.mean(widths)
    width_median = statistics.median(widths)
    axes[1].axvline(width_mean, color="#d62728", linestyle="--", linewidth=2, label="Mean")
    axes[1].axvline(width_median, color="#2ca02c", linestyle="-.", linewidth=2, label="Median")
    annotate_width = _create_stat_annotator(axes[1])
    annotate_width(width_mean, f"Mean {width_mean:.1f}")
    annotate_width(width_median, f"Median {width_median:.1f}")
    axes[1].legend()

    fig.suptitle("Merged Image Dimensions")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[hist] Wrote {output_path} ({len(widths)} samples)")


def save_class_ratio_chart(class_counts: Counter[int], output_path: Path) -> None:
    total = sum(class_counts.values())
    if total == 0:
        print("[pie] Skipping class ratio chart; no class_id values found.")
        return
    labels = [f"class_id={class_id}" for class_id in sorted(class_counts.keys())]
    sizes = [class_counts[class_id] for class_id in sorted(class_counts.keys())]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct * total / 100))})",
        pctdistance=0.8,
        labeldistance=0.6,
        textprops={"ha": "center", "va": "center"},
        startangle=90,
    )
    for text in texts:
        x, y = text.get_position()
        text.set_position((x, y + 0.1))
    ax.set_title("Merged Class Distribution")
    ax.axis("equal")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[pie] Wrote {output_path} ({total} samples)")


def main() -> None:
    args = _parse_args()
    base_dir = args.base_dir.resolve()
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Base directory does not exist: {base_dir}")
    image_root = args.image_root.resolve() if args.image_root else base_dir

    input_paths = [_resolve_path(candidate, base_dir) for candidate in args.inputs]

    output_path = _resolve_output_path(args.output, base_dir)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_df = merge_parquet_files(input_paths)
    merged_df.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(merged_df)} rows) from {len(input_paths)} source files.")

    widths, heights = collect_image_dimensions(merged_df, image_root)
    histogram_path = _resolve_output_path(args.histogram_file, base_dir)
    save_histogram(widths, heights, histogram_path)

    if "class_id" in merged_df.columns:
        class_counts = Counter(int(value) for value in merged_df["class_id"].dropna())
        pie_path = _resolve_output_path(args.class_ratio_file, base_dir)
        save_class_ratio_chart(class_counts, pie_path)
    else:
        print("[pie] Skipping class ratio chart; 'class_id' column not present.")


if __name__ == "__main__":
    main()
