#!/usr/bin/env python3
"""Convert annotation CSV into a parquet dataset consumable by sc.data."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_ANNOTATION_FILE = ROOT / "data" / "annotation.txt"
DEFAULT_IMAGE_DIR = ROOT / "data" / "images"
DEFAULT_OUTPUT_FILE = ROOT / "data" / "dataset.parquet"
LABEL_MAP = {0: "not_sitting", 1: "sitting"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset.parquet for binary sitting classification.",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=DEFAULT_ANNOTATION_FILE,
        help=f"Annotation CSV with filename,video_id,timestamp,person_id,class_id (default: {DEFAULT_ANNOTATION_FILE})",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory containing cropped images (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Destination parquet file (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Target ratio for the training split.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Target ratio for the validation split.")
    parser.add_argument("--seed", type=int, default=42, help="Seed controlling random split assignment.")
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help="Store raw image bytes inside the parquet (increases file size but removes disk dependency).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing parquet file.",
    )
    return parser.parse_args()


def _load_annotations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    df = pd.read_csv(
        path,
        header=None,
        names=["filename", "video_id", "timestamp", "person_id", "class_id"],
        dtype=str,
    )
    df = df.dropna(how="all")
    if df.empty:
        raise RuntimeError(f"No rows found in {path}")

    for column in ("filename", "video_id", "timestamp"):
        df[column] = df[column].astype(str).str.strip()
    df["person_id"] = pd.to_numeric(df["person_id"], errors="raise").astype(int)
    df["class_id"] = pd.to_numeric(df["class_id"], errors="raise").astype(int)
    df = df[df["filename"] != ""]
    if df.empty:
        raise RuntimeError("Annotation file does not contain usable filename entries.")
    return df


def _build_image_index(image_dir: Path) -> Dict[str, Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    entries: Dict[str, Path] = {}
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        resolved = path.resolve()
        if name in entries:
            conflict_a = entries[name]
            raise RuntimeError(
                f"Duplicate filename {name!r} encountered under {conflict_a} and {resolved}. "
                "Filenames must be unique."
            )
        entries[name] = resolved
    if not entries:
        raise RuntimeError(f"No image files found under {image_dir}")
    return entries


def _normalize_ratios(train_ratio: float, val_ratio: float) -> Dict[str, float]:
    if train_ratio <= 0:
        raise ValueError("train-ratio must be greater than zero.")
    if val_ratio < 0:
        raise ValueError("val-ratio must be non-negative.")
    total = train_ratio + val_ratio
    if total <= 0:
        raise ValueError("At least one of the split ratios must be positive.")
    return {
        "train": train_ratio / total,
        "val": val_ratio / total,
    }


def _assign_row_splits(count: int, ratios: Dict[str, float], seed: int) -> list[str]:
    if count <= 0:
        raise RuntimeError("Annotation file does not contain any usable rows.")
    rng = random.Random(seed)
    ordered_splits = list(ratios.keys())
    thresholds = []
    cumulative = 0.0
    for split in ordered_splits:
        cumulative += ratios[split]
        thresholds.append((split, cumulative))
    assignments: list[str] = []
    for _ in range(count):
        value = rng.random()
        for split, cutoff in thresholds:
            if value <= cutoff:
                assignments.append(split)
                break
        else:
            assignments.append(ordered_splits[-1])
    return assignments


def _summarize(df: pd.DataFrame) -> None:
    split_counts = Counter(df["split"])
    label_counts = Counter(df["label"])
    print("Split counts:")
    for split in ("train", "val"):
        print(f"  {split:>5}: {split_counts.get(split, 0)}")
    print("Label counts:")
    for label in ("not_sitting", "sitting"):
        print(f"  {label:>12}: {label_counts.get(label, 0)}")


def _format_image_path(path: Path, dataset_root: Path) -> str:
    try:
        return path.resolve().relative_to(dataset_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _attach_image_paths(df: pd.DataFrame, image_index: Dict[str, Path], dataset_root: Path) -> pd.DataFrame:
    df = df.copy()
    resolved_col = "_resolved_image_path"
    df[resolved_col] = df["filename"].map(image_index)
    missing_mask = df[resolved_col].isna()
    if missing_mask.any():
        missing_files = sorted(df.loc[missing_mask, "filename"].unique())
        preview = ", ".join(missing_files[:10])
        print(
            f"[warn] Dropping {missing_mask.sum()} rows because the corresponding images were not found. "
            f"Examples: {preview}",
            file=sys.stderr,
        )
        df = df[~missing_mask]
    if df.empty:
        raise RuntimeError("No annotation rows remain after resolving image paths.")
    df["image_path"] = df[resolved_col].apply(lambda path: _format_image_path(path, dataset_root))
    return df


def _read_image_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Image file missing while embedding: {path}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    annotation_df = _load_annotations(args.annotation_file)
    output_root = args.output.resolve().parent
    output_root.mkdir(parents=True, exist_ok=True)
    image_index = _build_image_index(args.image_dir)
    ratio_map = _normalize_ratios(args.train_ratio, args.val_ratio)
    dataset_df = _attach_image_paths(annotation_df, image_index, output_root)

    dataset_df["split"] = _assign_row_splits(len(dataset_df), ratio_map, args.seed)
    dataset_df["source"] = dataset_df["video_id"]
    dataset_df["label"] = dataset_df["class_id"].map(LABEL_MAP)
    if dataset_df["label"].isna().any():
        bad_values = sorted(dataset_df.loc[dataset_df["label"].isna(), "class_id"].unique())
        raise ValueError(f"Unsupported class_id values detected: {bad_values}")

    resolved_col = "_resolved_image_path"
    if args.embed_images:
        dataset_df["image_bytes"] = dataset_df[resolved_col].apply(_read_image_bytes)
    dataset_df = dataset_df.drop(columns=[resolved_col])

    ordered_columns = [
        "split",
        "image_path",
        "class_id",
        "label",
        "source",
        "filename",
        "video_id",
        "timestamp",
        "person_id",
    ]
    if "image_bytes" in dataset_df.columns:
        ordered_columns.insert(2, "image_bytes")
    dataset_df = dataset_df[ordered_columns]
    dataset_df = dataset_df.sort_values(["split", "video_id", "timestamp", "person_id"]).reset_index(drop=True)
    return dataset_df


def main() -> None:
    args = _parse_args()
    output = args.output
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; use --overwrite to replace it.")

    dataset_df = build_dataset(args)
    dataset_df.to_parquet(output, index=False)
    print(f"Wrote {output} ({len(dataset_df)} rows).")
    _summarize(dataset_df)


if __name__ == "__main__":
    main()
