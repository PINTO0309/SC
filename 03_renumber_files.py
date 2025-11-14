#!/usr/bin/env python3
"""Rename AVA videos to sequential IDs and sync annotation CSVs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
TRAINVAL_DIR = ROOT / "data/trainval"
TEST_DIR = ROOT / "data/test"
TRAIN_CSV = ROOT / "ava_v2.2" / "ava_train_v2.2.csv"
VAL_CSV = ROOT / "ava_v2.2" / "ava_val_v2.2.csv"
TRAIN_SEQ_CSV = TRAIN_CSV.with_name("ava_train_v2.2_seq.csv")
VAL_SEQ_CSV = VAL_CSV.with_name("ava_val_v2.2_seq.csv")
DEFAULT_SUFFIXES = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".flv",
    ".mpg",
    ".mpeg",
    ".m4v",
}


def iter_videos(directory: Path, *, suffixes: set[str] | None) -> list[Path]:
    """Return sorted files that look like videos inside directory."""
    if not directory.exists():
        raise FileNotFoundError(f"{directory} does not exist")
    files = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and (suffixes is None or path.suffix.lower() in suffixes)
    ]
    return sorted(files, key=lambda p: p.name)


def gather_videos(
    directories: Iterable[Path],
    *,
    suffixes: set[str] | None,
) -> list[Path]:
    """Collect videos from each directory while preserving directory order."""
    collected: list[Path] = []
    for directory in directories:
        collected.extend(iter_videos(directory, suffixes=suffixes))
    return collected


def rename_videos(
    files: list[Path],
    *,
    start_index: int,
    width: int,
    dry_run: bool,
) -> dict[str, str]:
    """Rename each file to a zero-padded sequential ID, returning old->new map."""
    if start_index < 1:
        raise ValueError("start_index must be at least 1")
    if width < 1:
        raise ValueError("width must be at least 1")

    mapping: dict[str, str] = {}
    operations: list[tuple[Path, Path]] = []
    current_index = start_index
    for file_path in files:
        old_base = file_path.stem
        if old_base in mapping:
            raise ValueError(f"Duplicate base name encountered: {old_base}")

        new_base = f"{current_index:0{width}d}"
        new_name = f"{new_base}{file_path.suffix}"
        new_path = file_path.with_name(new_name)

        mapping[old_base] = new_base
        if file_path != new_path:
            operations.append((file_path, new_path))
        current_index += 1

    if dry_run:
        for src, dst in operations:
            print(f"[rename] {src} -> {dst}")
    else:
        temp_ops: list[tuple[Path, Path]] = []
        for index, (src, dst) in enumerate(operations, start=1):
            tmp = src.with_name(f".renumber_tmp_{index:04d}{src.suffix}")
            while tmp.exists():
                tmp = tmp.with_name(tmp.name + "_")
            src.rename(tmp)
            temp_ops.append((tmp, dst))
        for tmp, dst in temp_ops:
            if dst.exists():
                raise FileExistsError(f"Target file already exists: {dst}")
            tmp.rename(dst)

    total = len(files)
    print(
        f"[summary] Prepared mapping for {total} files "
        f"({start_index:0{width}d}-{start_index + total - 1:0{width}d})"
    )
    return mapping


def rewrite_annotations(
    input_csv: Path,
    output_csv: Path,
    mapping: dict[str, str],
    *,
    dry_run: bool,
) -> None:
    """Rewrite annotation CSV so that video_id reflects the new sequential IDs."""
    if not input_csv.exists():
        print(f"[warn] {input_csv} not found; skipping", file=sys.stderr)
        return

    rows: list[list[str]] = []
    total = 0
    updated = 0
    missing_ids: set[str] = set()

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            total += 1
            video_id = row[0]
            new_id = mapping.get(video_id)
            if new_id is not None:
                row[0] = new_id
                updated += 1
            else:
                missing_ids.add(video_id)
            rows.append(row)

    if dry_run:
        print(
            f"[csv] Would write {output_csv} ({updated}/{total} rows updated, "
            f"{len(missing_ids)} video IDs missing)"
        )
    else:
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)
        print(
            f"[csv] Wrote {output_csv} ({updated}/{total} rows updated, "
            f"{len(missing_ids)} video IDs missing)"
        )

    if missing_ids:
        preview = ", ".join(sorted(missing_ids)[:5])
        print(
            f"[warn] {len(missing_ids)} annotation video IDs not found in renamed files. "
            f"Examples: {preview}",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename videos under data/trainval and data/test to sequential IDs "
            "and update AVA annotation CSVs to match."
        )
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=[str(TRAINVAL_DIR), str(TEST_DIR)],
        help="Video directories to process in order (default: trainval then test).",
    )
    parser.add_argument(
        "--train-csv",
        default=str(TRAIN_CSV),
        help="Input train annotation CSV (default: ava_v2.2/ava_train_v2.2.csv).",
    )
    parser.add_argument(
        "--val-csv",
        default=str(VAL_CSV),
        help="Input val annotation CSV (default: ava_v2.2/ava_val_v2.2.csv).",
    )
    parser.add_argument(
        "--train-out",
        default=str(TRAIN_SEQ_CSV),
        help="Output CSV path for train annotations (default: *_seq.csv).",
    )
    parser.add_argument(
        "--val-out",
        default=str(VAL_SEQ_CSV),
        help="Output CSV path for val annotations (default: *_seq.csv).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting index for sequential IDs (default: 1).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=4,
        help="Zero-pad width for the sequential IDs (default: 4).",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Rename every file in the directories, even if the extension is unknown.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show operations without renaming files or writing CSVs.",
    )
    parser.add_argument(
        "--skip-annotations",
        action="store_true",
        help="Rename files but skip CSV rewriting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directories = [(ROOT / directory).resolve() for directory in args.dirs]
    suffixes = None if args.all_files else DEFAULT_SUFFIXES

    video_files = gather_videos(directories, suffixes=suffixes)
    if not video_files:
        print("[error] No video files found in the provided directories.", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Found {len(video_files)} video files to renumber.")
    mapping = rename_videos(
        video_files,
        start_index=args.start,
        width=args.width,
        dry_run=args.dry_run,
    )

    if args.skip_annotations:
        return

    rewrite_annotations(
        Path(args.train_csv),
        Path(args.train_out),
        mapping,
        dry_run=args.dry_run,
    )
    rewrite_annotations(
        Path(args.val_csv),
        Path(args.val_out),
        mapping,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
