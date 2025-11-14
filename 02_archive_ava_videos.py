#!/usr/bin/env python3
"""Batch videos inside data/trainval and data/test into sequential .tar.gz files."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
DEFAULT_DIRECTORIES = ("data/trainval", "data/test")
DEFAULT_PREFIX_MAP = {
    (ROOT / "data/trainval").resolve(): "trainval",
    (ROOT / "data/test").resolve(): "test",
}
VIDEO_SUFFIXES = {
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


def iter_video_files(directory: Path) -> list[Path]:
    """Return sorted video files contained directly within directory."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def chunked(files: list[Path], size: int) -> Iterable[list[Path]]:
    """Yield successive chunks from files."""
    for idx in range(0, len(files), size):
        yield files[idx : idx + size]


def archive_directory(
    directory: Path,
    batch_size: int,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    prefix: str | None = None,
) -> list[Path]:
    """Create tar.gz archives containing up to batch_size files from directory."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if not directory.exists():
        raise FileNotFoundError(f"{directory} does not exist")

    video_files = iter_video_files(directory)
    if not video_files:
        print(f"[archive] No video files found in {directory}, skipping")
        return []

    chunks = list(chunked(video_files, batch_size))
    digits = max(2, len(str(len(chunks))))
    base_name = prefix or directory.name
    archives: list[Path] = []

    for index, group in enumerate(chunks, start=1):
        archive_name = f"{base_name}_{index:0{digits}d}.tar.gz"
        archive_path = directory.parent / archive_name

        if archive_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{archive_path} already exists. Use --overwrite to replace it."
                )
            if not dry_run:
                archive_path.unlink()

        print(f"[archive] {archive_path} ({len(group)} files)")
        if dry_run:
            archives.append(archive_path)
            continue

        with tarfile.open(archive_path, "w:gz") as tar:
            for file_path in group:
                tar.add(file_path, arcname=file_path.name)
        archives.append(archive_path)

    return archives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package videos under each specified directory into tar.gz archives "
            "containing up to N files (default 20)."
        )
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=DEFAULT_DIRECTORIES,
        help="Directories to archive (default: data/trainval data/test).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of files per archive (default: 20).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite archives if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the archives that would be created without writing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for directory in args.dirs:
        target_dir = (ROOT / directory).resolve()
        prefix = DEFAULT_PREFIX_MAP.get(target_dir)
        try:
            archive_directory(
                target_dir,
                args.batch_size,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                prefix=prefix,
            )
        except Exception as exc:
            print(f"[error] {target_dir}: {exc}")


if __name__ == "__main__":
    main()
