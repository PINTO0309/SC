#!/usr/bin/env python3
"""Download AVA train/val and test videos listed in the provided text files."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent

LIST_SPECS = {
    "train": ("train_list.txt", "data/trainval"),
    "test": ("test_list.txt", "data/test"),
}

MAX_WORKERS = 8


def parse_url(tokens: list[str]) -> str | None:
    """Return the first HTTP(S) argument inside a parsed wget command."""
    for token in tokens[1:]:
        if token.startswith(("http://", "https://")):
            return token
    return None


def download_from_list(
    list_name: str,
    target_subdir: str,
    *,
    resume: bool = True,
    dry_run: bool = False,
    skip_existing: bool = True,
    max_workers: int = 1,
) -> None:
    list_path = ROOT / list_name
    if not list_path.exists():
        raise FileNotFoundError(f"{list_path} not found")

    target_dir = ROOT / target_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[list[str], str]] = []
    with list_path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            command = raw_line.strip()
            if not command or command.startswith("#"):
                continue

            tokens = shlex.split(command)
            if not tokens or tokens[0] != "wget":
                print(
                    f"[skip] Unsupported line {list_path.name}:{line_no}: {command}",
                    file=sys.stderr,
                )
                continue

            url = parse_url(tokens)
            if not url:
                print(
                    f"[skip] Missing URL {list_path.name}:{line_no}: {command}",
                    file=sys.stderr,
                )
                continue

            filename = Path(urlparse(url).path).name or "download"
            output_path = target_dir / filename
            if skip_existing and output_path.exists():
                print(f"[skip] {output_path} already exists")
                continue

            exe_tokens = tokens
            if resume and "-c" not in tokens[1:]:
                exe_tokens = [tokens[0], "-c", *tokens[1:]]

            label = f"{list_path.name}:{line_no} -> {output_path}"
            print(f"[run] {label}")
            if dry_run:
                continue

            jobs.append((exe_tokens, label))

    if dry_run or not jobs:
        return

    worker_count = min(max_workers, MAX_WORKERS)
    if worker_count < 1:
        raise ValueError("max_workers must be at least 1")

    errors: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                subprocess.run,
                exe_tokens,
                cwd=target_dir,
                check=True,
            ): label
            for exe_tokens, label in jobs
        }
        for future in as_completed(future_map):
            label = future_map[future]
            try:
                future.result()
            except Exception as exc:  # capture CalledProcessError and unexpected issues
                errors.append((label, exc))
                print(f"[error] {label}: {exc}", file=sys.stderr)

    if errors:
        failed = ", ".join(label for label, _ in errors)
        raise RuntimeError(f"One or more downloads failed: {failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the AVA wget command lists so train/val videos land in "
            "data/trainval and test videos land in data/test."
        )
    )
    parser.add_argument(
        "--lists",
        choices=("train", "test", "all"),
        default="all",
        help="Choose which command lists to execute (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the commands that would be run without downloading anything.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable wget's -c option that resumes partially downloaded files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download files even if they already exist in the target directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=f"Number of parallel downloads per list (1-{MAX_WORKERS}, default: 8)",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")
    return args


def main() -> None:
    args = parse_args()
    if args.lists == "all":
        selections = ("train", "test")
    else:
        selections = (args.lists,)

    for key in selections:
        list_file, target_dir = LIST_SPECS[key]
        download_from_list(
            list_file,
            target_dir,
            resume=not args.no_resume,
            dry_run=args.dry_run,
            skip_existing=not args.force,
            max_workers=args.workers,
        )


if __name__ == "__main__":
    main()
