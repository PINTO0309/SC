#!/usr/bin/env python3
"""Generate annotated crops from videos located under real_data/."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = ROOT / "real_data"
DEFAULT_IMAGE_DIR = ROOT / "data" / "images"
DEFAULT_ANNOTATION_FILE = ROOT / "data" / "annotation.txt"
DEFAULT_BACKUP_SUFFIX = ".realdata.bak"
DEFAULT_IMAGES_PER_FOLDER = 1000
DEFAULT_FOLDER_START = 2001
DEFAULT_FRAME_STEP = 1
MIN_DIMENSION = 6
MAX_DIMENSION = 750
DEFAULT_DETECTOR_MODEL = ROOT / "deimv2_dinov3_x_wholebody34_680query_n_batch_640x640.onnx"
DETECTOR_INPUT_SIZE = 640
DETECTOR_BODY_LABEL = 0
DETECTOR_ABDOMEN_LABEL = 29
DETECTOR_HIP_LABEL = 30
DETECTOR_KNEE_LABELS = (31, 32)
DETECTOR_BODY_THRESHOLD = 0.35
DETECTOR_PART_THRESHOLD = 0.30
SITTING_KNEE_MIN_VERTICAL_DELTA = -0.05
SITTING_KNEE_MAX_VERTICAL_DELTA = 0.22


@dataclass
class FrameAnnotation:
    filename: str
    video_id: str
    timestamp: str
    person_id: int
    class_id: int


@dataclass
class DetectorCropStats:
    body_count: int
    abdomen_present: bool
    hip_present: bool
    knee_present: bool
    hip_knee_delta: float | None

    @property
    def hip_knee_in_range(self) -> bool:
        if self.hip_knee_delta is None:
            return False
        return SITTING_KNEE_MIN_VERTICAL_DELTA <= self.hip_knee_delta <= SITTING_KNEE_MAX_VERTICAL_DELTA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert mp4 videos under real_data/ into annotated PNG crops."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="Directory containing real_data mp4 files.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory where cropped PNGs are stored.")
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=DEFAULT_ANNOTATION_FILE,
        help="Annotation text file to append to.",
    )
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=DEFAULT_BACKUP_SUFFIX,
        help="Suffix appended to annotation file for backups (default: %(default)s).",
    )
    parser.add_argument("--images-per-folder", type=int, default=DEFAULT_IMAGES_PER_FOLDER, help="Maximum PNGs per folder (default: 1000).")
    parser.add_argument(
        "--start-folder",
        type=int,
        default=DEFAULT_FOLDER_START,
        help="Numeric folder index to start from when saving images (default: 2001).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=DEFAULT_FRAME_STEP,
        help="Take every Nth frame from each video (default: 1).",
    )
    parser.add_argument("--person-id", type=int, default=1, help="Person identifier stored in annotations (default: 1).")
    parser.add_argument("--min-dimension", type=int, default=MIN_DIMENSION, help="Minimum width/height required (default: 6).")
    parser.add_argument("--max-dimension", type=int, default=MAX_DIMENSION, help="Maximum width/height allowed (default: 750).")
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=DEFAULT_DETECTOR_MODEL,
        help="ONNX detector used for cropping and filtering (default: deimv2...640.onnx).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs if duplicates occur.")
    parser.add_argument("--dry-run", action="store_true", help="Plan operations without writing files.")
    args = parser.parse_args()
    if args.images_per_folder < 1:
        parser.error("--images-per-folder must be at least 1")
    if args.frame_step < 1:
        parser.error("--frame-step must be at least 1")
    if args.max_dimension < args.min_dimension:
        parser.error("--max-dimension must be greater than or equal to --min-dimension")
    return args


def load_detector_session(model_path: Path) -> tuple[ort.InferenceSession, str]:
    if not model_path.exists():
        raise FileNotFoundError(f"Detector model not found: {model_path}")
    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": ".",
                "trt_op_types_to_exclude": "NonMaxSuppression,NonZero,RoiAlign",
            },
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def _compute_center_y(det: np.ndarray) -> float:
    y1 = float(det[2])
    y2 = float(det[4])
    return (y1 + y2) / 2.0


def detector_evaluate_crop(
    session: ort.InferenceSession,
    input_name: str,
    image_path: Path,
) -> DetectorCropStats:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Detector input missing: {image_path}")
    stats = detector_evaluate_image(session, input_name, image)
    return stats


def detector_evaluate_image(
    session: ort.InferenceSession,
    input_name: str,
    image: np.ndarray,
) -> DetectorCropStats:
    detections = _run_detector(session, input_name, image)
    body_count = 0
    abdomen_present = False
    hip_present = False
    knee_present = False
    hip_centers: list[float] = []
    knee_centers: list[float] = []
    for det in detections:
        label = int(round(det[0]))
        score = float(det[5])
        if label == DETECTOR_BODY_LABEL and score >= DETECTOR_BODY_THRESHOLD:
            body_count += 1
        elif label == DETECTOR_ABDOMEN_LABEL and score >= DETECTOR_PART_THRESHOLD:
            abdomen_present = True
        elif label == DETECTOR_HIP_LABEL and score >= DETECTOR_PART_THRESHOLD:
            hip_present = True
            hip_centers.append(_compute_center_y(det))
        elif label in DETECTOR_KNEE_LABELS and score >= DETECTOR_PART_THRESHOLD:
            knee_present = True
            knee_centers.append(_compute_center_y(det))

    hip_knee_delta: float | None = None
    if hip_centers and knee_centers:
        _, hip_knee_delta = min(
            ((abs(k - h), k - h) for h in hip_centers for k in knee_centers),
            key=lambda item: item[0],
        )

    return DetectorCropStats(
        body_count=body_count,
        abdomen_present=abdomen_present,
        hip_present=hip_present,
        knee_present=knee_present,
        hip_knee_delta=hip_knee_delta,
    )


def _prepare_detector_blob(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        image,
        (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    blob = resized.transpose(2, 0, 1).astype(np.float32, copy=False)
    blob = np.expand_dims(blob, axis=0)
    return blob


def _run_detector(session: ort.InferenceSession, input_name: str, image: np.ndarray) -> np.ndarray:
    blob = _prepare_detector_blob(image)
    return session.run(None, {input_name: blob})[0][0]


def detect_person_box(
    session: ort.InferenceSession,
    input_name: str,
    frame: np.ndarray,
) -> tuple[tuple[float, float, float, float], float] | None:
    detections = _run_detector(session, input_name, frame)
    best_detection: tuple[np.ndarray, float] | None = None
    for det in detections:
        label = int(round(det[0]))
        score = float(det[5])
        if label != DETECTOR_BODY_LABEL or score < DETECTOR_BODY_THRESHOLD:
            continue
        if best_detection is None or score > best_detection[1]:
            best_detection = (det, score)
    if best_detection is None:
        return None
    det_array = best_detection[0]
    box = (float(det_array[1]), float(det_array[2]), float(det_array[3]), float(det_array[4]))
    return box, best_detection[1]


def crop_frame_using_box(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
) -> tuple[np.ndarray, int, int] | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    x2 = min(max(x2, 0.0), 1.0)
    y2 = min(max(y2, 0.0), 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    x1_px = max(int(round(x1 * width)), 0)
    y1_px = max(int(round(y1 * height)), 0)
    x2_px = min(int(round(x2 * width)), width)
    y2_px = min(int(round(y2 * height)), height)
    if x2_px <= x1_px or y2_px <= y1_px:
        return None
    crop = frame[y1_px:y2_px, x1_px:x2_px].copy()
    return crop, crop.shape[1], crop.shape[0]


def crop_passes_filters(class_id: int, stats: DetectorCropStats) -> bool:
    if class_id == 1:
        return (
            stats.body_count == 1
            and stats.abdomen_present
            and stats.hip_present
            and stats.knee_present
            and stats.hip_knee_in_range
        )
    return (
        stats.body_count == 1
        and stats.abdomen_present
        and stats.hip_present
        and stats.knee_present
        and stats.hip_knee_delta is not None
        and not stats.hip_knee_in_range
    )


def classify_video(path: Path) -> int | None:
    name = path.stem.lower()
    if name.startswith("sitting"):
        return 1
    if name.startswith("not_sitting"):
        return 0
    return None


def iter_video_files(input_dir: Path) -> list[tuple[Path, int]]:
    videos: list[tuple[Path, int]] = []
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    for video_path in sorted(input_dir.glob("*.mp4")):
        class_id = classify_video(video_path)
        if class_id is None:
            print(f"[skip] {video_path.name} does not match sitting/not_sitting pattern.", file=sys.stderr)
            continue
        videos.append((video_path, class_id))
    if not videos:
        print("[info] No matching mp4 files found.", file=sys.stderr)
    return videos


def ensure_backup(annotation_file: Path, suffix: str) -> Path | None:
    if not annotation_file.exists():
        return None
    backup_path = annotation_file.with_suffix(annotation_file.suffix + suffix)
    shutil.copy2(annotation_file, backup_path)
    print(f"[backup] Copied {annotation_file} -> {backup_path}")
    return backup_path


def next_image_path(
    image_dir: Path,
    start_folder: int,
    images_per_folder: int,
    generated_count: int,
    filename: str,
) -> Path:
    folder_index = start_folder + generated_count // images_per_folder
    folder_name = f"{folder_index:04d}"
    folder_path = image_dir / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path / filename


def save_frame(frame, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.save(output_path)


def process_video(
    video_path: Path,
    class_id: int,
    args: argparse.Namespace,
    generated_count: int,
    detector_session: ort.InferenceSession,
    detector_input_name: str,
) -> tuple[int, list[FrameAnnotation]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    fps = fps if fps > 0 else 30.0
    frame_index = 0
    kept = 0
    annotations: list[FrameAnnotation] = []
    video_id = video_path.stem

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        if frame_index % args.frame_step != 0:
            frame_index += 1
            continue

        detection = detect_person_box(detector_session, detector_input_name, frame)
        if detection is None:
            frame_index += 1
            continue
        box, _ = detection
        crop_result = crop_frame_using_box(frame, box)
        if crop_result is None:
            frame_index += 1
            continue
        crop, width_px, height_px = crop_result
        if (
            width_px < args.min_dimension
            or height_px < args.min_dimension
            or width_px > args.max_dimension
            or height_px > args.max_dimension
        ):
            frame_index += 1
            continue

        stats = detector_evaluate_image(detector_session, detector_input_name, crop)
        if not crop_passes_filters(class_id, stats):
            frame_index += 1
            continue

        timestamp_seconds = frame_index / fps
        timestamp = f"{timestamp_seconds:.3f}"
        filename = f"{video_id}_{frame_index:06d}_{class_id}.png"

        if not args.dry_run:
            output_path = next_image_path(
                args.image_dir, args.start_folder, args.images_per_folder, generated_count + kept, filename
            )
            try:
                save_frame(crop, output_path, overwrite=args.overwrite)
            except FileExistsError:
                frame_index += 1
                continue

        annotations.append(
            FrameAnnotation(
                filename=filename,
                video_id=video_id,
                timestamp=timestamp,
                person_id=args.person_id,
                class_id=class_id,
            )
        )
        kept += 1
        frame_index += 1

    capture.release()
    print(f"[info] Processed {video_path.name}: kept {kept} frames.")
    return kept, annotations


def append_annotations(annotation_file: Path, rows: list[FrameAnnotation]) -> None:
    if not rows:
        print("[annotation] No rows to append.")
        return
    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    with annotation_file.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(
            (row.filename, row.video_id, row.timestamp, row.person_id, row.class_id) for row in rows
        )
    print(f"[annotation] Appended {len(rows)} rows to {annotation_file}.")


def main() -> None:
    args = parse_args()
    videos = iter_video_files(args.input_dir)
    if not videos:
        return

    detector_session, detector_input_name = load_detector_session(args.detector_model)

    total_kept = 0
    all_rows: list[FrameAnnotation] = []
    for video_path, class_id in videos:
        kept, rows = process_video(
            video_path,
            class_id,
            args,
            total_kept,
            detector_session,
            detector_input_name,
        )
        total_kept += kept
        all_rows.extend(rows)

    if total_kept == 0:
        print("[info] No frames met the filtering criteria; nothing to write.")
        return

    if args.dry_run:
        print("[dry-run] Skipping file writes and annotation updates.")
        return

    ensure_backup(args.annotation_file, args.backup_suffix)
    append_annotations(args.annotation_file, all_rows)
    print(f"[done] Generated {total_kept} frames from {len(videos)} videos.")


if __name__ == "__main__":
    main()
