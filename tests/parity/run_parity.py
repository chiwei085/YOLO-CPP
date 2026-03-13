#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run lightweight parity checks")
    parser.add_argument(
        "--task",
        choices=("all", "detect", "classify", "seg"),
        default="all",
        help="Parity slice to run",
    )
    parser.add_argument(
        "--build-dir",
        default=str(root / "build" / "dev"),
        help="CMake build directory that contains yolo_cpp_parity_dump",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "tests" / "assets" / "baselines" / "parity"),
        help="Directory where python/cpp JSON summaries are written",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run the parity unittest modules after generating summaries",
    )
    return parser.parse_args()


def import_dependencies():
    try:
        import numpy as np
        from PIL import Image
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing parity dependency. Install ultralytics, onnx, "
            "onnxruntime, pillow, and numpy first."
        ) from exc

    return np, Image, YOLO


def task_configs(root: Path) -> dict[str, dict[str, object]]:
    images = root / "tests" / "assets" / "images"
    models = root / "tests" / "assets" / "models"
    return {
        "detect": {
            "task": "detect",
            "ultralytics_task": "detect",
            "model": models / "yolov8n.onnx",
            "images": [images / "test1.jpg", images / "test2.jpg"],
        },
        "classify": {
            "task": "classify",
            "ultralytics_task": "classify",
            "model": models / "yolov8n-cls.onnx",
            "images": [images / "test1.jpg", images / "test2.jpg"],
        },
        "seg": {
            "task": "seg",
            "ultralytics_task": "segment",
            "model": models / "yolov8n-seg.onnx",
            "images": [images / "test1.jpg"],
        },
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def bbox_from_xyxy(xyxy: list[float]) -> list[float]:
    return [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]


def encode_rle(mask_values: list[int]) -> list[int]:
    if not mask_values:
        return []

    runs: list[int] = []
    current = 0
    count = 0
    for value in mask_values:
        if value == current:
            count += 1
            continue

        runs.append(count)
        current = value
        count = 1

    runs.append(count)
    return runs


def python_detect_summary(model, image_path: Path) -> dict[str, object]:
    result = model.predict(str(image_path), verbose=False)[0]
    detections: list[dict[str, object]] = []
    if result.boxes is not None:
        boxes_xyxy = result.boxes.xyxy.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        for bbox, class_id, score in zip(boxes_xyxy, classes, scores):
            detections.append(
                {
                    "class_id": int(class_id),
                    "score": float(score),
                    "bbox": [float(value) for value in bbox_from_xyxy(bbox)],
                }
            )
    return {"image": image_path.name, "detections": detections}


def python_classify_summary(model, image_path: Path) -> dict[str, object]:
    result = model.predict(str(image_path), verbose=False)[0]
    probabilities = result.probs.data.cpu().tolist()
    ordered = sorted(
        enumerate(probabilities), key=lambda item: item[1], reverse=True
    )
    return {
        "image": image_path.name,
        "classes": [
            {"class_id": int(index), "score": float(score)}
            for index, score in ordered[:5]
        ],
        "scores": [float(score) for score in probabilities],
    }


def python_seg_summary(np, model, image_path: Path) -> dict[str, object]:
    result = model.predict(str(image_path), verbose=False, retina_masks=True)[0]
    instances: list[dict[str, object]] = []
    if result.boxes is not None:
        boxes_xyxy = result.boxes.xyxy.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        mask_data = None
        if result.masks is not None:
            mask_data = result.masks.data.cpu().numpy()
        for index, (bbox, class_id, score) in enumerate(
            zip(boxes_xyxy, classes, scores)
        ):
            mask_payload = {"size": [0, 0], "area": 0, "rle": []}
            if mask_data is not None and index < len(mask_data):
                binary_mask = (mask_data[index] > 0.5).astype(np.uint8)
                flat_mask = binary_mask.reshape(-1).tolist()
                mask_payload = {
                    "size": [int(binary_mask.shape[1]), int(binary_mask.shape[0])],
                    "area": int(binary_mask.sum()),
                    "rle": encode_rle([int(value) for value in flat_mask]),
                }
            instances.append(
                {
                    "class_id": int(class_id),
                    "score": float(score),
                    "bbox": [float(value) for value in bbox_from_xyxy(bbox)],
                    "mask": mask_payload,
                }
            )
    return {"image": image_path.name, "instances": instances}


def cpp_summary(
    task: str, tool_path: Path, model_path: Path, image_paths: list[Path]
) -> dict[str, object]:
    command = [str(tool_path), task, str(model_path), *[str(path) for path in image_paths]]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    json_start = completed.stdout.find('{"task"')
    if json_start < 0:
        raise RuntimeError(
            f"C++ parity tool did not emit JSON. stderr was:\n{completed.stderr}"
        )
    return json.loads(completed.stdout[json_start:])


def build_ppm_inputs(image_module, image_paths: list[Path], scratch_dir: Path) -> list[Path]:
    ppm_paths: list[Path] = []
    for image_path in image_paths:
        ppm_path = scratch_dir / f"{image_path.stem}.ppm"
        image_module.open(image_path).convert("RGB").save(ppm_path, format="PPM")
        ppm_paths.append(ppm_path)
    return ppm_paths


def maybe_run_checks(root: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.parity.test_detection_parity",
            "tests.parity.test_classification_parity",
            "tests.parity.test_segmentation_parity",
        ],
        cwd=root,
        check=True,
    )


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        np, image_module, yolo_model = import_dependencies()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    tool_path = Path(args.build_dir) / "yolo_cpp_parity_dump"
    if not tool_path.exists():
        print(
            "Missing C++ parity tool. Build yolo_cpp_parity_dump first.",
            file=sys.stderr,
        )
        return 1

    configs = task_configs(root)
    requested_tasks = (
        list(configs.keys()) if args.task == "all" else [args.task]
    )

    manifest: dict[str, object] = {
        "status": "ok",
        "tasks": {},
    }
    with tempfile.TemporaryDirectory(prefix="yolo_cpp_parity_") as scratch:
        scratch_dir = Path(scratch)
        for task in requested_tasks:
            config = configs[task]
            model_path = Path(config["model"])
            image_paths = [Path(path) for path in config["images"]]
            if not model_path.exists() or any(not image.exists() for image in image_paths):
                manifest["tasks"][task] = {
                    "status": "skipped",
                    "reason": "missing-model-or-image",
                }
                continue

            ppm_paths = build_ppm_inputs(image_module, image_paths, scratch_dir)
            model = yolo_model(str(model_path), task=str(config["ultralytics_task"]))

            if task == "detect":
                python_payload = {
                    "task": task,
                    "images": [
                        python_detect_summary(model, image_path)
                        for image_path in image_paths
                    ],
                }
            elif task == "classify":
                python_payload = {
                    "task": task,
                    "images": [
                        python_classify_summary(model, image_path)
                        for image_path in image_paths
                    ],
                }
            else:
                python_payload = {
                    "task": task,
                    "images": [
                        python_seg_summary(np, model, image_path)
                        for image_path in image_paths
                    ],
                }

            cpp_payload = cpp_summary(task, tool_path, model_path, ppm_paths)
            write_json(output_dir / f"{task}_python.json", python_payload)
            write_json(output_dir / f"{task}_cpp.json", cpp_payload)
            manifest["tasks"][task] = {
                "status": "ok",
                "python": f"{task}_python.json",
                "cpp": f"{task}_cpp.json",
            }

    write_json(output_dir / "manifest.json", manifest)
    if args.check:
        maybe_run_checks(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
