#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from tests.parity._shared import (
        TASKS,
        default_build_dir,
        default_output_dir,
        extract_json_payload,
        manifest_path,
        missing_asset_message,
        ok_manifest_entry,
        parity_summary_path,
        ppm_input_path,
        skipped_manifest_entry,
        write_json,
    )
except ModuleNotFoundError:
    from _shared import (
        TASKS,
        default_build_dir,
        default_output_dir,
        extract_json_payload,
        manifest_path,
        missing_asset_message,
        ok_manifest_entry,
        parity_summary_path,
        ppm_input_path,
        skipped_manifest_entry,
        write_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight parity checks")
    parser.add_argument(
        "--task",
        choices=("all", *TASKS),
        default="all",
        help="Parity slice to run",
    )
    parser.add_argument(
        "--build-dir",
        default=str(default_build_dir()),
        help="CMake build directory that contains yolo_cpp_parity_dump",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
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
        from PIL import Image
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing parity dependency. Install ultralytics, onnx, "
            "onnxruntime, pillow, and numpy first."
        ) from exc

    return Image, YOLO


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
        "pose": {
            "task": "pose",
            "ultralytics_task": "pose",
            "model": models / "yolov8n-pose.onnx",
            "images": [images / "test2.jpg"],
        },
        "obb": {
            "task": "obb",
            "ultralytics_task": "obb",
            "model": models / "yolov8n-obb.onnx",
            "images": [images / "test2.jpg"],
        },
    }


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


def python_seg_summary(np_module, model, image_path: Path) -> dict[str, object]:
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
                binary_mask = (mask_data[index] > 0.5).astype(np_module.uint8)
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


def python_pose_summary(model_path: Path, image_path: Path) -> dict[str, object]:
    import cv2
    import onnxruntime as ort
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import nms, ops

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    input_info = session.get_inputs()[0]
    _, channels, height, width = input_info.shape
    if channels != 3:
        raise RuntimeError(f"expected 3-channel input, got {input_info.shape}")

    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    letterbox = LetterBox(new_shape=(height, width), auto=False, stride=32)
    letterboxed = letterbox(image=original)
    rgb = letterboxed[..., ::-1]
    tensor = (
        np.ascontiguousarray(rgb.transpose((2, 0, 1))[None]).astype(np.float32)
        / 255.0
    )

    prediction = session.run(None, {input_info.name: tensor})[0]
    prediction_tensor = torch.from_numpy(prediction.copy())
    detections = nms.non_max_suppression(
        prediction_tensor,
        conf_thres=0.25,
        iou_thres=0.45,
        nc=1,
        agnostic=False,
        max_det=300,
    )[0]

    if len(detections) == 0:
        return {"image": image_path.name, "poses": []}

    detections_scaled = detections.clone()
    detections_scaled[:, :4] = ops.scale_boxes(
        (height, width), detections_scaled[:, :4], original.shape
    )
    keypoints = detections[:, 6:].view(detections.shape[0], 17, 3).clone()
    keypoints = ops.scale_coords((height, width), keypoints, original.shape)

    poses: list[dict[str, object]] = []
    for index in range(detections.shape[0]):
        row = detections_scaled[index]
        pose_keypoints: list[dict[str, object]] = []
        for keypoint in keypoints[index]:
            score = float(keypoint[2].item())
            pose_keypoints.append(
                {
                    "point": [float(keypoint[0].item()), float(keypoint[1].item())],
                    "score": score,
                    "visible": score > 0.0,
                }
            )

        poses.append(
            {
                "class_id": int(row[5].item()),
                "score": float(row[4].item()),
                "bbox": [
                    float(row[0].item()),
                    float(row[1].item()),
                    float((row[2] - row[0]).item()),
                    float((row[3] - row[1]).item()),
                ],
                "keypoints": pose_keypoints,
            }
        )

    return {"image": image_path.name, "poses": poses}


def python_obb_summary(model, image_path: Path) -> dict[str, object]:
    result = model.predict(str(image_path), verbose=False, imgsz=1024, device="cpu")[
        0
    ]
    boxes: list[dict[str, object]] = []
    if result.obb is not None:
        xywhr = result.obb.xywhr.cpu().tolist()
        classes = result.obb.cls.cpu().tolist()
        scores = result.obb.conf.cpu().tolist()
        corners = result.obb.xyxyxyxy.cpu().tolist()
        for box, class_id, score, quad in zip(xywhr, classes, scores, corners):
            boxes.append(
                {
                    "class_id": int(class_id),
                    "score": float(score),
                    "center": [float(box[0]), float(box[1])],
                    "size": [float(box[2]), float(box[3])],
                    "angle_radians": float(box[4]),
                    "corners": [[float(point[0]), float(point[1])] for point in quad],
                }
            )
    return {"image": image_path.name, "boxes": boxes}


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
    return extract_json_payload(
        completed.stdout, completed.stderr, "C++ parity tool"
    )


def build_ppm_inputs(image_module, image_paths: list[Path], scratch_dir: Path) -> list[Path]:
    ppm_paths: list[Path] = []
    for image_path in image_paths:
        ppm_path = ppm_input_path(image_path, scratch_dir)
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
            "tests.parity.test_pose_parity",
            "tests.parity.test_obb_parity",
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
        image_module, yolo_model = import_dependencies()
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
    requested_tasks = list(configs.keys()) if args.task == "all" else [args.task]

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
            if not model_path.exists() or any(
                not image.exists() for image in image_paths
            ):
                missing_detail = (
                    missing_asset_message("model", model_path)
                    if not model_path.exists()
                    else missing_asset_message(
                        "image", next(image for image in image_paths if not image.exists())
                    )
                )
                manifest["tasks"][task] = skipped_manifest_entry(
                    "missing-model-or-image", missing_detail
                )
                continue

            ppm_paths = build_ppm_inputs(image_module, image_paths, scratch_dir)
            if task == "pose":
                python_payload = {
                    "task": task,
                    "images": [
                        python_pose_summary(model_path, image_path)
                        for image_path in image_paths
                    ],
                }
            else:
                model = yolo_model(
                    str(model_path), task=str(config["ultralytics_task"])
                )

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
            elif task == "seg":
                python_payload = {
                    "task": task,
                    "images": [
                        python_seg_summary(np, model, image_path)
                        for image_path in image_paths
                    ],
                }
            elif task == "obb":
                python_payload = {
                    "task": task,
                    "images": [
                        python_obb_summary(model, image_path)
                        for image_path in image_paths
                    ],
                }

            cpp_payload = cpp_summary(task, tool_path, model_path, ppm_paths)
            write_json(parity_summary_path(task, "python", output_dir), python_payload)
            write_json(parity_summary_path(task, "cpp", output_dir), cpp_payload)
            manifest["tasks"][task] = ok_manifest_entry(task)

    write_json(manifest_path(output_dir), manifest)
    if args.check:
        maybe_run_checks(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
