#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

try:
    from tests.parity._shared import (
        POSE_DEBUG_STAGES,
        debug_comparison_path,
        debug_dump_path,
        debug_tolerances,
        default_build_dir,
        default_output_dir,
        extract_json_payload,
        missing_asset_message,
        ppm_input_path,
        write_json,
    )
except ModuleNotFoundError:
    from _shared import (
        POSE_DEBUG_STAGES,
        debug_comparison_path,
        debug_dump_path,
        debug_tolerances,
        default_build_dir,
        default_output_dir,
        extract_json_payload,
        missing_asset_message,
        ppm_input_path,
        write_json,
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate staged pose debug dumps for Python and C++"
    )
    parser.add_argument(
        "--model",
        default=str(
            Path(__file__).resolve().parents[2]
            / "tests"
            / "assets"
            / "models"
            / "yolov8n-pose.onnx"
        ),
    )
    parser.add_argument(
        "--image",
        default=str(root / "tests" / "assets" / "images" / "test2.jpg"),
    )
    parser.add_argument(
        "--build-dir",
        default=str(default_build_dir()),
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
    )
    return parser.parse_args()


def xyxy_to_xywh(values: list[float]) -> list[float]:
    return [
        float(values[0]),
        float(values[1]),
        float(values[2] - values[0]),
        float(values[3] - values[1]),
    ]


def build_python_debug(model_path: Path, image_path: Path) -> dict[str, object]:
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics.utils import nms, ops
    from PIL import Image

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    _, channels, height, width = input_info.shape
    if channels != 3:
        raise RuntimeError(f"expected 3-channel input, got {input_info.shape}")

    original = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
    source_height, source_width = original.shape[:2]

    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    pad_x = width - resized_width
    pad_y = height - resized_height
    pad_left = pad_x // 2
    pad_top = pad_y // 2
    pad_right = pad_x - pad_left
    pad_bottom = pad_y - pad_top

    tensor_hwc = np.full((height, width, 3), 114.0, dtype=np.float32)
    inner_x = np.arange(resized_width, dtype=np.float32)
    inner_y = np.arange(resized_height, dtype=np.float32)
    source_x = (inner_x + 0.5) / scale - 0.5
    source_y = (inner_y + 0.5) / scale - 0.5
    source_x = np.clip(source_x, 0.0, source_width - 1.0)
    source_y = np.clip(source_y, 0.0, source_height - 1.0)

    x0 = np.floor(source_x).astype(np.int32)
    y0 = np.floor(source_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    dx = (source_x - x0).reshape(1, resized_width, 1)
    dy = (source_y - y0).reshape(resized_height, 1, 1)

    p00 = original[y0[:, None], x0[None, :]]
    p10 = original[y0[:, None], x1[None, :]]
    p01 = original[y1[:, None], x0[None, :]]
    p11 = original[y1[:, None], x1[None, :]]
    top = p00 * (1.0 - dx) + p10 * dx
    bottom = p01 * (1.0 - dx) + p11 * dx
    resized = top * (1.0 - dy) + bottom * dy
    tensor_hwc[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
        :,
    ] = resized

    tensor = np.ascontiguousarray(tensor_hwc.transpose((2, 0, 1))[None]) / 255.0

    raw_outputs = session.run(None, {input_info.name: tensor})
    if len(raw_outputs) != 1:
        raise RuntimeError(f"expected 1 output, got {len(raw_outputs)}")

    prediction = raw_outputs[0]
    raw_pred = prediction[0]
    class_count = 1
    keypoint_count = (raw_pred.shape[0] - 4 - class_count) // 3

    decoded_candidates: list[dict[str, object]] = []
    for index in range(raw_pred.shape[1]):
        score = float(raw_pred[4, index])
        cx = float(raw_pred[0, index])
        cy = float(raw_pred[1, index])
        w = float(raw_pred[2, index])
        h = float(raw_pred[3, index])
        keypoints: list[dict[str, object]] = []
        keypoint_base = 5
        for keypoint_index in range(keypoint_count):
            offset = keypoint_base + keypoint_index * 3
            kp_score = float(raw_pred[offset + 2, index])
            keypoints.append(
                {
                    "point": [
                        float(raw_pred[offset, index]),
                        float(raw_pred[offset + 1, index]),
                    ],
                    "score": kp_score,
                    "visible": kp_score > 0.0,
                }
            )
        decoded_candidates.append(
            {
                "class_id": 0,
                "score": score,
                "bbox_xywh": [cx - w / 2.0, cy - h / 2.0, w, h],
                "keypoints": keypoints,
            }
        )

    decoded_candidates.sort(key=lambda item: item["score"], reverse=True)
    confidence_filtered = [item for item in decoded_candidates if item["score"] >= 0.25]

    prediction_tensor = torch.from_numpy(prediction.copy())
    detections = nms.non_max_suppression(
        prediction_tensor,
        conf_thres=0.25,
        iou_thres=0.45,
        nc=1,
        agnostic=False,
        max_det=300,
    )[0]

    nms_candidates: list[dict[str, object]] = []
    final_poses: list[dict[str, object]] = []
    if len(detections) > 0:
        keypoints = detections[:, 6:].view(detections.shape[0], keypoint_count, 3).clone()
        detections_scaled = detections.clone()
        detections_scaled[:, :4] = ops.scale_boxes(
            (height, width), detections_scaled[:, :4], original.shape
        )
        keypoints = ops.scale_coords((height, width), keypoints, original.shape)

        for index in range(detections.shape[0]):
            row = detections[index]
            scaled = detections_scaled[index]
            nms_keypoints: list[dict[str, object]] = []
            final_keypoints: list[dict[str, object]] = []
            for keypoint in row[6:].view(keypoint_count, 3):
                score = float(keypoint[2].item())
                nms_keypoints.append(
                    {
                        "point": [float(keypoint[0].item()), float(keypoint[1].item())],
                        "score": score,
                        "visible": score > 0.0,
                    }
                )
            for keypoint in keypoints[index]:
                score = float(keypoint[2].item())
                final_keypoints.append(
                    {
                        "point": [float(keypoint[0].item()), float(keypoint[1].item())],
                        "score": score,
                        "visible": score > 0.0,
                    }
                )

            nms_candidates.append(
                {
                    "class_id": int(row[5].item()),
                    "score": float(row[4].item()),
                    "bbox_xywh": xyxy_to_xywh([float(value) for value in row[:4].tolist()]),
                    "keypoints": nms_keypoints,
                }
            )
            final_poses.append(
                {
                    "class_id": int(scaled[5].item()),
                    "score": float(scaled[4].item()),
                    "bbox_xywh": xyxy_to_xywh([float(value) for value in scaled[:4].tolist()]),
                    "keypoints": final_keypoints,
                }
            )

    return {
        "task": "pose",
        "image": image_path.name,
        "preprocess": {
            "tensor_shape": [int(value) for value in tensor.shape],
            "tensor_values": [float(value) for value in tensor.reshape(-1).tolist()],
            "record": {
                "source_size": [int(source_width), int(source_height)],
                "target_size": [int(width), int(height)],
                "resized_size": [int(resized_width), int(resized_height)],
                "resize_scale": [
                    float(scale),
                    float(scale),
                ],
                "padding": {
                    "left": int(pad_left),
                    "top": int(pad_top),
                    "right": int(pad_right),
                    "bottom": int(pad_bottom),
                },
            },
        },
        "raw_outputs": [
            {
                "name": output.name,
                "shape": [int(value) for value in array.shape],
                "values": [float(value) for value in array.reshape(-1).tolist()],
            }
            for output, array in zip(session.get_outputs(), raw_outputs)
        ],
        "decoded": {
            "candidate_count": len(decoded_candidates),
            "candidates": decoded_candidates,
        },
        "confidence_filtered": {
            "candidate_count": len(confidence_filtered),
            "candidates": confidence_filtered,
        },
        "nms": {
            "candidate_count": len(nms_candidates),
            "candidates": nms_candidates,
        },
        "final": {
            "poses": final_poses,
        },
    }


def ppm_inputs(image_path: Path, scratch_dir: Path) -> Path:
    from PIL import Image

    ppm_path = scratch_dir / f"{image_path.stem}.ppm"
    Image.open(image_path).convert("RGB").save(ppm_path, format="PPM")
    return ppm_path


def build_cpp_debug(tool_path: Path, model_path: Path, ppm_path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [str(tool_path), str(model_path), str(ppm_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return extract_json_payload(
        completed.stdout, completed.stderr, "C++ pose debug tool"
    )


def max_abs_diff(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs):
        return math.inf
    if not lhs:
        return 0.0
    return max(abs(left - right) for left, right in zip(lhs, rhs))


def compare_keypoints(lhs: list[dict[str, object]], rhs: list[dict[str, object]]) -> tuple[float, float]:
    if len(lhs) != len(rhs):
        return math.inf, math.inf

    point_diff = 0.0
    score_diff = 0.0
    for left, right in zip(lhs, rhs):
        point_diff = max(point_diff, max_abs_diff(left["point"], right["point"]))
        score_diff = max(score_diff, abs(left["score"] - right["score"]))
    return point_diff, score_diff


def compare_debug(python_payload: dict[str, object], cpp_payload: dict[str, object]) -> dict[str, object]:
    tolerances = debug_tolerances("pose")
    comparisons: list[dict[str, object]] = []

    def record(stage: str, ok: bool, detail: str) -> None:
        comparisons.append({"stage": stage, "ok": ok, "detail": detail})

    preprocess_diff = max_abs_diff(
        python_payload["preprocess"]["tensor_values"],
        cpp_payload["preprocess"]["tensor_values"],
    )
    record(
        POSE_DEBUG_STAGES[0],
        preprocess_diff <= tolerances["preprocess"],
        f"max_abs_diff={preprocess_diff}",
    )

    python_raw = python_payload["raw_outputs"]
    cpp_raw = cpp_payload["raw_outputs"]
    raw_ok = len(python_raw) == len(cpp_raw)
    raw_detail = [f"count={len(python_raw)}:{len(cpp_raw)}"]
    if raw_ok:
        for index, (py_output, cpp_output) in enumerate(zip(python_raw, cpp_raw)):
            diff = max_abs_diff(py_output["values"], cpp_output["values"])
            raw_detail.append(f"{index}:{diff}")
            if diff > tolerances["raw_outputs"]:
                raw_ok = False
    record(POSE_DEBUG_STAGES[1], raw_ok, ", ".join(raw_detail))

    python_nms = python_payload["nms"]["candidates"]
    cpp_nms = cpp_payload["nms"]["candidates"]
    nms_ok = bool(python_nms) and bool(cpp_nms)
    if nms_ok:
        py_top = python_nms[0]
        cpp_top = cpp_nms[0]
        bbox_diff = max_abs_diff(py_top["bbox_xywh"], cpp_top["bbox_xywh"])
        score_diff = abs(py_top["score"] - cpp_top["score"])
        keypoint_point_diff, keypoint_score_diff = compare_keypoints(
            py_top["keypoints"], cpp_top["keypoints"]
        )
        class_ok = py_top["class_id"] == cpp_top["class_id"]
        nms_ok = (
            class_ok
            and bbox_diff <= tolerances["bbox_delta"]
            and score_diff <= tolerances["score_delta"]
            and keypoint_point_diff <= tolerances["keypoint_point_delta"]
            and keypoint_score_diff <= tolerances["keypoint_score_delta"]
        )
        record(
            POSE_DEBUG_STAGES[4],
            nms_ok,
            "class_match="
            f"{class_ok}, bbox_max_abs_diff={bbox_diff}, score_diff={score_diff}, "
            f"keypoint_point_diff={keypoint_point_diff}, keypoint_score_diff={keypoint_score_diff}",
        )
    else:
        record(POSE_DEBUG_STAGES[4], False, "missing top pose in python or cpp payload")

    python_final = python_payload["final"]["poses"]
    cpp_final = cpp_payload["final"]["poses"]
    final_ok = bool(python_final) and bool(cpp_final)
    if final_ok:
        py_top = python_final[0]
        cpp_top = cpp_final[0]
        bbox_diff = max_abs_diff(py_top["bbox_xywh"], cpp_top["bbox_xywh"])
        score_diff = abs(py_top["score"] - cpp_top["score"])
        keypoint_point_diff, keypoint_score_diff = compare_keypoints(
            py_top["keypoints"], cpp_top["keypoints"]
        )
        class_ok = py_top["class_id"] == cpp_top["class_id"]
        final_ok = (
            class_ok
            and bbox_diff <= tolerances["bbox_delta"]
            and score_diff <= tolerances["score_delta"]
            and keypoint_point_diff <= tolerances["keypoint_point_delta"]
            and keypoint_score_diff <= tolerances["keypoint_score_delta"]
        )
        record(
            POSE_DEBUG_STAGES[5],
            final_ok,
            "class_match="
            f"{class_ok}, bbox_max_abs_diff={bbox_diff}, score_diff={score_diff}, "
            f"keypoint_point_diff={keypoint_point_diff}, keypoint_score_diff={keypoint_score_diff}",
        )
    else:
        record(POSE_DEBUG_STAGES[5], False, "missing final pose in python or cpp payload")

    first_fail = next((entry["stage"] for entry in comparisons if not entry["ok"]), None)
    return {
        "comparisons": comparisons,
        "first_fail": first_fail,
    }


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    tool_path = Path(args.build_dir) / "yolo_cpp_pose_debug_dump"

    if not model_path.exists():
        raise SystemExit(missing_asset_message("model", model_path))
    if not image_path.exists():
        raise SystemExit(missing_asset_message("image", image_path))
    if not tool_path.exists():
        raise SystemExit(missing_asset_message("tool", tool_path))

    with tempfile.TemporaryDirectory(prefix="yolo_cpp_pose_debug_") as scratch:
        ppm_path = ppm_input_path(image_path, Path(scratch))
        from PIL import Image

        Image.open(image_path).convert("RGB").save(ppm_path, format="PPM")
        python_payload = build_python_debug(model_path, ppm_path)
        cpp_payload = build_cpp_debug(tool_path, model_path, ppm_path)
    comparison_payload = compare_debug(python_payload, cpp_payload)

    write_json(debug_dump_path("pose", "python", output_dir), python_payload)
    write_json(debug_dump_path("pose", "cpp", output_dir), cpp_payload)
    write_json(debug_comparison_path("pose", output_dir), comparison_payload)

    first_fail = comparison_payload["first_fail"]
    if first_fail is None:
        print("pose debug comparison: all stages within tolerance")
    else:
        print(f"pose debug comparison: first_fail={first_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
