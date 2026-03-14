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
        OBB_DEBUG_STAGES,
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
        OBB_DEBUG_STAGES,
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
        description="Generate staged OBB debug dumps for Python and C++"
    )
    parser.add_argument(
        "--model",
        default=str(
            Path(__file__).resolve().parents[2]
            / "tests"
            / "assets"
            / "models"
            / "yolov8n-obb.onnx"
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


def ppm_inputs(image_path: Path, scratch_dir: Path) -> Path:
    from PIL import Image

    ppm_path = ppm_input_path(image_path, scratch_dir)
    Image.open(image_path).convert("RGB").save(ppm_path, format="PPM")
    return ppm_path


def corners_from_xywhr(box: list[float]) -> list[list[float]]:
    import numpy as np

    x, y, w, h, angle = box
    cos_value = math.cos(angle)
    sin_value = math.sin(angle)
    vec1 = np.array([w * 0.5 * cos_value, w * 0.5 * sin_value], dtype=np.float32)
    vec2 = np.array([-h * 0.5 * sin_value, h * 0.5 * cos_value], dtype=np.float32)
    center = np.array([x, y], dtype=np.float32)
    corners = [
        center + vec1 + vec2,
        center + vec1 - vec2,
        center - vec1 - vec2,
        center - vec1 + vec2,
    ]
    return [[float(point[0]), float(point[1])] for point in corners]


def canonicalize_box(box: list[float]) -> list[float]:
    x, y, w, h, angle = box
    angle = math.fmod(angle, math.pi)
    if angle < 0.0:
        angle += math.pi
    if angle >= math.pi / 2.0:
        w, h = h, w
        angle -= math.pi / 2.0
    return [float(x), float(y), float(w), float(h), float(max(0.0, angle))]


def restore_box(box: list[float], scale: float, pad_left: int, pad_top: int) -> list[float]:
    x, y, w, h, angle = box
    return [
        float((x - pad_left) / scale),
        float((y - pad_top) / scale),
        float(w / scale),
        float(h / scale),
        float(angle),
    ]


def encode_box_payload(class_id: int, score: float, box: list[float]) -> dict[str, object]:
    return {
        "class_id": int(class_id),
        "score": float(score),
        "center": [float(box[0]), float(box[1])],
        "size": [float(box[2]), float(box[3])],
        "angle_radians": float(box[4]),
        "corners": corners_from_xywhr(box),
    }


def build_python_debug(model_path: Path, image_path: Path) -> dict[str, object]:
    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image
    from ultralytics.utils import nms

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
    source_x = np.clip((inner_x + 0.5) / scale - 0.5, 0.0, source_width - 1.0)
    source_y = np.clip((inner_y + 0.5) / scale - 0.5, 0.0, source_height - 1.0)
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
    tensor_hwc[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
        :,
    ] = top * (1.0 - dy) + bottom * dy
    tensor = np.ascontiguousarray(tensor_hwc.transpose((2, 0, 1))[None]) / 255.0

    raw_outputs = session.run(None, {input_info.name: tensor})
    if len(raw_outputs) != 1:
        raise RuntimeError(f"expected 1 output, got {len(raw_outputs)}")

    prediction = raw_outputs[0]
    raw_pred = prediction[0]
    class_count = raw_pred.shape[0] - 5
    class_channel_offset = 4
    angle_channel_offset = 4 + class_count

    decoded_candidates: list[dict[str, object]] = []
    canonicalized_candidates: list[dict[str, object]] = []
    for index in range(raw_pred.shape[1]):
        scores = raw_pred[class_channel_offset:angle_channel_offset, index]
        class_id = int(np.argmax(scores))
        score = float(scores[class_id])
        box = [
            float(raw_pred[0, index]),
            float(raw_pred[1, index]),
            float(raw_pred[2, index]),
            float(raw_pred[3, index]),
            float(raw_pred[angle_channel_offset, index]),
        ]
        decoded_candidates.append(encode_box_payload(class_id, score, box))
        canonicalized_candidates.append(
            encode_box_payload(class_id, score, canonicalize_box(box))
        )

    decoded_candidates.sort(key=lambda item: item["score"], reverse=True)
    canonicalized_candidates.sort(key=lambda item: item["score"], reverse=True)
    confidence_filtered = [
        item for item in canonicalized_candidates if item["score"] >= 0.25
    ]

    prediction_tensor = torch.from_numpy(prediction.copy())
    detections = nms.non_max_suppression(
        prediction_tensor,
        conf_thres=0.25,
        iou_thres=0.45,
        nc=class_count,
        agnostic=False,
        max_det=300,
        rotated=True,
    )[0]

    nms_boxes: list[dict[str, object]] = []
    final_boxes: list[dict[str, object]] = []
    for row in detections.tolist():
        box = canonicalize_box(row[:5])
        nms_boxes.append(encode_box_payload(int(row[6]), float(row[5]), box))
        final_boxes.append(
            encode_box_payload(
                int(row[6]),
                float(row[5]),
                restore_box(box, scale, pad_left, pad_top),
            )
        )

    return {
        "task": "obb",
        "image": image_path.name,
        "preprocess": {
            "tensor_shape": [int(value) for value in tensor.shape],
            "tensor_values": [float(value) for value in tensor.reshape(-1).tolist()],
            "record": {
                "source_size": [int(source_width), int(source_height)],
                "target_size": [int(width), int(height)],
                "resized_size": [int(resized_width), int(resized_height)],
                "resize_scale": [float(scale), float(scale)],
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
        "canonicalized": {
            "candidate_count": len(canonicalized_candidates),
            "candidates": canonicalized_candidates,
        },
        "confidence_filtered": {
            "candidate_count": len(confidence_filtered),
            "candidates": confidence_filtered,
        },
        "nms": {
            "candidate_count": len(nms_boxes),
            "candidates": nms_boxes,
        },
        "final": {
            "boxes": final_boxes,
        },
    }

def build_cpp_debug(tool_path: Path, model_path: Path, ppm_path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [str(tool_path), str(model_path), str(ppm_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return extract_json_payload(
        completed.stdout, completed.stderr, "C++ OBB debug tool"
    )


def max_abs_diff(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs):
        return math.inf
    return max((abs(left - right) for left, right in zip(lhs, rhs)), default=0.0)


def compare_box(lhs: dict[str, object], rhs: dict[str, object]) -> tuple[float, float, float]:
    center_diff = max_abs_diff(lhs["center"], rhs["center"])
    size_diff = max_abs_diff(lhs["size"], rhs["size"])
    angle_diff = abs(lhs["angle_radians"] - rhs["angle_radians"])
    return center_diff, size_diff, angle_diff


def compare_debug(python_payload: dict[str, object], cpp_payload: dict[str, object]) -> dict[str, object]:
    tolerances = debug_tolerances("obb")
    comparisons: list[dict[str, object]] = []

    def record(stage: str, ok: bool, detail: str) -> None:
        comparisons.append({"stage": stage, "ok": ok, "detail": detail})

    preprocess_diff = max_abs_diff(
        python_payload["preprocess"]["tensor_values"],
        cpp_payload["preprocess"]["tensor_values"],
    )
    record(
        OBB_DEBUG_STAGES[0],
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
    record(OBB_DEBUG_STAGES[1], raw_ok, ", ".join(raw_detail))

    for stage_name in (
        OBB_DEBUG_STAGES[3],
        OBB_DEBUG_STAGES[5],
        OBB_DEBUG_STAGES[6],
    ):
        python_boxes = python_payload[stage_name]["candidates" if stage_name != "final" else "boxes"]
        cpp_boxes = cpp_payload[stage_name]["candidates" if stage_name != "final" else "boxes"]
        ok = bool(python_boxes) and bool(cpp_boxes)
        if ok:
            py_top = python_boxes[0]
            cpp_top = cpp_boxes[0]
            center_diff, size_diff, angle_diff = compare_box(py_top["box"] if "box" in py_top else py_top,
                                                             cpp_top["box"] if "box" in cpp_top else cpp_top)
            class_ok = py_top["class_id"] == cpp_top["class_id"]
            score_diff = abs(py_top["score"] - cpp_top["score"])
            ok = (
                class_ok
                and center_diff <= tolerances["center_delta"]
                and size_diff <= tolerances["size_delta"]
                and angle_diff <= tolerances["angle_delta"]
                and score_diff <= tolerances["score_delta"]
            )
            record(
                stage_name,
                ok,
                f"class_match={class_ok}, center_diff={center_diff}, size_diff={size_diff}, angle_diff={angle_diff}, score_diff={score_diff}",
            )
        else:
            record(stage_name, False, "missing top box in python or cpp payload")

    first_fail = next((entry["stage"] for entry in comparisons if not entry["ok"]), None)
    return {"comparisons": comparisons, "first_fail": first_fail}


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    tool_path = Path(args.build_dir) / "yolo_cpp_obb_debug_dump"

    if not model_path.exists():
        raise SystemExit(missing_asset_message("model", model_path))
    if not image_path.exists():
        raise SystemExit(missing_asset_message("image", image_path))
    if not tool_path.exists():
        raise SystemExit(missing_asset_message("tool", tool_path))

    with tempfile.TemporaryDirectory(prefix="yolo_cpp_obb_debug_") as scratch:
        ppm_path = ppm_inputs(image_path, Path(scratch))
        python_payload = build_python_debug(model_path, ppm_path)
        cpp_payload = build_cpp_debug(tool_path, model_path, ppm_path)
    comparison_payload = compare_debug(python_payload, cpp_payload)

    write_json(debug_dump_path("obb", "python", output_dir), python_payload)
    write_json(debug_dump_path("obb", "cpp", output_dir), cpp_payload)
    write_json(debug_comparison_path("obb", output_dir), comparison_payload)

    first_fail = comparison_payload["first_fail"]
    if first_fail is None:
        print("obb debug comparison: all stages within tolerance")
    else:
        print(f"obb debug comparison: first_fail={first_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
