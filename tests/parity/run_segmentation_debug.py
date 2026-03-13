#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate staged segmentation debug dumps for Python and C++"
    )
    parser.add_argument(
        "--model",
        default=str(root / "tests" / "assets" / "models" / "yolov8n-seg.onnx"),
    )
    parser.add_argument(
        "--image",
        default=str(root / "tests" / "assets" / "images" / "test1.jpg"),
    )
    parser.add_argument(
        "--build-dir",
        default=str(root / "build" / "dev"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "tests" / "assets" / "baselines" / "parity"),
    )
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def xyxy_to_xywh(values: list[float]) -> list[float]:
    return [
        float(values[0]),
        float(values[1]),
        float(values[2] - values[0]),
        float(values[3] - values[1]),
    ]


def build_python_debug(model_path: Path, image_path: Path) -> dict[str, object]:
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics.utils import nms, ops
    from ultralytics.data.augment import LetterBox

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
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
    tensor = np.ascontiguousarray(rgb.transpose((2, 0, 1))[None]).astype(np.float32) / 255.0

    raw_outputs = session.run(None, {input_info.name: tensor})
    if len(raw_outputs) != 2:
        raise RuntimeError(f"expected 2 outputs, got {len(raw_outputs)}")

    prediction = next(output for output in raw_outputs if output.ndim == 3)
    proto = next(output for output in raw_outputs if output.ndim == 4)

    prediction_tensor = torch.from_numpy(prediction.copy())
    proto_tensor = torch.from_numpy(proto.copy())

    detections = nms.non_max_suppression(
        prediction_tensor,
        conf_thres=0.25,
        iou_thres=0.45,
        nc=80,
        agnostic=False,
        max_det=300,
    )[0]

    decoded_candidates: list[dict[str, object]] = []
    raw_pred = prediction[0]
    proposal_count = raw_pred.shape[1]
    stride = raw_pred.shape[0]
    class_count = stride - 4 - proto.shape[1]
    for index in range(proposal_count):
        scores = raw_pred[4 : 4 + class_count, index]
        class_id = int(np.argmax(scores))
        score = float(scores[class_id])
        cx = float(raw_pred[0, index])
        cy = float(raw_pred[1, index])
        w = float(raw_pred[2, index])
        h = float(raw_pred[3, index])
        decoded_candidates.append(
            {
                "class_id": class_id,
                "score": score,
                "bbox_xywh": [cx - w / 2.0, cy - h / 2.0, w, h],
                "bbox_xyxy": [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                "mask_coefficients": [
                    float(value)
                    for value in raw_pred[4 + class_count :, index].tolist()
                ],
            }
        )

    decoded_candidates.sort(key=lambda item: item["score"], reverse=True)
    confidence_filtered = [item for item in decoded_candidates if item["score"] >= 0.25]

    nms_candidates: list[dict[str, object]] = []
    projection_logits: list[float] = []
    final_instances: list[dict[str, object]] = []

    if len(detections) > 0:
        coeffs = detections[:, 6:]
        proto_chw = proto_tensor[0]
        logits = (coeffs[0] @ proto_chw.reshape(proto_chw.shape[0], -1).float()).reshape(
            proto_chw.shape[1], proto_chw.shape[2]
        )
        projection_logits = [float(value) for value in logits.reshape(-1).tolist()]

        detections_scaled = detections.clone()
        detections_scaled[:, :4] = ops.scale_boxes(
            (height, width), detections_scaled[:, :4], original.shape
        )
        masks_native = ops.process_mask_native(
            proto_chw,
            coeffs,
            detections_scaled[:, :4],
            original.shape[:2],
        )

        for det_index in range(detections.shape[0]):
            row = detections[det_index]
            nms_candidates.append(
                {
                    "class_id": int(row[5].item()),
                    "score": float(row[4].item()),
                    "bbox_xyxy": [float(value) for value in row[:4].tolist()],
                    "bbox_xywh": xyxy_to_xywh([float(value) for value in row[:4].tolist()]),
                    "mask_coefficients": [float(value) for value in row[6:].tolist()],
                }
            )

            scaled = detections_scaled[det_index]
            binary_mask = masks_native[det_index].cpu().numpy().astype(np.uint8)
            flat_mask = binary_mask.reshape(-1).tolist()
            final_instances.append(
                {
                    "class_id": int(scaled[5].item()),
                    "score": float(scaled[4].item()),
                    "bbox_xywh": xyxy_to_xywh([float(value) for value in scaled[:4].tolist()]),
                    "mask": {
                        "size": [int(binary_mask.shape[1]), int(binary_mask.shape[0])],
                        "area": int(binary_mask.sum()),
                        "rle": encode_rle([int(value) for value in flat_mask]),
                    },
                }
            )

    return {
        "task": "seg",
        "image": str(image_path),
        "preprocess": {
            "tensor_shape": [int(value) for value in tensor.shape],
            "tensor_values": [float(value) for value in tensor.reshape(-1).tolist()],
            "record": {
                "source_size": [int(original.shape[1]), int(original.shape[0])],
                "target_size": [int(width), int(height)],
                "resized_size": [int(width), int(height)],
                "resize_scale": [
                    float(min(width / original.shape[1], height / original.shape[0])),
                    float(min(width / original.shape[1], height / original.shape[0])),
                ],
                "padding": {
                    "left": int(round((width - original.shape[1] * min(width / original.shape[1], height / original.shape[0])) / 2 - 0.1)),
                    "top": int(round((height - original.shape[0] * min(width / original.shape[1], height / original.shape[0])) / 2 - 0.1)),
                    "right": int(width - round(original.shape[1] * min(width / original.shape[1], height / original.shape[0])) - int(round((width - original.shape[1] * min(width / original.shape[1], height / original.shape[0])) / 2 - 0.1))),
                    "bottom": int(height - round(original.shape[0] * min(width / original.shape[1], height / original.shape[0])) - int(round((height - original.shape[0] * min(width / original.shape[1], height / original.shape[0])) / 2 - 0.1))),
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
        "mask_projection": {
            "proto_size": [int(proto.shape[3]), int(proto.shape[2])],
            "top_candidate_logits": projection_logits,
        },
        "final": {
            "instances": final_instances,
        },
    }


def ppm_inputs(image_path: Path, scratch_dir: Path) -> Path:
    from PIL import Image

    ppm_path = scratch_dir / f"{image_path.stem}.ppm"
    Image.open(image_path).convert("RGB").save(ppm_path, format="PPM")
    return ppm_path


def build_cpp_debug(tool_path: Path, model_path: Path, image_path: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="yolo_cpp_seg_debug_") as scratch:
        ppm_path = ppm_inputs(image_path, Path(scratch))
        completed = subprocess.run(
            [str(tool_path), str(model_path), str(ppm_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    json_start = completed.stdout.find('{"task"')
    if json_start < 0:
        raise RuntimeError(
            f"C++ segmentation debug tool did not emit JSON. stderr was:\n{completed.stderr}"
        )
    return json.loads(completed.stdout[json_start:])


def max_abs_diff(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs):
        return math.inf
    if not lhs:
        return 0.0
    return max(abs(left - right) for left, right in zip(lhs, rhs))


def compare_debug(python_payload: dict[str, object], cpp_payload: dict[str, object]) -> dict[str, object]:
    comparisons: list[dict[str, object]] = []

    def record(stage: str, ok: bool, detail: str) -> None:
        comparisons.append({"stage": stage, "ok": ok, "detail": detail})

    preprocess_diff = max_abs_diff(
        python_payload["preprocess"]["tensor_values"],
        cpp_payload["preprocess"]["tensor_values"],
    )
    record("preprocess", preprocess_diff <= 1e-4, f"max_abs_diff={preprocess_diff}")

    python_raw = python_payload["raw_outputs"]
    cpp_raw = cpp_payload["raw_outputs"]
    raw_ok = len(python_raw) == len(cpp_raw)
    raw_detail = [f"count={len(python_raw)}:{len(cpp_raw)}"]
    if raw_ok:
        for index, (py_output, cpp_output) in enumerate(zip(python_raw, cpp_raw)):
            diff = max_abs_diff(py_output["values"], cpp_output["values"])
            raw_detail.append(f"{index}:{diff}")
            if diff > 1e-4:
                raw_ok = False
    record("raw_outputs", raw_ok, ", ".join(raw_detail))

    python_nms = python_payload["nms"]["candidates"]
    cpp_nms = cpp_payload["nms"]["candidates"]
    nms_ok = bool(python_nms) and bool(cpp_nms)
    if nms_ok:
        py_top = python_nms[0]
        cpp_top = cpp_nms[0]
        bbox_diff = max_abs_diff(py_top["bbox_xywh"], cpp_top["bbox_xywh"])
        score_diff = abs(py_top["score"] - cpp_top["score"])
        class_ok = py_top["class_id"] == cpp_top["class_id"]
        nms_ok = class_ok and bbox_diff <= 8.0 and score_diff <= 0.2
        record(
            "nms",
            nms_ok,
            f"class_match={class_ok}, bbox_max_abs_diff={bbox_diff}, score_diff={score_diff}",
        )
    else:
        record("nms", False, "missing top candidate in python or cpp payload")

    projection_diff = max_abs_diff(
        python_payload["mask_projection"]["top_candidate_logits"],
        cpp_payload["mask_projection"]["top_candidate_logits"],
    )
    record(
        "mask_projection",
        projection_diff <= 1e-3,
        f"max_abs_diff={projection_diff}",
    )

    python_final = python_payload["final"]["instances"]
    cpp_final = cpp_payload["final"]["instances"]
    final_ok = bool(python_final) and bool(cpp_final)
    if final_ok:
        py_top = python_final[0]
        cpp_top = cpp_final[0]
        area_diff = abs(py_top["mask"]["area"] - cpp_top["mask"]["area"])
        bbox_diff = max_abs_diff(py_top["bbox_xywh"], cpp_top["bbox_xywh"])
        score_diff = abs(py_top["score"] - cpp_top["score"])
        class_ok = py_top["class_id"] == cpp_top["class_id"]
        final_ok = class_ok and bbox_diff <= 8.0 and score_diff <= 0.2 and area_diff <= 2000
        record(
            "final",
            final_ok,
            f"class_match={class_ok}, bbox_max_abs_diff={bbox_diff}, score_diff={score_diff}, area_diff={area_diff}",
        )
    else:
        record("final", False, "missing final instance in python or cpp payload")

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
    tool_path = Path(args.build_dir) / "yolo_cpp_segmentation_debug_dump"

    if not model_path.exists():
        raise SystemExit(f"missing model: {model_path}")
    if not image_path.exists():
        raise SystemExit(f"missing image: {image_path}")
    if not tool_path.exists():
        raise SystemExit(f"missing tool: {tool_path}")

    python_payload = build_python_debug(model_path, image_path)
    cpp_payload = build_cpp_debug(tool_path, model_path, image_path)
    comparison_payload = compare_debug(python_payload, cpp_payload)

    write_json(output_dir / "seg_python_debug.json", python_payload)
    write_json(output_dir / "seg_cpp_debug.json", cpp_payload)
    write_json(output_dir / "seg_debug_comparison.json", comparison_payload)

    first_fail = comparison_payload["first_fail"]
    if first_fail is None:
        print("segmentation debug comparison: all stages within tolerance")
    else:
        print(f"segmentation debug comparison: first_fail={first_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
