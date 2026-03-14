from __future__ import annotations

import json
from pathlib import Path

TASKS = ("detect", "classify", "seg", "pose", "obb")

STAGE_PREPROCESS = "preprocess"
STAGE_RAW_OUTPUTS = "raw_outputs"
STAGE_DECODED = "decoded"
STAGE_CONFIDENCE_FILTERED = "confidence_filtered"
STAGE_NMS = "nms"
STAGE_FINAL = "final"
STAGE_MASK_PROJECTION = "mask_projection"
STAGE_CANONICALIZED = "canonicalized"

COMMON_DEBUG_STAGES = (
    STAGE_PREPROCESS,
    STAGE_RAW_OUTPUTS,
    STAGE_DECODED,
    STAGE_CONFIDENCE_FILTERED,
    STAGE_NMS,
    STAGE_FINAL,
)
SEGMENTATION_DEBUG_STAGES = COMMON_DEBUG_STAGES[:-1] + (
    STAGE_MASK_PROJECTION,
    STAGE_FINAL,
)
POSE_DEBUG_STAGES = COMMON_DEBUG_STAGES
OBB_DEBUG_STAGES = (
    STAGE_PREPROCESS,
    STAGE_RAW_OUTPUTS,
    STAGE_DECODED,
    STAGE_CANONICALIZED,
    STAGE_CONFIDENCE_FILTERED,
    STAGE_NMS,
    STAGE_FINAL,
)

PARITY_TOLERANCES: dict[str, dict[str, float]] = {
    "detect": {"count_delta": 2.0, "score_delta": 0.15, "bbox_delta": 6.0},
    "classify": {"score_delta": 0.1},
    "seg": {
        "count_delta": 1.0,
        "score_delta": 0.2,
        "bbox_delta": 8.0,
        "mask_area_delta": 2000.0,
        "mask_iou_min": 0.5,
    },
    "pose": {
        "count_delta": 1.0,
        "score_delta": 0.2,
        "bbox_delta": 8.0,
        "keypoint_point_delta": 8.0,
        "keypoint_score_delta": 0.25,
    },
    "obb": {
        "count_delta": 1.0,
        "score_delta": 0.15,
        "center_delta": 6.0,
        "size_delta": 6.0,
        "angle_delta": 0.15,
        "corner_delta": 8.0,
    },
}

DEBUG_TOLERANCES: dict[str, dict[str, float]] = {
    "seg": {
        "preprocess": 1e-4,
        "raw_outputs": 1e-4,
        "bbox_delta": 8.0,
        "score_delta": 0.2,
        "mask_projection": 1e-3,
        "mask_area_delta": 2000.0,
    },
    "pose": {
        "preprocess": 1e-4,
        "raw_outputs": 1e-3,
        "bbox_delta": 8.0,
        "score_delta": 0.2,
        "keypoint_point_delta": 8.0,
        "keypoint_score_delta": 0.25,
    },
    "obb": {
        "preprocess": 1e-4,
        "raw_outputs": 1e-3,
        "center_delta": 8.0,
        "size_delta": 8.0,
        "angle_delta": 0.2,
        "score_delta": 0.2,
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_build_dir() -> Path:
    return repo_root() / "build" / "dev"


def default_output_dir() -> Path:
    return repo_root() / "tests" / "assets" / "baselines" / "parity"


def parity_dir() -> Path:
    return default_output_dir()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parity_summary_path(task: str, side: str, output_dir: Path | None = None) -> Path:
    root = output_dir if output_dir is not None else default_output_dir()
    return root / f"{task}_{side}.json"


def debug_dump_path(task: str, side: str, output_dir: Path | None = None) -> Path:
    root = output_dir if output_dir is not None else default_output_dir()
    return root / f"{task}_{side}_debug.json"


def debug_comparison_path(task: str, output_dir: Path | None = None) -> Path:
    root = output_dir if output_dir is not None else default_output_dir()
    return root / f"{task}_debug_comparison.json"


def manifest_path(output_dir: Path | None = None) -> Path:
    root = output_dir if output_dir is not None else default_output_dir()
    return root / "manifest.json"


def ppm_input_path(image_path: Path, scratch_dir: Path) -> Path:
    return scratch_dir / f"{image_path.stem}.ppm"


def parity_skip_message(task: str) -> str:
    return f"run_parity.py has not generated {task} parity JSON yet"


def missing_asset_message(kind: str, path: Path) -> str:
    return f"missing {kind}: {path}"


def skipped_manifest_entry(reason: str, detail: str | None = None) -> dict[str, str]:
    entry = {
        "status": "skipped",
        "reason": reason,
    }
    if detail is not None:
        entry["detail"] = detail
    return entry


def ok_manifest_entry(task: str) -> dict[str, str]:
    return {
        "status": "ok",
        "python": parity_summary_path(task, "python").name,
        "cpp": parity_summary_path(task, "cpp").name,
    }


def extract_json_payload(stdout: str, stderr: str, tool_name: str) -> dict[str, object]:
    json_start = stdout.find('{"task"')
    if json_start < 0:
        raise RuntimeError(f"{tool_name} did not emit JSON. stderr was:\n{stderr}")
    return json.loads(stdout[json_start:])


def parity_tolerances(task: str) -> dict[str, float]:
    return PARITY_TOLERANCES[task]


def debug_tolerances(task: str) -> dict[str, float]:
    return DEBUG_TOLERANCES[task]
