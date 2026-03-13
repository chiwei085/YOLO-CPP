from __future__ import annotations

import json
import os
from pathlib import Path


def parity_dir() -> Path:
    env_dir = os.environ.get("YOLO_CPP_PARITY_DIR")
    if env_dir:
        return Path(env_dir)

    return Path(__file__).resolve().parents[1] / "assets" / "baselines" / "parity"


def load_pair(task: str) -> tuple[dict, dict] | tuple[None, None]:
    root = parity_dir()
    cpp_path = root / f"{task}_cpp.json"
    python_path = root / f"{task}_python.json"
    if not cpp_path.exists() or not python_path.exists():
        return None, None

    return (
        json.loads(python_path.read_text(encoding="utf-8")),
        json.loads(cpp_path.read_text(encoding="utf-8")),
    )


def decode_rle(mask: dict) -> list[int]:
    size = mask.get("size", [0, 0])
    total = int(size[0]) * int(size[1])
    values: list[int] = []
    current = 0
    for run in mask.get("rle", []):
        values.extend([current] * int(run))
        current = 1 - current
    if len(values) < total:
        values.extend([0] * (total - len(values)))
    return values[:total]


def mask_iou(lhs: dict, rhs: dict) -> float:
    lhs_values = decode_rle(lhs)
    rhs_values = decode_rle(rhs)
    overlap = sum(1 for a, b in zip(lhs_values, rhs_values) if a and b)
    union = sum(1 for a, b in zip(lhs_values, rhs_values) if a or b)
    if union == 0:
        return 1.0
    return overlap / union
