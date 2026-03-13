#!/usr/bin/env python3

from pathlib import Path
import json
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    models_dir = root / "tests" / "assets" / "models"
    baselines_dir = root / "tests" / "assets" / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "status": "skeleton",
        "models_dir": str(models_dir),
        "message": (
            "Parity runner is not implemented yet. Add Ultralytics-backed "
            "reference inference here for detect/classify."
        ),
    }
    (baselines_dir / "parity_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest["message"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
