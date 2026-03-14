#Parity Checks

These parity checks stay out of CTest on purpose. They are a lightweight manual
runner for comparing this repo's C++ pipeline against Ultralytics Python on a
small set of detect / classify / segmentation / pose / OBB assets.

Dependencies:

- `ultralytics`
- `onnx`
- `onnxruntime`
- `pillow`
- `numpy`

Current scope:

- detection parity: `tests/assets/models/yolov8n.onnx` on `test1.jpg` and `test2.jpg`
- classification parity: `tests/assets/models/yolov8n-cls.onnx` on `test1.jpg` and `test2.jpg`
- segmentation parity: `tests/assets/models/yolov8n-seg.onnx` on `test1.jpg` when that model exists
- pose parity: `tests/assets/models/yolov8n-pose.onnx` on `test2.jpg`
- obb parity: `tests/assets/models/yolov8n-obb.onnx` on `test2.jpg`

Current status:

- detect parity: aligned
- classify parity: aligned
- segmentation parity: runner and staged debug dump are available once
  `tests/assets/models/yolov8n-seg.onnx` exists
- pose parity: runner and staged debug dump are available on
  `tests/assets/models/yolov8n-pose.onnx`
- obb parity: runner and staged debug dump are available on
  `tests/assets/models/yolov8n-obb.onnx`

Acceptance rule:

- parity is release-blocking when the generated parity unit tests pass on the
  checked-in assets
- staged debug is diagnostic-first; a first-fail report only blocks release if
  it corresponds to a final-result regression
- current accepted residual drift is summarized in the repo
  [`README.md`](../../README.md)

Runner flow:

1. Build the JSON dump tool:
   `cmake --build build/dev --target yolo_cpp_parity_dump`
2. Generate Python and C++ summaries:
   `uv run python tests/parity/run_parity.py --check`
3. Inspect `tests/assets/baselines/parity/manifest.json` plus the generated
   `*_python.json` and `*_cpp.json` files.

Common staged debug stage names:

- common stages: `preprocess`, `raw_outputs`, `decoded`,
  `confidence_filtered`, `nms`, `final`
- segmentation adds: `mask_projection`
- obb adds: `canonicalized`

Output layout:

- parity summaries: `tests/assets/baselines/parity/<task>_python.json`
- parity summaries: `tests/assets/baselines/parity/<task>_cpp.json`
- staged debug dumps: `tests/assets/baselines/parity/<task>_python_debug.json`
- staged debug dumps: `tests/assets/baselines/parity/<task>_cpp_debug.json`
- staged debug comparisons:
  `tests/assets/baselines/parity/<task>_debug_comparison.json`

Segmentation staged debug dump:

1. Build the debug tool:
   `cmake --build build/dev --target yolo_cpp_segmentation_debug_dump`
2. Generate Python/C++ staged dumps and a first-fail summary:
   `uv run python tests/parity/run_segmentation_debug.py`
3. Inspect:
   `tests/assets/baselines/parity/seg_python_debug.json`
   `tests/assets/baselines/parity/seg_cpp_debug.json`
   `tests/assets/baselines/parity/seg_debug_comparison.json`

Pose staged debug dump:

1. Build the debug tool:
   `cmake --build build/dev --target yolo_cpp_pose_debug_dump`
2. Generate Python/C++ staged dumps and a first-fail summary:
   `uv run python tests/parity/run_pose_debug.py`
3. Inspect:
   `tests/assets/baselines/parity/pose_python_debug.json`
   `tests/assets/baselines/parity/pose_cpp_debug.json`
   `tests/assets/baselines/parity/pose_debug_comparison.json`

OBB staged debug dump:

1. Build the debug tool:
   `cmake --build build/dev --target yolo_cpp_obb_debug_dump`
2. Generate Python/C++ staged dumps and a first-fail summary:
   `uv run python tests/parity/run_obb_debug.py`
3. Inspect:
   `tests/assets/baselines/parity/obb_python_debug.json`
   `tests/assets/baselines/parity/obb_cpp_debug.json`
   `tests/assets/baselines/parity/obb_debug_comparison.json`

Notes:

- The runner converts JPG test images to temporary PPM inputs for the C++ tool.
- Tolerances are intentionally loose enough to absorb exporter and ORT minor-version drift.
- Missing models or missing Python dependencies are reported as explicit skips instead of affecting regular `ctest`.
- `uv run ...` is the supported entrypoint for parity tooling in this repo.
