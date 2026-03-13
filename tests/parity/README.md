# Parity Checks

These parity checks stay out of CTest on purpose. They are a lightweight manual
runner for comparing this repo's C++ pipeline against Ultralytics Python on a
small set of detect / classify / segmentation assets.

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

Current status:

- detect parity: aligned
- classify parity: aligned
- segmentation parity: pending once `tests/assets/models/yolov8n-seg.onnx` is available

Runner flow:

1. Build the JSON dump tool:
   `cmake --build build/dev --target yolo_cpp_parity_dump`
2. Generate Python and C++ summaries:
   `.venv-tests/bin/python tests/parity/run_parity.py --check`
3. Inspect `tests/assets/baselines/parity/manifest.json` plus the generated
   `*_python.json` and `*_cpp.json` files.

Notes:

- The runner converts JPG test images to temporary PPM inputs for the C++ tool.
- Tolerances are intentionally loose enough to absorb exporter and ORT minor-version drift.
- Missing models or missing Python dependencies are reported as explicit skips instead of affecting regular `ctest`.
