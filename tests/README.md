# Test Layers

This repo uses a layered test strategy so that fast model-free checks stay
available even when ONNX assets are missing.

## Layers

- `tests/unit`: helper contracts, shared semantics, and small pure-data logic
- `tests/component`: decode, preprocess, postprocess, and task-runtime behavior
  with synthetic tensors
- `tests/adapter`: Ultralytics binding/probe rules and task hard constraints
- `tests/integration`: model-backed end-to-end smoke tests
- `tests/parity`: manual Python-vs-C++ comparisons using Ultralytics

## Gate Policy

### Baseline C++ Gate

The baseline gate is intended to run everywhere:

- configure with `cmake --preset dev --fresh` or `cmake --preset ci --fresh`
- build `yolo_cpp_tests`
- run model-free `ctest`

This gate should not depend on model assets or Python parity dependencies.

### Full Model Gate

The full gate extends the baseline with:

- integration smoke tests for checked-in ONNX assets
- parity JSON generation and parity unit checks
- staged debug runners for first-fail analysis

This gate is what the repo treats as release-readiness coverage.

## Asset Gate Rules

- `unit`, `component`, and `adapter` tests are model-free by design
- integration tests are only added when the corresponding model asset exists
  under `tests/assets/models`
- parity runners report missing models as explicit skips instead of failing the
  regular `ctest` suite

## Checked-In Assets

Current model-backed tasks use:

- `tests/assets/models/yolov8n.onnx`
- `tests/assets/models/yolov8n-cls.onnx`
- `tests/assets/models/yolov8n-seg.onnx`
- `tests/assets/models/yolov8n-pose.onnx`
- `tests/assets/models/yolov8n-obb.onnx`

## Parity And Debug UX

The common parity entrypoint is:

```bash
uv run python tests/parity/run_parity.py --check
```

Task-specific staged debug entrypoints are:

- `uv run python tests/parity/run_segmentation_debug.py`
- `uv run python tests/parity/run_pose_debug.py`
- `uv run python tests/parity/run_obb_debug.py`

Common staged debug names:

- common stages: `preprocess`, `raw_outputs`, `decoded`, `confidence_filtered`,
  `nms`, `final`
- segmentation adds: `mask_projection`
- obb adds: `canonicalized`

Output artifacts are written under `tests/assets/baselines/parity/`.

## Residual Drift Policy

- drift that changes final task contract is release-blocking
- drift isolated to earlier staged-debug layers is documented and tracked, but
  may be accepted if parity and integration remain aligned
- current accepted examples are summarized in the repo
  [`README.md`](../README.md)
