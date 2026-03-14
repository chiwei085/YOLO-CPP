# YOLO - CPP

Minimal YOLO runtime in C++20 with a small facade API and task coverage for
detect / classify / seg / pose / obb.

## Supported Tasks

- supports `detect`, `classify`, `seg`, `pose`, and `obb`
- includes integration tests and parity tooling for checked-in assets
- examples use a simple `PPM` (`P5`/`P6`) loader instead of OpenCV

## Prerequisites

- CMake 3.20+
- Ninja
- a C++20 compiler
- `vcpkg`

This repo assumes `VCPKG_ROOT` points at your local `vcpkg` checkout. The
presets are written for:

```bash
export VCPKG_ROOT="$HOME/.local/share/vcpkg"
```

`VCPKG_OVERLAY_PORTS` and `VCPKG_BINARY_SOURCES=clear` are already wired into
the repo presets, so you do not need to export them manually when using
`cmake --preset ...`. The binary-cache setting is intentional here: it keeps
the local ONNX overlay from being replaced by a stale cached package.

## Build

### CPU

Configure:

```bash
cmake --preset dev --fresh
```

Build:

```bash
cmake --build build/dev
```

This preset uses:

- `YOLO_CPP_ORT_PROVIDER=cpu`
- `VCPKG_TARGET_TRIPLET=x64-linux-dynamic`
- `VCPKG_MANIFEST_FEATURES=cpu`
- `VCPKG_BINARY_SOURCES=clear`

The recommended baseline is shared ONNX Runtime via `x64-linux-dynamic` with
the local overlay port in [`vcpkg-overlay-ports/onnx`](vcpkg-overlay-ports/onnx).
That combination avoids the static ONNX registration conflict that breaks model
loading in the default static package combination.

### CUDA

Configure:

```bash
cmake --preset dev-cuda
```

Build:

```bash
cmake --build build/cuda
```

### Library-Only

```bash
cmake --preset dev -DYOLO_CPP_BUILD_EXAMPLES=OFF
cmake --build build/dev
```

## Examples

Built executables:

- `build/dev/detect_image`
- `build/dev/classify_image`

Detect:

```bash
./build/dev/detect_image /path/to/model.onnx /path/to/image.ppm
```

Classify:

```bash
./build/dev/classify_image /path/to/model.onnx /path/to/image.ppm
```

If your input is `jpg`/`png`, convert it first:

```bash
magick input.jpg output.ppm
```

## Tests

Testing is enabled by default with `BUILD_TESTING=ON`.

Build and run:

```bash
cmake --preset dev --fresh
cmake --build build/dev --target check
```

Or:

```bash
ctest --test-dir build/dev --output-on-failure
```

Test layers:

- `tests/unit`: helper contracts and semantics
- `tests/component`: decode, preprocess, postprocess behavior with synthetic data
- `tests/adapter`: Ultralytics probe/binding behavior and task constraints
- `tests/integration`: model-backed pipeline tests
- `tests/parity`: manual parity checks against Ultralytics Python

`unit` / `component` / `adapter` tests are model-free. Integration tests are
only added when the required ONNX assets exist under `tests/assets/models/`.

More detail lives in [`tests/README.md`](tests/README.md).

## Notes

- parity checks are available through the Python tooling under
  [`tests/parity`](tests/parity)
- integration tests are only added when the required ONNX assets exist under
  `tests/assets/models/`
- `pose` and `obb` include extra debug helpers for parity investigation on the
  checked-in assets

Run parity manually with the project test environment:

```bash
uv run python tests/parity/run_parity.py --check
```

Run the staged segmentation debug dump:

```bash
uv run python tests/parity/run_segmentation_debug.py
```

Run the staged pose debug dump:

```bash
uv run python tests/parity/run_pose_debug.py
```

Run the staged OBB debug dump:

```bash
uv run python tests/parity/run_obb_debug.py
```

More detail lives in [`tests/parity/README.md`](tests/parity/README.md) and
[`docs/vcpkg_overlay_onnx_fix.md`](docs/vcpkg_overlay_onnx_fix.md). Additional
build notes for ONNX Runtime packaging live in
[`docs/ort_static_vs_shared_plan.md`](docs/ort_static_vs_shared_plan.md).

## Public API

Umbrella header:

```cpp
#include "yolo/yolos.hpp"
```

Main facade:

```cpp
#include "yolo/facade.hpp"
```

Minimal usage:

```cpp
#include "yolo/facade.hpp"

auto pipeline_result = yolo::create_pipeline(yolo::ModelSpec{
    .path = "model.onnx",
});

if (!pipeline_result.ok()) {
    yolo::throw_if_error(pipeline_result.error);
}

const auto& pipeline = *pipeline_result.value;
```

From there you can inspect `pipeline->info()` and call task-specific entrypoints
such as `detect`, `classify`, `segment`, `detect_pose`, `detect_obb`, or
`run_raw`.
