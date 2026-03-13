# YOLO-CPP

Minimal YOLO runtime in C++20, with a small facade API, Ultralytics-oriented
adapter probing, and a layered test suite around detect / classify / seg.

## Current State

- `detect`: usable and parity-aligned with Ultralytics Python on the current
  parity assets
- `classify`: usable and parity-aligned with Ultralytics Python on the current
  parity assets
- `seg`: binding-driven runtime is in place; integration/parity depend on a
  local segmentation ONNX asset
- `pose` / `obb`: not part of the current parity/integration mainline

The examples are intentionally small and currently use a simple `PPM`
(`P5`/`P6`) loader instead of OpenCV.

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

## Recommended Build Baseline

The verified `vcpkg` baseline for development and model-backed tests is:

- shared ONNX Runtime via `x64-linux-dynamic`
- the local ONNX overlay port in [`vcpkg-overlay-ports/onnx`](vcpkg-overlay-ports/onnx)

This avoids the static ONNX registration conflict that breaks model loading in
the default static package combination.

See:

- [`docs/vcpkg_overlay_onnx_fix.md`](docs/vcpkg_overlay_onnx_fix.md)
- [`docs/ort_static_vs_shared_plan.md`](docs/ort_static_vs_shared_plan.md)

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

## Testing

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

## Parity

Current parity status:

- detect: aligned
- classify: aligned
- segmentation: runner ready; requires `tests/assets/models/yolov8n-seg.onnx`

Run parity manually with the project test environment:

```bash
.venv-tests/bin/python tests/parity/run_parity.py --check
```

More detail lives in [`tests/parity/README.md`](tests/parity/README.md).

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

From there you can:

- inspect `pipeline->info()`
- call `pipeline->detect(image)`
- call `pipeline->classify(image)`
- call `pipeline->segment(image)`
- call `pipeline->run_raw(image)`
