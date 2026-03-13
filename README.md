# YOLO-CPP

Minimal YOLO runtime in C++20

## Status

Current milestones implemented:

- runtime core and ONNX session abstraction
- image preprocess contract
- task skeletons, with `detect` and `classify` usable
- Ultralytics adapter probing for `detect`, `classify`, and `seg`
- facade-based loading and inference entry points

Current examples are intentionally small and currently use a simple `PPM` loader (`P5`/`P6`) instead of OpenCV.

## Prerequisites

- CMake 3.20+
- Ninja
- a C++20 compiler
- `vcpkg`

This repo is set up for `vcpkg` manifest mode and expects `VCPKG_ROOT` to point to your local `vcpkg` checkout.

The provided presets assume:

```bash
export VCPKG_ROOT="$HOME/.local/share/vcpkg"
```

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

This will:

- install the `cpu` manifest feature from `vcpkg.json`
- pull `onnxruntime` with the `x64-linux-dynamic` triplet
- generate `build/dev/compile_commands.json`
- symlink `build/compile_commands.json` for `clangd`

The `dev` preset intentionally uses shared ONNX Runtime. The previous
`x64-linux` static triplet pulled standalone ONNX archives into the process and
triggered duplicate schema registration during model-backed integration tests.

For vcpkg-based development, the currently verified working baseline is:

- shared ORT via `x64-linux-dynamic`
- the local ONNX overlay port under
  [`vcpkg-overlay-ports/onnx/portfile.cmake`](vcpkg-overlay-ports/onnx/portfile.cmake)

The overlay forces `ONNX_DISABLE_STATIC_REGISTRATION=ON`, which restores clean
model loading and unblocks the model-backed integration tests. See
[`docs/vcpkg_overlay_onnx_fix.md`](docs/vcpkg_overlay_onnx_fix.md)
for reinstall and verification steps.

### CUDA

Configure:

```bash
cmake --preset dev-cuda
```

Build:

```bash
cmake --build build/cuda
```

This switches the manifest feature to `cuda` and requests `onnxruntime-gpu`.

## Disable Examples

Examples are enabled by default.

To build library-only:

```bash
cmake --preset dev -DYOLO_CPP_BUILD_EXAMPLES=OFF
cmake --build build/dev
```

## Run Examples

The examples build these executables:

- `build/dev/detect_image`
- `build/dev/classify_image`

### Detect

```bash
./build/dev/detect_image /path/to/model.onnx /path/to/image.ppm
```

### Classify

```bash
./build/dev/classify_image /path/to/model.onnx /path/to/image.ppm
```

Both examples:

- load a model through the high-level facade
- auto-probe a supported Ultralytics adapter
- run inference
- print a small result preview

## Tests

Testing is enabled by default through CMake's `BUILD_TESTING=ON`.

Configure and build:

```bash
cmake --preset dev --fresh
cmake --build build/dev --target check
```

Or run tests directly:

```bash
cd build/dev
ctest --output-on-failure
```

### Test Layout

- `tests/unit`: pure helper and semantics tests
- `tests/component`: decode/postprocess behavior tests with synthetic tensors
- `tests/adapter`: Ultralytics binding/probe tests and task hard constraints
- `tests/integration`: model-backed pipeline tests
- `tests/parity`: parity runner skeletons, not registered in CTest yet

### Model-Free Tests

These tests do not require exported ONNX models:

- all `unit`, `component`, and `adapter` tests

They are intended to stay fast and rely on synthetic tensors and synthetic
metadata rather than real models.

### Model-Backed Integration Tests

Integration tests are only added when both of these files exist:

- `tests/assets/models/yolov8n.onnx`
- `tests/assets/models/yolov8n-cls.onnx`

They also require ONNX Runtime to be available in the configured build.
The supported integration path is the `dev` preset with the ONNX overlay
baseline:

- `VCPKG_TARGET_TRIPLET=x64-linux-dynamic`
- `VCPKG_OVERLAY_PORTS=./vcpkg-overlay-ports`

If the model assets are missing, the regular non-integration test suite still
builds and runs normally.

## Example Input Format

The current example loader supports only:

- `P6` RGB PPM
- `P5` grayscale PPM

If your source image is `png`/`jpg`, convert it first, for example:

```bash
magick input.jpg output.ppm
```

## Public Entry Point

The umbrella header is:

```cpp
#include "yolo/yolos.hpp"
```

The main high-level API is:

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
- call `pipeline->run_raw(image)`

## Notes

- `detect` and `classify` are the main usable paths right now.
- `seg` has adapter probing, but task decode/postprocess is still stubbed.
- `pose` and `obb` are still framework-level stubs.
- examples are intended as smoke tests and API demonstrations, not benchmarks.
