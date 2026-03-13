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
cmake --preset dev
```

Build:

```bash
cmake --build build/dev
```

This will:

- install the `cpu` manifest feature from `vcpkg.json`
- pull `onnxruntime`
- generate `build/dev/compile_commands.json`
- symlink `build/compile_commands.json` for `clangd`

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
