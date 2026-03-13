# Vcpkg Overlay for ONNX Static Registration

## Background

This repo reproduced duplicate ONNX schema registration in two stages:

- the original static `x64-linux` package shape linked standalone ONNX archives
  directly into the final binary
- even after moving the main development path to shared ORT
  (`x64-linux-dynamic`), model loading still failed during `Ort::Session`
  creation

The remaining failure was traced to the vcpkg package layer:

- `onnxruntime` warns that the `onnx` port must be built with
  `ONNX_DISABLE_STATIC_REGISTRATION=ON`
- the stock vcpkg `onnx` port still builds as a static library with
  `ONNX_DISABLE_STATIC_REGISTRATION=OFF`

That package combination is enough to reproduce duplicate schema registration
when loading YOLO ONNX models.

## Symptom

The failure mode looked like this:

- `Ort::Env` creation succeeded
- `Ort::Session(env, model_path, options)` failed while loading a YOLO ONNX
  model
- stderr showed repeated `Schema error: Trying to register schema ... but it is
  already registered ...`

At the repo level, that surfaced as model-backed integration tests being built
but skipped because `create_pipeline(...)` failed during model load.

## Why Use an Overlay Port

The fix belongs in the dependency package definition, not in application code.

This repo therefore provides a local overlay port at
[`vcpkg-overlay-ports/onnx/portfile.cmake`](../vcpkg-overlay-ports/onnx/portfile.cmake)
that is based on the current upstream vcpkg `onnx` port with one minimal change:

- add `-DONNX_DISABLE_STATIC_REGISTRATION=ON` to `vcpkg_cmake_configure(...)`

The overlay does not:

- switch ONNX to a shared library
- patch production inference code
- change tests or assertions
- change adapter/runtime behavior

## Why Shared ORT Was Only a Partial Fix

Switching the repo to `x64-linux-dynamic` was still the right move:

- it removed the bad final-binary linkage shape that directly pulled
  standalone ONNX archives into the executable
- it made the minimal `Ort::Env` probe clean

But shared ORT alone was not enough, because the stock vcpkg `onnx` package was
still built as:

- static ONNX library
- `ONNX_DISABLE_STATIC_REGISTRATION=OFF`

That meant the schema-registration conflict moved from obvious link-time shape
to model-load time inside `Ort::Session`.

## Reinstall with the Overlay

From the repo root:

```bash
vcpkg remove --classic \
  onnx:x64-linux-dynamic onnxruntime:x64-linux-dynamic

vcpkg install --classic \
  onnxruntime:x64-linux-dynamic \
  --overlay-ports="$PWD/vcpkg-overlay-ports"
```

This keeps the experiment at the package layer first, before re-running the
full repo build.

## Verify the Configure Cache

After installation, confirm that the overlay actually changed the ONNX build:

```bash
grep -RIn "ONNX_DISABLE_STATIC_REGISTRATION" \
  "$HOME/.local/share/vcpkg/buildtrees/onnx" | head -n 20
```

The expected result is a cache entry like:

```text
ONNX_DISABLE_STATIC_REGISTRATION:BOOL=ON
```

If the value is still `OFF`, the overlay was not applied correctly.

## Re-run the Probes

Once the package cache shows `ONNX_DISABLE_STATIC_REGISTRATION=ON`, re-run the
diagnostic probes before changing anything else:

```bash
cmake --build build/dev --target ort_session_probe onnx_session_probe -j1

build/dev/ort_session_probe \
  tests/assets/models/yolov8n.onnx \
  tests/assets/models/yolov8n-cls.onnx

build/dev/onnx_session_probe \
  tests/assets/models/yolov8n.onnx \
  tests/assets/models/yolov8n-cls.onnx
```

The goal is to verify that model loading succeeds at the package/runtime layer
before spending time on higher-level pipeline paths.

## Validation Result

With the overlay-installed package set:

- `ONNX_DISABLE_STATIC_REGISTRATION:BOOL=ON` appears in the ONNX build cache
- the direct `Ort::Session` probe succeeds for
  `tests/assets/models/yolov8n.onnx`
- the direct `Ort::Session` probe succeeds for
  `tests/assets/models/yolov8n-cls.onnx`
- the repo-level `OnnxSession::create(...)` probe succeeds for both models
- the previously skipped model-backed integration tests run and pass

That is why the overlay is treated as the current vcpkg development baseline
for this repo.
