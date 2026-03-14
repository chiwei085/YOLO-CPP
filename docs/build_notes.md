# Build Notes

This document describes the current ONNX Runtime build baseline for YOLO-CPP.
It focuses on the supported package shape and the parts users need to know when
configuring the project.

## CPU Baseline

The main CPU development path uses:

- shared ONNX Runtime via `x64-linux-dynamic`
- the repo's local ONNX overlay port in [`vcpkg-overlay-ports/onnx`](../vcpkg-overlay-ports/onnx)
- `YOLO_CPP_ORT_PROVIDER=cpu`

In practice, the normal CPU route is:

```bash
cmake --preset dev --fresh
cmake --build build/dev
```

## Why Shared ORT Is The Baseline

This repo previously hit duplicate ONNX schema-registration failures when using
static package shapes. Shared ONNX Runtime is the recommended baseline because
it avoids the bad final-link shape that pulled standalone ONNX archives into
the executable.

That makes the main CPU path more stable for:

- local development
- model-backed integration tests
- parity and debug tooling that constructs `Ort::Session`

## Why The Repo Uses An ONNX Overlay Port

Shared ORT was necessary, but it was not sufficient by itself.

The stock vcpkg `onnxruntime` package expects the `onnx` port to be built with:

```text
ONNX_DISABLE_STATIC_REGISTRATION=ON
```

This repo therefore carries a local overlay port that applies that setting for
the `onnx` package. Without it, model loading can still fail with duplicate
schema-registration errors during `Ort::Session` creation.

The overlay is part of the validated dependency baseline, not an optional extra
patch for debugging.

## What Users Need To Know

- Use the provided presets instead of hand-assembling dependency paths.
- Keep `VCPKG_ROOT` pointed at your local vcpkg checkout.
- Treat `dev` as the main CPU baseline.
- Treat `dev-opencv` as an optional local visualization route layered on top of
  the same CPU dependency policy.

If you configure with the repo presets, the required vcpkg environment and
overlay-port settings are already wired in.

## If Model Loading Starts Failing Again

If you see duplicate schema-registration failures while creating an ONNX
Runtime session:

1. Confirm you configured with a repo preset.
2. Confirm the build is using `x64-linux-dynamic`.
3. Reconfigure from a fresh build tree so the overlay-backed package set is
   resolved again.

The first thing to inspect is dependency and package shape, not inference code.
