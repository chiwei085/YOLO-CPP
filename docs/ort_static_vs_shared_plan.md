# ORT Static vs Shared: Diagnosis Record

## Summary

YOLO-CPP now uses a shared ONNX Runtime package for its main CPU development
path.

This document is now mainly a diagnosis record. The current working baseline is
documented in
[`docs/vcpkg_overlay_onnx_fix.md`](vcpkg_overlay_onnx_fix.md):

- shared ORT via `x64-linux-dynamic`
- local ONNX overlay port with `ONNX_DISABLE_STATIC_REGISTRATION=ON`

The shared triplet was a necessary step, but not the full fix on its own.

## Confirmed Diagnosis

The static package path was narrowed down with four consistent findings:

- `onnxruntime::onnxruntime` from the static install was exported as
  `INTERFACE IMPORTED`.
- Its `INTERFACE_LINK_LIBRARIES` included `ONNX::onnx` and
  `ONNX::onnx_proto`.
- A minimal probe in
  [`tools/dev/ort_link_probe/CMakeLists.txt`](../tools/dev/ort_link_probe/CMakeLists.txt)
  still linked `libonnx.a` and `libonnx_proto.a` when it only requested
  `onnxruntime::onnxruntime`.
- Runtime failures showed repeated duplicate-schema-registration errors such as
  `Schema error: Trying to register schema ... but it is already registered ...`.

Together, those results point to static package composition and transitive
linkage as the root cause.

## Shared Validation Result

The shared validation environment used:

- triplet: `x64-linux-dynamic`
- package: `onnxruntime:x64-linux-dynamic`
- probe: a minimal executable that only constructs `Ort::Env`

The shared package exports `onnxruntime::onnxruntime` as `SHARED IMPORTED`, and
the standalone probe link line resolved through `libonnxruntime.so` instead of
directly linking `libonnx.a` or `libonnx_proto.a`.

The probe then ran successfully with exit code `0` and no duplicate schema
registration errors.

## Intermediate Repo-Level Validation Result

After switching the main `dev` preset to `x64-linux-dynamic`, the repo:

- configured successfully with shared ORT
- built `yolo_cpp`, `yolo_cpp_tests`, and `yolo_cpp_integration_tests`
- still reproduced duplicate schema registration during model load

That intermediate result is what led to the package-level investigation that
eventually identified the missing `ONNX_DISABLE_STATIC_REGISTRATION=ON`
configure flag in the vcpkg `onnx` port.

## Repo Policy Going Forward

The main CPU development and integration-test path should use shared ORT:

- preset: `dev`
- triplet: `x64-linux-dynamic`
- provider: `cpu`

Unit, component, and adapter tests remain model-free and are unaffected by this
change. They should continue to be expanded normally.

Integration tests remain the official model-backed validation path. The
validated configuration is the shared `dev` preset plus the ONNX overlay
baseline described in
[`docs/vcpkg_overlay_onnx_fix.md`](vcpkg_overlay_onnx_fix.md).

## How To Use The Shared ORT Path

Configure from a clean build tree:

```bash
cmake --preset dev --fresh
```

Build the project and all default tests:

```bash
cmake --build build/dev
```

Run the full registered test suite:

```bash
cd build/dev
ctest --output-on-failure
```

Or run the repo-level convenience target:

```bash
cmake --build build/dev --target check
```

## Verification Commands

Confirm the active triplet in the configured build:

```bash
cmake -LA -N build/dev | rg 'VCPKG_TARGET_TRIPLET|YOLO_CPP_ORT_PROVIDER'
```

Expected configure result:

```text
VCPKG_TARGET_TRIPLET:STRING=x64-linux-dynamic
YOLO_CPP_ORT_PROVIDER:STRING=cpu
```

Inspect the shared package export:

```bash
rg -n 'add_library\\(onnxruntime::onnxruntime|IMPORTED_LOCATION|IMPORTED_LINK_DEPENDENT_LIBRARIES' \
  "$VCPKG_ROOT/installed/x64-linux-dynamic/share/onnxruntime/onnxruntimeTargets"*.cmake
```

Rebuild the standalone probe if you need to re-check linkage shape:

```bash
PROBE_BUILD_DIR=build/ort_link_probe_shared

cmake -S tools/dev/ort_link_probe -B "$PROBE_BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DVCPKG_TARGET_TRIPLET=x64-linux-dynamic \
  -DCMAKE_CXX_FLAGS="-isystem $VCPKG_ROOT/installed/x64-linux-dynamic/include"

cmake --build "$PROBE_BUILD_DIR" --verbose -j1
```

In the verbose probe link line, the expected good shape is a direct link to
`libonnxruntime.so`, without `libonnx.a` or `libonnx_proto.a`.

You can double-check runtime dependencies with:

```bash
ldd "$PROBE_BUILD_DIR/ort_link_probe"
objdump -p "$PROBE_BUILD_DIR/ort_link_probe" | rg 'NEEDED|RUNPATH|RPATH'
```

## What This Document Is For

Use this note when you need the reasoning trail for:

- why the repo switched away from the static `x64-linux` baseline
- why shared ORT alone was not enough
- why the ONNX overlay port became the final fix

## If This Regresses

If duplicate schema registration appears again in the future:

1. Confirm the configure preset is still using `x64-linux-dynamic`.
2. Check the resolved link line for `libonnx.a` or `libonnx_proto.a`.
3. Re-run the standalone probe before touching production code or test
   assertions.

The first response should be to inspect package shape and linkage, not to alter
inference behavior.
