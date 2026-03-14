# Tools

This directory contains developer-facing executables that support bring-up,
parity convergence, and environment diagnosis. They are intentionally kept
outside the main library so task-specific debug logic does not leak into the
runtime API.

Current layout:

- `parity/`: release-worthy parity dump tools used by `tests/parity`
- `debug/`: release-worthy staged debug dump tools used for first-fail bisect
- `dev/`: developer-only environment and linkage probes

Current binaries:

- `parity/yolo_cpp_parity_dump.cpp`
- `debug/yolo_cpp_segmentation_debug_dump.cpp`
- `debug/yolo_cpp_pose_debug_dump.cpp`
- `debug/yolo_cpp_obb_debug_dump.cpp`
- `dev/onnx_session_probe.cpp`
- `dev/ort_session_probe.cpp`
- `dev/ort_link_probe/`

The `parity/` and `debug/` tools are part of the repo's supported validation
workflow. The `dev/` probes are internal diagnostics and may evolve more
freely.

The source tree is organized by role, but the built executables still land in
the top-level build directory, for example `build/dev/yolo_cpp_parity_dump`, so
existing parity runners and docs do not need different invocation paths.
