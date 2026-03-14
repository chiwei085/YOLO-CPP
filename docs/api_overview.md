# API Overview

This document summarizes the public C++ API exposed by YOLO-CPP.

## Header Layout

Two common include styles are available:

- `#include "yolo/yolos.hpp"`
  Umbrella header that includes the public API surface.
- `#include "yolo/facade.hpp"`
  Focused include for the main pipeline interface.

## Core Value Types

Declared in `yolo/core/types.hpp`.

Enums:

- `yolo::TaskKind`
  Task selector for `detect`, `classify`, `seg`, `pose`, and `obb`.
- `yolo::PixelFormat`
  Supported image formats: `bgr8`, `rgb8`, `gray8`, `bgra8`, `rgba8`.
- `yolo::TensorDataType`
  Tensor element type metadata used by public tensor descriptors.

Geometry and shared structs:

- `yolo::Size2i`
- `yolo::Padding2i`
- `yolo::Scale2f`
- `yolo::Size2f`
- `yolo::Point2f`
- `yolo::RectI`
- `yolo::RectF`
- `yolo::Keypoint`

Aliases:

- `yolo::ClassId`
- `yolo::Shape`

## Error And Result Model

Declared in `yolo/core/error.hpp`.

Enums and structs:

- `yolo::ErrorCode`
- `yolo::ErrorContext`
- `yolo::Error`
- `yolo::Result<T>`

Exception support:

- `yolo::YoloException`
- `yolo::throw_if_error(const Error&)`

Utility:

- `yolo::make_error(...)`

The public API is primarily status-style. Most creation and inference entry
points return a result object that carries either a payload or a structured
`yolo::Error`.

## Image And Preprocess Types

Declared in `yolo/core/image.hpp`.

Enums:

- `yolo::ResizeMode`
- `yolo::ColorConversion`
- `yolo::TensorLayout`

Structs:

- `yolo::NormalizeSpec`
- `yolo::PreprocessPolicy`
- `yolo::PreprocessRecord`
- `yolo::ImageView`

Factory helpers:

- `yolo::make_detection_preprocess_policy(Size2i target_size)`
- `yolo::make_classification_preprocess_policy(Size2i target_size)`

`yolo::ImageView` is the main image input type used by the public task and
pipeline APIs. It describes:

- a byte span
- image size
- row stride
- pixel format

## Model And Session Configuration

Declared in `yolo/core/model_spec.hpp` and `yolo/core/session_options.hpp`.

`yolo::ModelSpec` fields:

- `path`
- `task`
- `model_name`
- `adapter`
- `input_size`
- `class_count`
- `labels`
- `valid()`

`yolo::SessionOptions` fields:

- `intra_op_threads`
- `inter_op_threads`
- `enable_profiling`
- `enable_memory_pattern`
- `enable_fp16`
- `graph_optimization`
- `providers`

Related enums and structs:

- `yolo::ExecutionProvider`
- `yolo::GraphOptimizationLevel`
- `yolo::ExecutionProviderOptions`

## Tensor Metadata Types

Declared in `yolo/core/tensor.hpp`.

Structs:

- `yolo::TensorDimension`
- `yolo::TensorShape`
- `yolo::TensorInfo`
- `yolo::TensorView<T>`

Notable member functions:

- `TensorDimension::dynamic()`
- `TensorDimension::fixed(...)`
- `TensorDimension::is_dynamic()`
- `TensorShape::empty()`
- `TensorShape::rank()`
- `TensorShape::is_dynamic()`
- `TensorShape::element_count()`
- `TensorView<T>::empty()`

These types are used for public metadata and raw output inspection.

## Shared Inference Metadata

Declared in `yolo/core/result.hpp`.

Enums:

- `yolo::ClassificationScoreSemantics`

Structs:

- `yolo::InferenceMetadata`

`InferenceMetadata` carries task-independent output information such as:

- task kind
- optional model and adapter names
- optional provider name
- optional original image size
- optional preprocess record
- output tensor descriptors
- classification score semantics when relevant
- optional latency

## Task-Specific APIs

### Detection

Declared in `yolo/tasks/detection.hpp`.

Structs:

- `yolo::Detection`
- `yolo::DetectionOptions`
- `yolo::DetectionResult`

Interface and factory:

- `yolo::Detector`
- `yolo::create_detector(ModelSpec, SessionOptions, DetectionOptions)`

`DetectionOptions` fields:

- `confidence_threshold`
- `nms_iou_threshold`
- `max_detections`
- `class_agnostic_nms`

`DetectionResult` fields:

- `detections`
- `metadata`
- `error`
- `ok()`

### Classification

Declared in `yolo/tasks/classification.hpp`.

Structs:

- `yolo::Classification`
- `yolo::ClassificationOptions`
- `yolo::ClassificationResult`

Interface and factory:

- `yolo::Classifier`
- `yolo::create_classifier(ModelSpec, SessionOptions, ClassificationOptions)`

`ClassificationOptions` fields:

- `top_k`

`ClassificationResult` fields:

- `classes`
- `scores`
- `metadata`
- `error`
- `ok()`

Public classification scores are normalized probabilities.

### Segmentation

Declared in `yolo/tasks/segmentation.hpp`.

Structs:

- `yolo::SegmentationMask`
- `yolo::SegmentationInstance`
- `yolo::SegmentationOptions`
- `yolo::SegmentationResult`

Interface and factory:

- `yolo::Segmenter`
- `yolo::create_segmenter(ModelSpec, SessionOptions, SegmentationOptions)`

`SegmentationResult` fields:

- `instances`
- `metadata`
- `error`
- `ok()`

### Pose

Declared in `yolo/tasks/pose.hpp`.

Structs:

- `yolo::PoseKeypoint`
- `yolo::PoseDetection`
- `yolo::PoseOptions`
- `yolo::PoseResult`

Interface and factory:

- `yolo::PoseEstimator`
- `yolo::create_pose_estimator(ModelSpec, SessionOptions, PoseOptions)`

`PoseResult` fields:

- `poses`
- `metadata`
- `error`
- `ok()`

### Oriented Bounding Boxes

Declared in `yolo/tasks/obb.hpp`.

Constants:

- `yolo::kObbPi`
- `yolo::kObbHalfPi`

Structs and functions:

- `yolo::OrientedBox`
- `yolo::canonicalize_oriented_box(OrientedBox)`
- `yolo::OrientedDetection`
- `yolo::ObbOptions`
- `yolo::ObbResult`

Interface and factory:

- `yolo::OrientedDetector`
- `yolo::create_obb_detector(ModelSpec, SessionOptions, ObbOptions)`

Notable `OrientedBox` member functions:

- `angle_degrees()`
- `corners()`

`ObbResult` fields:

- `boxes`
- `metadata`
- `error`
- `ok()`

## Pipeline Facade

Declared in `yolo/facade.hpp`.

Structs:

- `yolo::PipelineOptions`
- `yolo::RawOutputTensor`
- `yolo::RawInferenceResult`
- `yolo::PipelineInfo`

Aliases:

- `yolo::InferenceResult`

Class and factories:

- `yolo::Pipeline`
- `yolo::Pipeline::create(...)`
- `yolo::create_pipeline(...)`

`PipelineOptions` groups per-task options:

- `detection`
- `classification`
- `segmentation`
- `pose`
- `obb`

`RawOutputTensor` fields and helpers:

- `info`
- `storage`
- `bytes()`

`RawInferenceResult` fields:

- `outputs`
- `metadata`
- `error`
- `ok()`

`PipelineInfo` fields:

- `model`
- `inputs`
- `outputs`
- `preprocess`
- `adapter_binding`

`Pipeline` member functions:

- `info()`
- `run(const ImageView&)`
- `run_raw(const ImageView&)`
- `detect(const ImageView&)`
- `classify(const ImageView&)`
- `segment(const ImageView&)`
- `estimate_pose(const ImageView&)`
- `detect_obb(const ImageView&)`

Use the pipeline facade when you want:

- one object that owns the model session
- one uniform entry point for multiple task methods
- public access to model and preprocess metadata through `PipelineInfo`
- raw tensor access through `run_raw(...)`

## Ultralytics Adapter API

Declared in `yolo/adapters/ultralytics.hpp`.

This namespace is the advanced public API for probing Ultralytics-style model
metadata and output contracts.

Namespace:

- `yolo::adapters::ultralytics`

Constants:

- `kAdapterName`

Enums:

- `OutputRole`
- `PoseKeypointSemantic`
- `ObbBoxEncoding`
- `DetectionHeadLayout`
- `ClassificationScoreKind`

Structs:

- `OutputBinding`
- `DetectionBindingSpec`
- `ClassificationBindingSpec`
- `SegmentationBindingSpec`
- `PoseBindingSpec`
- `ObbBindingSpec`
- `AdapterBindingSpec`

Probe functions from explicit tensor metadata:

- `probe_detection(...)`
- `probe_classification(...)`
- `probe_segmentation(...)`
- `probe_pose(...)`
- `probe_obb(...)`

Probe functions from a model path and session options:

- `probe_detection_model(...)`
- `probe_classification_model(...)`
- `probe_segmentation_model(...)`
- `probe_pose_model(...)`
- `probe_obb_model(...)`

`AdapterBindingSpec` includes:

- resolved adapter name
- model spec
- preprocess policy
- output bindings
- optional task-specific binding metadata
- `task()`

For most application code, the high-level pipeline and task APIs are the main
surface. The Ultralytics adapter API is useful when you need explicit model
introspection or adapter-derived binding information.

## Public API Levels

The public surface can be read in three layers:

- foundational types
  `core/types.hpp`, `core/error.hpp`, `core/image.hpp`,
  `core/model_spec.hpp`, `core/session_options.hpp`, `core/tensor.hpp`,
  `core/result.hpp`
- task and pipeline interfaces
  `tasks/*.hpp`, `facade.hpp`
- advanced model introspection
  `adapters/ultralytics.hpp`

Anything under `include/yolo/detail/` should be treated as non-public
implementation detail, even though the headers are visible in the source tree.

## Related Documents

- [Get Started](get_started.md)
- [Build Notes](build_notes.md)
- [../README.md](../README.md)
