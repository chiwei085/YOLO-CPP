#pragma once

#include <memory>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/tasks/classification.hpp"
#include "yolo/tasks/detection.hpp"
#include "yolo/tasks/obb.hpp"
#include "yolo/tasks/pose.hpp"
#include "yolo/tasks/segmentation.hpp"

namespace yolo::detail
{

class RuntimeEngine;

[[nodiscard]] std::unique_ptr<Detector> create_detector_with_engine(
    adapters::ultralytics::AdapterBindingSpec binding, SessionOptions session,
    DetectionOptions options, std::shared_ptr<RuntimeEngine> engine);

[[nodiscard]] std::unique_ptr<Classifier> create_classifier_with_engine(
    adapters::ultralytics::AdapterBindingSpec binding, SessionOptions session,
    ClassificationOptions options, std::shared_ptr<RuntimeEngine> engine);

[[nodiscard]] std::unique_ptr<Segmenter> create_segmenter_with_engine(
    adapters::ultralytics::AdapterBindingSpec binding, SessionOptions session,
    SegmentationOptions options, std::shared_ptr<RuntimeEngine> engine);

[[nodiscard]] std::unique_ptr<PoseEstimator> create_pose_estimator_with_engine(
    adapters::ultralytics::AdapterBindingSpec binding, SessionOptions session,
    PoseOptions options, std::shared_ptr<RuntimeEngine> engine);

[[nodiscard]] std::unique_ptr<OrientedDetector> create_obb_detector_with_engine(
    adapters::ultralytics::AdapterBindingSpec binding, SessionOptions session,
    ObbOptions options, std::shared_ptr<RuntimeEngine> engine);

}  // namespace yolo::detail
