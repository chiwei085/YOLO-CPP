#pragma once

#include <memory>

#include "yolo/tasks/classification.hpp"
#include "yolo/tasks/detection.hpp"

namespace yolo::detail
{

class RuntimeEngine;

[[nodiscard]] std::unique_ptr<Detector> create_detector_with_engine(
    ModelSpec spec, SessionOptions session, DetectionOptions options,
    std::shared_ptr<RuntimeEngine> engine);

[[nodiscard]] std::unique_ptr<Classifier> create_classifier_with_engine(
    ModelSpec spec, SessionOptions session, ClassificationOptions options,
    std::shared_ptr<RuntimeEngine> engine);

}  // namespace yolo::detail
