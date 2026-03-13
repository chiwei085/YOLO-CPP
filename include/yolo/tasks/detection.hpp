#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "yolo/core/error.hpp"
#include "yolo/core/image.hpp"
#include "yolo/core/model_spec.hpp"
#include "yolo/core/result.hpp"
#include "yolo/core/session_options.hpp"
#include "yolo/core/types.hpp"

namespace yolo
{

struct Detection
{
    RectF bbox{};
    float score{0.0F};
    ClassId class_id{0};
    std::optional<std::string> label{};
};

struct DetectionOptions
{
    float confidence_threshold{0.25F};
    float nms_iou_threshold{0.45F};
    std::size_t max_detections{300};
    bool class_agnostic_nms{false};
};

struct DetectionResult
{
    std::vector<Detection> detections{};
    InferenceMetadata metadata{};
    Error error{};

    [[nodiscard]] constexpr bool ok() const noexcept { return error.ok(); }
};

class Detector
{
public:
    virtual ~Detector() = default;
    // RAII task handle with a stable task-specific interface.
    [[nodiscard]] virtual const ModelSpec& model() const noexcept = 0;
    virtual DetectionResult run(const ImageView& image) const = 0;
};

[[nodiscard]] std::unique_ptr<Detector> create_detector(
    ModelSpec spec, SessionOptions session = {}, DetectionOptions options = {});

}  // namespace yolo
