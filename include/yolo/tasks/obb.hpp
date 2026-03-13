#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "yolo/core/error.hpp"
#include "yolo/core/image.hpp"
#include "yolo/core/model_spec.hpp"
#include "yolo/core/result.hpp"
#include "yolo/core/session_options.hpp"
#include "yolo/core/types.hpp"

namespace yolo
{

struct OrientedBox
{
    Point2f center{};
    Size2f size{};
    float angle_degrees{0.0F};
    float score{0.0F};
    ClassId class_id{0};
    std::optional<std::string> label{};
};

struct ObbOptions
{
    float confidence_threshold{0.25F};
    float nms_iou_threshold{0.45F};
    std::size_t max_detections{300};
};

struct ObbResult
{
    std::vector<OrientedBox> boxes{};
    InferenceMetadata metadata{};
    Error error{};

    [[nodiscard]] constexpr bool ok() const noexcept { return error.ok(); }
};

class OrientedDetector
{
public:
    virtual ~OrientedDetector() = default;
    // RAII task handle with a stable task-specific interface.
    [[nodiscard]] virtual const ModelSpec& model() const noexcept = 0;
    virtual ObbResult run(const ImageView& image) const = 0;
};

[[nodiscard]] std::unique_ptr<OrientedDetector> create_obb_detector(
    ModelSpec spec, SessionOptions session = {}, ObbOptions options = {});

}  // namespace yolo
