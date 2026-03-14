#pragma once

#include <array>
#include <cmath>
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

constexpr float kObbPi = 3.14159265358979323846F;
constexpr float kObbHalfPi = kObbPi * 0.5F;

struct OrientedBox
{
    Point2f center{};
    Size2f size{};
    float angle_radians{0.0F};

    [[nodiscard]] float angle_degrees() const noexcept {
        return angle_radians * 180.0F / kObbPi;
    }

    [[nodiscard]] std::array<Point2f, 4> corners() const noexcept {
        const float cos_value = std::cos(angle_radians);
        const float sin_value = std::sin(angle_radians);
        const Point2f width_axis{
            .x = size.width * 0.5F * cos_value,
            .y = size.width * 0.5F * sin_value,
        };
        const Point2f height_axis{
            .x = -size.height * 0.5F * sin_value,
            .y = size.height * 0.5F * cos_value,
        };
        return {
            Point2f{.x = center.x + width_axis.x + height_axis.x,
                    .y = center.y + width_axis.y + height_axis.y},
            Point2f{.x = center.x + width_axis.x - height_axis.x,
                    .y = center.y + width_axis.y - height_axis.y},
            Point2f{.x = center.x - width_axis.x - height_axis.x,
                    .y = center.y - width_axis.y - height_axis.y},
            Point2f{.x = center.x - width_axis.x + height_axis.x,
                    .y = center.y - width_axis.y + height_axis.y},
        };
    }
};

[[nodiscard]] inline OrientedBox canonicalize_oriented_box(OrientedBox box) {
    if (box.size.width <= 0.0F || box.size.height <= 0.0F) {
        box.size.width = std::max(0.0F, box.size.width);
        box.size.height = std::max(0.0F, box.size.height);
        box.angle_radians = 0.0F;
        return box;
    }

    float angle = std::fmod(box.angle_radians, kObbPi);
    if (angle < 0.0F) {
        angle += kObbPi;
    }
    if (angle >= kObbHalfPi) {
        std::swap(box.size.width, box.size.height);
        angle -= kObbHalfPi;
    }
    if (angle >= kObbHalfPi) {
        angle = std::nextafter(kObbHalfPi, 0.0F);
    }
    box.angle_radians = std::max(0.0F, angle);
    return box;
}

struct OrientedDetection
{
    OrientedBox box{};
    float score{0.0F};
    ClassId class_id{0};
    std::optional<std::string> label{};
};

struct ObbOptions
{
    float confidence_threshold{0.25F};
    float nms_iou_threshold{0.45F};
    std::size_t max_detections{300};
    bool class_agnostic_nms{false};
};

struct ObbResult
{
    std::vector<OrientedDetection> boxes{};
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
