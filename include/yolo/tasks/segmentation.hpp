#pragma once

#include <cstdint>
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

struct SegmentationMask
{
    Size2i size{};
    std::vector<std::uint8_t> data{};
};

struct SegmentationInstance
{
    RectF bbox{};
    float score{0.0F};
    ClassId class_id{0};
    std::optional<std::string> label{};
    SegmentationMask mask{};
};

struct SegmentationOptions
{
    float confidence_threshold{0.25F};
    float nms_iou_threshold{0.45F};
    std::size_t max_detections{300};
};

struct SegmentationResult
{
    std::vector<SegmentationInstance> instances{};
    InferenceMetadata metadata{};
    Error error{};

    [[nodiscard]] constexpr bool ok() const noexcept { return error.ok(); }
};

class Segmenter
{
public:
    virtual ~Segmenter() = default;
    // RAII task handle with a stable task-specific interface.
    [[nodiscard]] virtual const ModelSpec& model() const noexcept = 0;
    virtual SegmentationResult run(const ImageView& image) const = 0;
};

[[nodiscard]] std::unique_ptr<Segmenter> create_segmenter(
    ModelSpec spec, SessionOptions session = {},
    SegmentationOptions options = {});

}  // namespace yolo
