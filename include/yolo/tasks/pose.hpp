#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "yolo/core/error.hpp"
#include "yolo/core/image.hpp"
#include "yolo/core/model_spec.hpp"
#include "yolo/core/result.hpp"
#include "yolo/core/session_options.hpp"
#include "yolo/core/types.hpp"

namespace yolo
{

struct Pose
{
    RectF bbox{};
    float score{0.0F};
    std::vector<Keypoint> keypoints{};
};

struct PoseOptions
{
    float confidence_threshold{0.25F};
    float nms_iou_threshold{0.45F};
    std::size_t max_detections{300};
};

struct PoseResult
{
    std::vector<Pose> poses{};
    InferenceMetadata metadata{};
    Error error{};

    [[nodiscard]] constexpr bool ok() const noexcept { return error.ok(); }
};

class PoseEstimator
{
public:
    virtual ~PoseEstimator() = default;
    // RAII task handle with a stable task-specific interface.
    [[nodiscard]] virtual const ModelSpec& model() const noexcept = 0;
    virtual PoseResult run(const ImageView& image) const = 0;
};

[[nodiscard]] std::unique_ptr<PoseEstimator> create_pose_estimator(
    ModelSpec spec, SessionOptions session = {}, PoseOptions options = {});

}  // namespace yolo
