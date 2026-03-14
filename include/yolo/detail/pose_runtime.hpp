#pragma once

#include <memory>
#include <vector>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/detection_runtime.hpp"
#include "yolo/detail/tensor_utils.hpp"
#include "yolo/tasks/pose.hpp"

namespace yolo::detail
{

class RuntimeEngine;

struct PoseCandidate
{
    RectF bbox{};
    float score{0.0F};
    ClassId class_id{0};
    std::vector<PoseKeypoint> keypoints{};
};

struct PoseDecodeSpec
{
    std::size_t output_index{0};
    DetectionLayout layout{DetectionLayout::xywh_class_scores_last};
    std::size_t proposal_count{0};
    std::size_t class_count{0};
    std::size_t keypoint_count{0};
    std::size_t keypoint_dimension{3};
    adapters::ultralytics::PoseKeypointSemantic keypoint_semantic{
        adapters::ultralytics::PoseKeypointSemantic::xyscore};
};

[[nodiscard]] Result<PoseDecodeSpec> pose_decode_spec_from_binding(
    const adapters::ultralytics::AdapterBindingSpec& binding);

[[nodiscard]] Result<std::vector<PoseCandidate>> decode_poses(
    const RawOutputTensors& outputs, const PoseDecodeSpec& spec);

[[nodiscard]] std::vector<PoseDetection> postprocess_poses(
    std::vector<PoseCandidate> candidates, const PreprocessRecord& record,
    const PoseOptions& options, const ModelSpec& spec);

[[nodiscard]] std::unique_ptr<PoseEstimator> create_pose_estimator_with_engine(
    adapters::ultralytics::AdapterBindingSpec binding, SessionOptions session,
    PoseOptions options, std::shared_ptr<RuntimeEngine> engine);

}  // namespace yolo::detail
