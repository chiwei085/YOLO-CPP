#pragma once

#include <memory>
#include <vector>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/detection_runtime.hpp"
#include "yolo/detail/tensor_utils.hpp"
#include "yolo/tasks/obb.hpp"

namespace yolo::detail
{

class RuntimeEngine;

struct ObbCandidate
{
    OrientedBox box{};
    float score{0.0F};
    ClassId class_id{0};
};

struct ObbDecodeSpec
{
    std::size_t output_index{0};
    DetectionLayout layout{DetectionLayout::xywh_class_scores_last};
    std::size_t proposal_count{0};
    std::size_t class_count{0};
    std::size_t box_coordinate_count{4};
    std::size_t class_channel_offset{4};
    std::size_t angle_channel_offset{4};
    bool angle_is_radians{true};
};

[[nodiscard]] Result<ObbDecodeSpec> obb_decode_spec_from_binding(
    const adapters::ultralytics::AdapterBindingSpec& binding);

[[nodiscard]] Result<std::vector<ObbCandidate>> decode_obb_candidates(
    const RawOutputTensors& outputs, const ObbDecodeSpec& spec);

[[nodiscard]] std::vector<OrientedDetection> postprocess_obb(
    std::vector<ObbCandidate> candidates, const PreprocessRecord& record,
    const ObbOptions& options, const ModelSpec& spec);

[[nodiscard]] std::unique_ptr<OrientedDetector> create_obb_detector_with_engine(
    adapters::ultralytics::AdapterBindingSpec binding, SessionOptions session,
    ObbOptions options, std::shared_ptr<RuntimeEngine> engine);

}  // namespace yolo::detail
