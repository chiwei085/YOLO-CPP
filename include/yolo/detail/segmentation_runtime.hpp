#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/detection_runtime.hpp"
#include "yolo/detail/tensor_utils.hpp"
#include "yolo/tasks/segmentation.hpp"

namespace yolo::detail
{

class RuntimeEngine;

struct SegmentationCandidate
{
    RectF bbox{};
    float score{0.0F};
    ClassId class_id{0};
    std::vector<float> mask_coefficients{};
};

struct ProtoMaskTensor
{
    Size2i size{};
    std::size_t channel_count{0};
    std::vector<float> values{};
};

struct SegmentationDecodeSpec
{
    std::size_t output_index{0};
    std::size_t proto_output_index{0};
    DetectionLayout layout{DetectionLayout::xywh_class_scores_last};
    std::size_t proposal_count{0};
    std::size_t class_count{0};
    std::size_t mask_channel_count{0};
    Size2i proto_size{};
};

struct DecodedSegmentation
{
    std::vector<SegmentationCandidate> candidates{};
    ProtoMaskTensor proto{};
};

[[nodiscard]] Result<SegmentationDecodeSpec> segmentation_decode_spec_from_binding(
    const adapters::ultralytics::AdapterBindingSpec& binding);

[[nodiscard]] Result<DecodedSegmentation> decode_segmentation(
    const RawOutputTensors& outputs, const SegmentationDecodeSpec& spec);

[[nodiscard]] SegmentationMask project_segmentation_mask(
    const SegmentationCandidate& candidate, const ProtoMaskTensor& proto,
    const PreprocessRecord& record, float threshold = 0.5F);

[[nodiscard]] std::vector<SegmentationInstance> postprocess_segmentation(
    std::vector<SegmentationCandidate> candidates, const ProtoMaskTensor& proto,
    const PreprocessRecord& record, const SegmentationOptions& options,
    const ModelSpec& spec);

[[nodiscard]] std::unique_ptr<Segmenter> create_segmenter_with_engine(
    adapters::ultralytics::AdapterBindingSpec binding, SessionOptions session,
    SegmentationOptions options, std::shared_ptr<RuntimeEngine> engine);

}  // namespace yolo::detail
