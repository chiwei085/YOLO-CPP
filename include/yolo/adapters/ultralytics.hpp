#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "yolo/core/error.hpp"
#include "yolo/core/image.hpp"
#include "yolo/core/model_spec.hpp"
#include "yolo/core/session_options.hpp"
#include "yolo/core/tensor.hpp"
#include "yolo/core/types.hpp"

namespace yolo::adapters::ultralytics
{

inline constexpr std::string_view kAdapterName = "ultralytics";

enum class OutputRole
{
    predictions,
    proto,
};

enum class DetectionHeadLayout
{
    xywh_class_scores_last,
    xywh_class_scores_first,
    xyxy_score_class,
};

enum class ClassificationScoreKind
{
    unknown,
    logits,
    probabilities,
};

struct OutputBinding
{
    std::size_t index{0};
    std::string name{};
    OutputRole role{OutputRole::predictions};
    TensorDataType data_type{TensorDataType::float32};
    TensorShape shape{};
};

struct DetectionBindingSpec
{
    DetectionHeadLayout layout{DetectionHeadLayout::xywh_class_scores_first};
    std::size_t proposal_count{0};
    std::size_t class_count{0};
    bool external_nms{false};
};

struct ClassificationBindingSpec
{
    std::size_t class_count{0};
    ClassificationScoreKind score_kind{ClassificationScoreKind::unknown};
};

struct SegmentationBindingSpec
{
    DetectionHeadLayout layout{DetectionHeadLayout::xywh_class_scores_first};
    std::size_t proposal_count{0};
    std::size_t class_count{0};
    std::size_t mask_channel_count{0};
    bool has_proto{true};
    bool external_nms{false};
};

struct AdapterBindingSpec
{
    std::string adapter_name{std::string{kAdapterName}};
    ModelSpec model{};
    PreprocessPolicy preprocess{};
    std::vector<OutputBinding> outputs{};
    std::optional<DetectionBindingSpec> detection{};
    std::optional<ClassificationBindingSpec> classification{};
    std::optional<SegmentationBindingSpec> segmentation{};

    [[nodiscard]] TaskKind task() const noexcept { return model.task; }
};

[[nodiscard]] Result<AdapterBindingSpec> probe_detection(
    const ModelSpec& model, const std::vector<TensorInfo>& inputs,
    const std::vector<TensorInfo>& outputs);
[[nodiscard]] Result<AdapterBindingSpec> probe_classification(
    const ModelSpec& model, const std::vector<TensorInfo>& inputs,
    const std::vector<TensorInfo>& outputs);
[[nodiscard]] Result<AdapterBindingSpec> probe_segmentation(
    const ModelSpec& model, const std::vector<TensorInfo>& inputs,
    const std::vector<TensorInfo>& outputs);

[[nodiscard]] Result<AdapterBindingSpec> probe_detection_model(
    const ModelSpec& model, SessionOptions session = {});
[[nodiscard]] Result<AdapterBindingSpec> probe_classification_model(
    const ModelSpec& model, SessionOptions session = {});
[[nodiscard]] Result<AdapterBindingSpec> probe_segmentation_model(
    const ModelSpec& model, SessionOptions session = {});

}  // namespace yolo::adapters::ultralytics
