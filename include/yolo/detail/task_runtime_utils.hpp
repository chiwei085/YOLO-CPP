#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "yolo/core/result.hpp"
#include "yolo/core/session_options.hpp"
#include "yolo/core/tensor.hpp"
#include "yolo/detail/engine.hpp"
#include "yolo/detail/image_preprocess.hpp"

namespace yolo::detail
{

[[nodiscard]] std::string provider_name(ExecutionProvider provider);
[[nodiscard]] std::string provider_name_from_options(
    const SessionOptions& options);

[[nodiscard]] bool is_image_like_input(const TensorInfo& input) noexcept;

[[nodiscard]] Result<TensorInfo> select_primary_input(
    const std::vector<TensorInfo>& inputs, std::string_view component);
[[nodiscard]] Result<TensorInfo> select_primary_input(
    const RuntimeEngine& engine, std::string_view component);

[[nodiscard]] Result<TensorInfo> validate_primary_input(
    const TensorInfo& input, std::string_view component,
    std::optional<std::string_view> expected_name = std::nullopt);

[[nodiscard]] Result<std::int64_t> require_positive_dimension(
    const TensorInfo& info, std::size_t index, std::string_view component,
    std::string_view dimension_label);

[[nodiscard]] Result<std::int64_t> require_count(
    std::optional<std::size_t> count, std::string_view component,
    std::string_view count_label);

[[nodiscard]] InferenceMetadata make_common_metadata(
    TaskKind task, const ModelSpec& model, const PreprocessRecord& preprocessed,
    const SessionOptions& session, std::vector<TensorInfo> outputs,
    std::optional<ClassificationScoreSemantics> score_semantics = std::nullopt,
    std::optional<ClassificationScoreSemantics> source_score_semantics =
        std::nullopt);

[[nodiscard]] InferenceMetadata make_task_metadata(
    TaskKind task, const RuntimeEngine& engine,
    const PreprocessedImage& preprocessed, const SessionOptions& session,
    std::optional<ClassificationScoreSemantics> score_semantics = std::nullopt,
    std::optional<ClassificationScoreSemantics> source_score_semantics =
        std::nullopt);

[[nodiscard]] InferenceMetadata make_raw_metadata(
    const ModelSpec& model, const PreprocessedImage& preprocessed,
    const SessionOptions& session, const RawOutputTensors& outputs);

}  // namespace yolo::detail
