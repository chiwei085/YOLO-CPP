#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "yolo/core/result.hpp"
#include "yolo/detail/engine.hpp"
#include "yolo/detail/image_preprocess.hpp"

namespace yolo::detail
{

[[nodiscard]] std::string provider_name_from_options(
    const SessionOptions& options);

[[nodiscard]] Result<TensorInfo> select_primary_input(
    const RuntimeEngine& engine, std::string_view component);

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
