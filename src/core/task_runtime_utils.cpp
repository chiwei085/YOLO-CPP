#include "yolo/detail/task_runtime_utils.hpp"

namespace yolo::detail
{

std::string provider_name_from_options(const SessionOptions& options) {
    if (options.providers.empty()) {
        return "cpu";
    }

    switch (options.providers.front().provider) {
        case ExecutionProvider::cpu:
            return "cpu";
        case ExecutionProvider::cuda:
            return "cuda";
        case ExecutionProvider::tensorrt:
            return "tensorrt";
    }

    return "unknown";
}

Result<TensorInfo> select_primary_input(const RuntimeEngine& engine,
                                        std::string_view component) {
    const auto& inputs = engine.description().inputs;
    if (inputs.empty()) {
        return {.error = make_error(
                    ErrorCode::invalid_state,
                    "Runtime engine has no input tensor metadata.",
                    ErrorContext{.component = std::string{component}})};
    }

    return {.value = inputs.front(), .error = {}};
}

InferenceMetadata make_task_metadata(
    TaskKind task, const RuntimeEngine& engine,
    const PreprocessedImage& preprocessed, const SessionOptions& session,
    std::optional<ClassificationScoreSemantics> score_semantics,
    std::optional<ClassificationScoreSemantics> source_score_semantics) {
    return InferenceMetadata{
        .task = task,
        .model_name = engine.model().model_name,
        .adapter_name = engine.model().adapter,
        .provider_name = provider_name_from_options(session),
        .original_image_size = preprocessed.record.source_size,
        .preprocess = preprocessed.record,
        .outputs = engine.description().outputs,
        .classification_score_semantics = score_semantics,
        .source_classification_score_semantics = source_score_semantics,
        .latency_ms = std::nullopt,
    };
}

InferenceMetadata make_raw_metadata(const ModelSpec& model,
                                    const PreprocessedImage& preprocessed,
                                    const SessionOptions& session,
                                    const RawOutputTensors& outputs) {
    std::vector<TensorInfo> output_infos{};
    output_infos.reserve(outputs.size());
    for (const auto& output : outputs) {
        output_infos.push_back(output.info);
    }

    return InferenceMetadata{
        .task = model.task,
        .model_name = model.model_name,
        .adapter_name = model.adapter,
        .provider_name = provider_name_from_options(session),
        .original_image_size = preprocessed.record.source_size,
        .preprocess = preprocessed.record,
        .outputs = std::move(output_infos),
        .latency_ms = std::nullopt,
    };
}

}  // namespace yolo::detail
