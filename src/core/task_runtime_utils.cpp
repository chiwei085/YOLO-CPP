#include "yolo/detail/task_runtime_utils.hpp"

#include <utility>

namespace yolo::detail
{

std::string provider_name(ExecutionProvider provider) {
    switch (provider) {
        case ExecutionProvider::cpu:
            return "cpu";
        case ExecutionProvider::cuda:
            return "cuda";
        case ExecutionProvider::tensorrt:
            return "tensorrt";
    }

    return "unknown";
}

std::string provider_name_from_options(const SessionOptions& options) {
    if (options.providers.empty()) {
        return "cpu";
    }

    return provider_name(options.providers.front().provider);
}

bool is_image_like_input(const TensorInfo& input) noexcept {
    if (input.shape.rank() != 4) {
        return false;
    }

    const auto& dims = input.shape.dims;
    if (dims.size() != 4) {
        return false;
    }

    const auto is_channel_dim = [](const TensorDimension& dim) {
        return dim.value.has_value() &&
               (*dim.value == 1 || *dim.value == 3 || *dim.value == 4);
    };

    const bool nchw = is_channel_dim(dims[1]);
    const bool nhwc = is_channel_dim(dims[3]);
    return nchw || nhwc;
}

Result<TensorInfo> select_primary_input(const std::vector<TensorInfo>& inputs,
                                        std::string_view component) {
    if (inputs.empty()) {
        return {.error = make_error(
                    ErrorCode::invalid_state,
                    "Runtime engine has no input tensor metadata.",
                    ErrorContext{.component = std::string{component}})};
    }

    if (inputs.size() == 1) {
        return {.value = inputs.front(), .error = {}};
    }

    for (const TensorInfo& input : inputs) {
        if (is_image_like_input(input)) {
            return {.value = input, .error = {}};
        }
    }

    return {.error = make_error(
                ErrorCode::unsupported_model,
                "No image-like primary input tensor could be selected.",
                ErrorContext{
                    .component = std::string{component},
                    .expected = std::string{"at least one rank-4 image input"},
                    .actual = std::to_string(inputs.size()) + " candidate inputs",
                })};
}

Result<TensorInfo> select_primary_input(const RuntimeEngine& engine,
                                        std::string_view component) {
    return select_primary_input(engine.description().inputs, component);
}

Result<TensorInfo> validate_primary_input(
    const TensorInfo& input, std::string_view component,
    std::optional<std::string_view> expected_name) {
    if (expected_name.has_value() && input.name != *expected_name) {
        return {.error = make_error(
                    ErrorCode::invalid_argument,
                    "Primary input tensor name does not match the expected binding.",
                    ErrorContext{
                        .component = std::string{component},
                        .input_name = input.name,
                        .expected = std::string{*expected_name},
                        .actual = input.name,
                    })};
    }

    if (input.data_type != TensorDataType::float32) {
        return {.error = make_error(
                    ErrorCode::type_mismatch,
                    "Primary input tensor must use float32 data.",
                    ErrorContext{
                        .component = std::string{component},
                        .input_name = input.name,
                        .expected = std::string{"float32"},
                        .actual = format_data_type(input.data_type),
                    })};
    }

    if (input.shape.rank() != 4) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "Primary input tensor must be rank-4.",
                    ErrorContext{
                        .component = std::string{component},
                        .input_name = input.name,
                        .expected = std::string{"rank 4"},
                        .actual = std::to_string(input.shape.rank()),
                    })};
    }

    if (!is_image_like_input(input)) {
        return {.error = make_error(
                    ErrorCode::unsupported_model,
                    "Primary input tensor is not image-like.",
                    ErrorContext{
                        .component = std::string{component},
                        .input_name = input.name,
                        .expected = std::string{"NCHW or NHWC image tensor"},
                    })};
    }

    return {.value = input, .error = {}};
}

Result<std::int64_t> require_positive_dimension(const TensorInfo& info,
                                                std::size_t index,
                                                std::string_view component,
                                                std::string_view dimension_label) {
    if (index >= info.shape.dims.size()) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "Tensor rank is smaller than the requested dimension.",
                    ErrorContext{
                        .component = std::string{component},
                        .input_name = info.name,
                        .expected = std::string{dimension_label},
                        .actual = std::to_string(info.shape.rank()),
                    })};
    }

    const TensorDimension& dimension = info.shape.dims[index];
    if (!dimension.value.has_value()) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "Tensor dimension must be static.",
                    ErrorContext{
                        .component = std::string{component},
                        .input_name = info.name,
                        .expected = std::string{"static "} +
                                        std::string{dimension_label},
                        .actual = std::string{"dynamic"},
                    })};
    }

    if (*dimension.value <= 0) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "Tensor dimension must be positive.",
                    ErrorContext{
                        .component = std::string{component},
                        .input_name = info.name,
                        .expected = std::string{"positive "} +
                                        std::string{dimension_label},
                        .actual = std::to_string(*dimension.value),
                    })};
    }

    return {.value = *dimension.value, .error = {}};
}

Result<std::int64_t> require_count(std::optional<std::size_t> count,
                                   std::string_view component,
                                   std::string_view count_label) {
    if (!count.has_value()) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "Count must be known and non-dynamic.",
                    ErrorContext{
                        .component = std::string{component},
                        .expected = std::string{"static "} +
                                        std::string{count_label},
                        .actual = std::string{"dynamic"},
                    })};
    }

    if (*count == 0U) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "Count must be greater than zero.",
                    ErrorContext{
                        .component = std::string{component},
                        .expected = std::string{"positive "} +
                                        std::string{count_label},
                        .actual = std::string{"0"},
                    })};
    }

    return {.value = static_cast<std::int64_t>(*count), .error = {}};
}

InferenceMetadata make_common_metadata(
    TaskKind task, const ModelSpec& model, const PreprocessRecord& preprocessed,
    const SessionOptions& session, std::vector<TensorInfo> outputs,
    std::optional<ClassificationScoreSemantics> score_semantics,
    std::optional<ClassificationScoreSemantics> source_score_semantics) {
    return InferenceMetadata{
        .task = task,
        .model_name = model.model_name,
        .adapter_name = model.adapter,
        .provider_name = provider_name_from_options(session),
        .original_image_size = preprocessed.source_size,
        .preprocess = preprocessed,
        .outputs = std::move(outputs),
        .classification_score_semantics = score_semantics,
        .source_classification_score_semantics = source_score_semantics,
        .latency_ms = std::nullopt,
    };
}

InferenceMetadata make_task_metadata(
    TaskKind task, const RuntimeEngine& engine,
    const PreprocessedImage& preprocessed, const SessionOptions& session,
    std::optional<ClassificationScoreSemantics> score_semantics,
    std::optional<ClassificationScoreSemantics> source_score_semantics) {
    return make_common_metadata(task, engine.model(), preprocessed.record,
                                session, engine.description().outputs,
                                score_semantics, source_score_semantics);
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

    return make_common_metadata(model.task, model, preprocessed.record, session,
                                std::move(output_infos));
}

}  // namespace yolo::detail
