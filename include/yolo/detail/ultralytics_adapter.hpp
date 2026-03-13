#pragma once

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/core/error.hpp"
#include "yolo/core/image.hpp"
#include "yolo/core/model_spec.hpp"
#include "yolo/core/tensor.hpp"
#include "yolo/detail/onnx_session.hpp"
#include "yolo/detail/tensor_utils.hpp"

namespace yolo::detail
{

[[nodiscard]] inline Error make_adapter_error(
    std::string_view component, ErrorCode code, std::string_view message,
    std::optional<ErrorContext> context = std::nullopt) {
    ErrorContext resolved = context.value_or(ErrorContext{});
    if (!resolved.component.has_value()) {
        resolved.component = std::string{component};
    }

    return make_error(code, message, std::move(resolved));
}

[[nodiscard]] inline bool is_channel_dimension(const TensorDimension& dim) {
    return dim.value.has_value() &&
           (*dim.value == 1 || *dim.value == 3 || *dim.value == 4);
}

[[nodiscard]] inline std::optional<Size2i> infer_image_input_size(
    const TensorInfo& input) {
    if (input.shape.rank() != 4) {
        return std::nullopt;
    }

    const auto& dims = input.shape.dims;
    if (is_channel_dimension(dims[1]) && dims[2].value.has_value() &&
        dims[3].value.has_value()) {
        return Size2i{static_cast<int>(*dims[3].value),
                      static_cast<int>(*dims[2].value)};
    }

    if (dims[1].value.has_value() && dims[2].value.has_value() &&
        is_channel_dimension(dims[3])) {
        return Size2i{static_cast<int>(*dims[2].value),
                      static_cast<int>(*dims[1].value)};
    }

    return std::nullopt;
}

[[nodiscard]] inline std::optional<std::size_t> infer_input_channels(
    const TensorInfo& input) {
    if (input.shape.rank() != 4) {
        return std::nullopt;
    }

    const auto& dims = input.shape.dims;
    if (is_channel_dimension(dims[1])) {
        return static_cast<std::size_t>(*dims[1].value);
    }

    if (is_channel_dimension(dims[3])) {
        return static_cast<std::size_t>(*dims[3].value);
    }

    return std::nullopt;
}

[[nodiscard]] inline bool expects_rgb_input(const TensorInfo& input) {
    const std::optional<std::size_t> channels = infer_input_channels(input);
    return channels.has_value() && *channels >= 3;
}

[[nodiscard]] inline std::string lowercase_copy(std::string_view value) {
    std::string lowered{value};
    std::transform(
        lowered.begin(), lowered.end(), lowered.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return lowered;
}

[[nodiscard]] inline adapters::ultralytics::ClassificationScoreKind
infer_classification_score_kind(const TensorInfo& output) {
    const std::string lowered = lowercase_copy(output.name);
    if (lowered.find("prob") != std::string::npos ||
        lowered.find("softmax") != std::string::npos) {
        return adapters::ultralytics::ClassificationScoreKind::probabilities;
    }

    if (lowered.find("logit") != std::string::npos) {
        return adapters::ultralytics::ClassificationScoreKind::logits;
    }

    return adapters::ultralytics::ClassificationScoreKind::unknown;
}

[[nodiscard]] inline Result<SessionDescription> describe_model(
    const ModelSpec& model, const SessionOptions& session) {
    const auto session_result = OnnxSession::create(model, session);
    if (!session_result.ok()) {
        return {.error = session_result.error};
    }

    return {.value = (*session_result.value)->description(), .error = {}};
}

}  // namespace yolo::detail
