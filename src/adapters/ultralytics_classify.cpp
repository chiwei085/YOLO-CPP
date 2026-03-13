#include <optional>
#include <string>
#include <vector>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/tensor_utils.hpp"
#include "yolo/detail/ultralytics_adapter.hpp"

namespace yolo::adapters::ultralytics
{
namespace
{

constexpr std::string_view kComponent = "ultralytics_classify_adapter";

Result<AdapterBindingSpec> build_classification_binding(
    const ModelSpec& model, const std::vector<TensorInfo>& inputs,
    const std::vector<TensorInfo>& outputs) {
    if (inputs.size() != 1) {
        return {
            .error = detail::make_adapter_error(
                kComponent, ErrorCode::unsupported_model,
                "Ultralytics classification models must expose exactly one "
                "input.",
                ErrorContext{
                    .expected = std::string{"1 input tensor"},
                    .actual = std::to_string(inputs.size()) + " input tensors",
                })};
    }

    if (outputs.size() != 1) {
        return {.error = detail::make_adapter_error(
                    kComponent, ErrorCode::unsupported_model,
                    "Ultralytics classification models must expose exactly one "
                    "output.",
                    ErrorContext{
                        .expected = std::string{"1 output tensor"},
                        .actual =
                            std::to_string(outputs.size()) + " output tensors",
                    })};
    }

    const TensorInfo& input = inputs.front();
    const std::optional<Size2i> input_size =
        detail::infer_image_input_size(input);
    const std::optional<std::size_t> channels =
        detail::infer_input_channels(input);
    if (!input_size.has_value() || !channels.has_value() || *channels != 3U) {
        return {
            .error = detail::make_adapter_error(
                kComponent, ErrorCode::unsupported_model,
                "Ultralytics classification expects a rank-4 RGB image input.",
                ErrorContext{
                    .input_name = input.name,
                    .expected = std::string{"[N,3,H,W] or [N,H,W,3]"},
                    .actual = detail::format_shape(input.shape),
                })};
    }

    const TensorInfo& output = outputs.front();
    if (output.data_type != TensorDataType::float32) {
        return {.error = detail::make_adapter_error(
                    kComponent, ErrorCode::type_mismatch,
                    "Ultralytics classification output must be float32.",
                    ErrorContext{
                        .output_name = output.name,
                        .expected = std::string{"float32"},
                        .actual = detail::format_data_type(output.data_type),
                    })};
    }

    std::size_t class_count = 0;
    if (output.shape.rank() == 1 && output.shape.dims[0].value.has_value()) {
        class_count = static_cast<std::size_t>(*output.shape.dims[0].value);
    }
    else if (output.shape.rank() == 2 &&
             output.shape.dims[0].value.has_value() &&
             output.shape.dims[1].value.has_value() &&
             *output.shape.dims[0].value == 1) {
        class_count = static_cast<std::size_t>(*output.shape.dims[1].value);
    }
    else if (output.shape.rank() == 3 &&
             output.shape.dims[0].value.has_value() &&
             output.shape.dims[1].value.has_value() &&
             output.shape.dims[2].value.has_value() &&
             *output.shape.dims[0].value == 1 &&
             *output.shape.dims[1].value == 1) {
        class_count = static_cast<std::size_t>(*output.shape.dims[2].value);
    }
    else {
        return {.error = detail::make_adapter_error(
                    kComponent, ErrorCode::unsupported_model,
                    "Ultralytics classification output shape family was not "
                    "recognized.",
                    ErrorContext{
                        .output_name = output.name,
                        .expected = std::string{"[C], [1,C], or [1,1,C]"},
                        .actual = detail::format_shape(output.shape),
                    })};
    }

    if (model.class_count.has_value() && class_count != *model.class_count) {
        return {
            .error = detail::make_adapter_error(
                kComponent, ErrorCode::shape_mismatch,
                "Ultralytics classification class count does not match the "
                "declared ModelSpec.",
                ErrorContext{
                    .output_name = output.name,
                    .expected = std::to_string(*model.class_count) + " classes",
                    .actual = std::to_string(class_count) + " classes",
                })};
    }

    ModelSpec resolved = model;
    resolved.task = TaskKind::classify;
    resolved.adapter = std::string{kAdapterName};
    if (!resolved.input_size.has_value()) {
        resolved.input_size = *input_size;
    }
    if (!resolved.class_count.has_value()) {
        resolved.class_count = class_count;
    }

    AdapterBindingSpec binding{};
    binding.model = std::move(resolved);
    binding.preprocess = make_classification_preprocess_policy(*input_size);
    binding.outputs.push_back(OutputBinding{
        .index = 0,
        .name = output.name,
        .role = OutputRole::predictions,
        .data_type = output.data_type,
        .shape = output.shape,
    });
    binding.classification = ClassificationBindingSpec{
        .class_count = class_count,
        .score_kind = detail::infer_classification_score_kind(output),
    };

    return {.value = std::move(binding), .error = {}};
}

}  // namespace

Result<AdapterBindingSpec> probe_classification(
    const ModelSpec& model, const std::vector<TensorInfo>& inputs,
    const std::vector<TensorInfo>& outputs) {
    return build_classification_binding(model, inputs, outputs);
}

Result<AdapterBindingSpec> probe_classification_model(const ModelSpec& model,
                                                      SessionOptions session) {
    const auto description = detail::describe_model(model, session);
    if (!description.ok()) {
        return {.error = description.error};
    }

    return build_classification_binding(model, description.value->inputs,
                                        description.value->outputs);
}

}  // namespace yolo::adapters::ultralytics
