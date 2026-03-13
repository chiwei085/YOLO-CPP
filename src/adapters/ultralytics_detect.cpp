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

constexpr std::string_view kComponent = "ultralytics_detect_adapter";

Result<AdapterBindingSpec> build_detection_binding(
    const ModelSpec& model, const std::vector<TensorInfo>& inputs,
    const std::vector<TensorInfo>& outputs) {
    if (inputs.size() != 1) {
        return {
            .error = detail::make_adapter_error(
                kComponent, ErrorCode::unsupported_model,
                "Ultralytics detection models must expose exactly one input.",
                ErrorContext{
                    .expected = std::string{"1 input tensor"},
                    .actual = std::to_string(inputs.size()) + " input tensors",
                })};
    }

    if (outputs.size() != 1) {
        return {
            .error = detail::make_adapter_error(
                kComponent, ErrorCode::unsupported_model,
                "Ultralytics detection models must expose exactly one output.",
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
        return {.error = detail::make_adapter_error(
                    kComponent, ErrorCode::unsupported_model,
                    "Ultralytics detection expects a rank-4 RGB image input.",
                    ErrorContext{
                        .input_name = input.name,
                        .expected = std::string{"[N,3,H,W] or [N,H,W,3]"},
                        .actual = detail::format_shape(input.shape),
                    })};
    }

    const TensorInfo& output = outputs.front();
    DetectionBindingSpec detection{};
    if (output.data_type != TensorDataType::float32) {
        return {.error = detail::make_adapter_error(
                    kComponent, ErrorCode::type_mismatch,
                    "Ultralytics detection output must be float32.",
                    ErrorContext{
                        .output_name = output.name,
                        .expected = std::string{"float32"},
                        .actual = detail::format_data_type(output.data_type),
                    })};
    }

    if (output.shape.rank() == 3 && output.shape.dims[0].value.has_value() &&
        *output.shape.dims[0].value == 1 &&
        output.shape.dims[1].value.has_value() &&
        output.shape.dims[2].value.has_value()) {
        const std::size_t first =
            static_cast<std::size_t>(*output.shape.dims[1].value);
        const std::size_t second =
            static_cast<std::size_t>(*output.shape.dims[2].value);
        const std::size_t minimum_width = 6;

        if (second >= minimum_width &&
            (first < minimum_width || first >= second)) {
            detection.layout = DetectionHeadLayout::xywh_class_scores_last;
            detection.proposal_count = first;
            detection.class_count = model.class_count.value_or(second - 4);
        }
        else if (first >= minimum_width) {
            detection.layout = DetectionHeadLayout::xywh_class_scores_first;
            detection.proposal_count = second;
            detection.class_count = model.class_count.value_or(first - 4);
        }
        else {
            return {.error = detail::make_adapter_error(
                        kComponent, ErrorCode::unsupported_model,
                        "Ultralytics detection output shape family was not "
                        "recognized.",
                        ErrorContext{
                            .output_name = output.name,
                            .expected = std::string{"[1,N,4+C] or [1,4+C,N]"},
                            .actual = detail::format_shape(output.shape),
                        })};
        }
    }
    else if (output.shape.rank() == 2 &&
             output.shape.dims[0].value.has_value() &&
             output.shape.dims[1].value.has_value() &&
             *output.shape.dims[1].value >= 6) {
        detection.layout = DetectionHeadLayout::xyxy_score_class;
        detection.proposal_count =
            static_cast<std::size_t>(*output.shape.dims[0].value);
        detection.class_count = model.class_count.value_or(0U);
        detection.external_nms = true;
    }
    else {
        return {
            .error = detail::make_adapter_error(
                kComponent, ErrorCode::unsupported_model,
                "Ultralytics detection output shape family was not recognized.",
                ErrorContext{
                    .output_name = output.name,
                    .expected = std::string{"[1,N,4+C], [1,4+C,N], or [N,6+]"},
                    .actual = detail::format_shape(output.shape),
                })};
    }

    if (model.class_count.has_value() &&
        detection.class_count != *model.class_count &&
        detection.class_count != 0U) {
        return {
            .error = detail::make_adapter_error(
                kComponent, ErrorCode::shape_mismatch,
                "Ultralytics detection class count does not match the declared "
                "ModelSpec.",
                ErrorContext{
                    .output_name = output.name,
                    .expected = std::to_string(*model.class_count) + " classes",
                    .actual =
                        std::to_string(detection.class_count) + " classes",
                })};
    }

    ModelSpec resolved = model;
    resolved.task = TaskKind::detect;
    resolved.adapter = std::string{kAdapterName};
    if (!resolved.input_size.has_value()) {
        resolved.input_size = *input_size;
    }
    if (!resolved.class_count.has_value() && detection.class_count != 0U) {
        resolved.class_count = detection.class_count;
    }

    AdapterBindingSpec binding{};
    binding.model = std::move(resolved);
    binding.preprocess = make_detection_preprocess_policy(*input_size);
    binding.outputs.push_back(OutputBinding{
        .index = 0,
        .name = output.name,
        .role = OutputRole::predictions,
        .data_type = output.data_type,
        .shape = output.shape,
    });
    binding.detection = detection;

    return {.value = std::move(binding), .error = {}};
}

}  // namespace

Result<AdapterBindingSpec> probe_detection(
    const ModelSpec& model, const std::vector<TensorInfo>& inputs,
    const std::vector<TensorInfo>& outputs) {
    return build_detection_binding(model, inputs, outputs);
}

Result<AdapterBindingSpec> probe_detection_model(const ModelSpec& model,
                                                 SessionOptions session) {
    const auto description = detail::describe_model(model, session);
    if (!description.ok()) {
        return {.error = description.error};
    }

    return build_detection_binding(model, description.value->inputs,
                                   description.value->outputs);
}

}  // namespace yolo::adapters::ultralytics
