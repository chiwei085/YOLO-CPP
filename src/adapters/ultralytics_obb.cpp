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

constexpr std::string_view kComponent = "ultralytics_obb_adapter";
constexpr std::size_t kObbCoordinateCount = 4;
constexpr std::size_t kObbAngleChannelCount = 1;
constexpr std::size_t kObbMinimumChannels =
    kObbCoordinateCount + kObbAngleChannelCount + 1;
constexpr std::size_t kObbMaxInferredClassCount = 32;

std::optional<ObbBindingSpec> infer_obb_binding_for_width(
    std::size_t width, std::size_t proposal_count, const ModelSpec& model,
    DetectionHeadLayout layout) {
    if (width < kObbMinimumChannels) {
        return std::nullopt;
    }

    const std::size_t class_count =
        model.class_count.has_value()
            ? *model.class_count
            : (width - kObbCoordinateCount - kObbAngleChannelCount);
    if (class_count == 0 ||
        width != kObbCoordinateCount + class_count + kObbAngleChannelCount) {
        return std::nullopt;
    }

    // Without an explicit class count, keep inference narrow enough to avoid
    // misclassifying common detect exports like [1,84,8400] as OBB.
    if (!model.class_count.has_value() &&
        class_count > kObbMaxInferredClassCount) {
        return std::nullopt;
    }

    return ObbBindingSpec{
        .layout = layout,
        .proposal_count = proposal_count,
        .class_count = class_count,
        .box_coordinate_count = kObbCoordinateCount,
        .class_channel_offset = kObbCoordinateCount,
        .angle_channel_offset = kObbCoordinateCount + class_count,
        .box_encoding = ObbBoxEncoding::center_size_rotation,
        .angle_is_radians = true,
        .external_nms = false,
    };
}

Result<AdapterBindingSpec> build_obb_binding(
    const ModelSpec& model, const std::vector<TensorInfo>& inputs,
    const std::vector<TensorInfo>& outputs) {
    if (inputs.size() != 1) {
        return {
            .error = detail::make_adapter_error(
                kComponent, ErrorCode::unsupported_model,
                "Ultralytics OBB models must expose exactly one input.",
                ErrorContext{
                    .expected = std::string{"1 input tensor"},
                    .actual = std::to_string(inputs.size()) + " input tensors",
                })};
    }

    if (outputs.size() != 1) {
        return {.error = detail::make_adapter_error(
                    kComponent, ErrorCode::unsupported_model,
                    "Ultralytics OBB models must expose exactly one output.",
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
                    "Ultralytics OBB expects a rank-4 RGB image input.",
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
                    "Ultralytics OBB output must be float32.",
                    ErrorContext{
                        .output_name = output.name,
                        .expected = std::string{"float32"},
                        .actual = detail::format_data_type(output.data_type),
                    })};
    }

    std::optional<ObbBindingSpec> obb{};
    if (output.shape.rank() == 3 && output.shape.dims[0].value.has_value() &&
        *output.shape.dims[0].value == 1 &&
        output.shape.dims[1].value.has_value() &&
        output.shape.dims[2].value.has_value()) {
        const std::size_t first =
            static_cast<std::size_t>(*output.shape.dims[1].value);
        const std::size_t second =
            static_cast<std::size_t>(*output.shape.dims[2].value);

        obb = infer_obb_binding_for_width(
            first, second, model, DetectionHeadLayout::xywh_class_scores_first);
        if (!obb.has_value()) {
            obb = infer_obb_binding_for_width(
                second, first, model,
                DetectionHeadLayout::xywh_class_scores_last);
        }
    }

    if (!obb.has_value()) {
        return {.error = detail::make_adapter_error(
                    kComponent, ErrorCode::unsupported_model,
                    "Ultralytics OBB output shape family was not recognized.",
                    ErrorContext{
                        .output_name = output.name,
                        .expected = std::string{"[1,4+C+1,N] or [1,N,4+C+1]"},
                        .actual = detail::format_shape(output.shape),
                    })};
    }

    if (model.class_count.has_value() &&
        obb->class_count != *model.class_count) {
        return {
            .error = detail::make_adapter_error(
                kComponent, ErrorCode::shape_mismatch,
                "Ultralytics OBB class count does not match the declared "
                "ModelSpec.",
                ErrorContext{
                    .output_name = output.name,
                    .expected = std::to_string(*model.class_count) + " classes",
                    .actual = std::to_string(obb->class_count) + " classes",
                })};
    }

    ModelSpec resolved = model;
    resolved.task = TaskKind::obb;
    resolved.adapter = std::string{kAdapterName};
    if (!resolved.input_size.has_value()) {
        resolved.input_size = *input_size;
    }
    if (!resolved.class_count.has_value()) {
        resolved.class_count = obb->class_count;
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
    binding.obb = *obb;

    return {.value = std::move(binding), .error = {}};
}

}  // namespace

Result<AdapterBindingSpec> probe_obb(const ModelSpec& model,
                                     const std::vector<TensorInfo>& inputs,
                                     const std::vector<TensorInfo>& outputs) {
    return build_obb_binding(model, inputs, outputs);
}

Result<AdapterBindingSpec> probe_obb_model(const ModelSpec& model,
                                           SessionOptions session) {
    const auto description = detail::describe_model(model, session);
    if (!description.ok()) {
        return {.error = description.error};
    }

    return build_obb_binding(model, description.value->inputs,
                             description.value->outputs);
}

}  // namespace yolo::adapters::ultralytics
