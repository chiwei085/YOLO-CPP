#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/tensor_utils.hpp"
#include "yolo/detail/ultralytics_adapter.hpp"

namespace yolo::adapters::ultralytics
{
namespace
{

constexpr std::string_view kComponent = "ultralytics_seg_adapter";

bool is_proto_tensor(const TensorInfo& tensor) {
    return tensor.shape.rank() == 4 && tensor.shape.dims.size() == 4 &&
           tensor.shape.dims[0].value.has_value() &&
           *tensor.shape.dims[0].value == 1 &&
           tensor.shape.dims[1].value.has_value() &&
           tensor.shape.dims[2].value.has_value() &&
           tensor.shape.dims[3].value.has_value();
}

Result<AdapterBindingSpec> build_segmentation_binding(
    const ModelSpec& model, const std::vector<TensorInfo>& inputs,
    const std::vector<TensorInfo>& outputs) {
    if (inputs.size() != 1) {
        return {
            .error = detail::make_adapter_error(
                kComponent, ErrorCode::unsupported_model,
                "Ultralytics segmentation models must expose exactly one "
                "input.",
                ErrorContext{
                    .expected = std::string{"1 input tensor"},
                    .actual = std::to_string(inputs.size()) + " input tensors",
                })};
    }

    if (outputs.size() != 2) {
        return {.error = detail::make_adapter_error(
                    kComponent, ErrorCode::unsupported_model,
                    "Ultralytics segmentation models must expose prediction "
                    "and proto outputs.",
                    ErrorContext{
                        .expected = std::string{"2 output tensors"},
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
                "Ultralytics segmentation expects a rank-4 RGB image input.",
                ErrorContext{
                    .input_name = input.name,
                    .expected = std::string{"[N,3,H,W] or [N,H,W,3]"},
                    .actual = detail::format_shape(input.shape),
                })};
    }

    std::optional<std::pair<std::size_t, TensorInfo>> proto_output{};
    std::optional<std::pair<std::size_t, TensorInfo>> prediction_output{};
    for (std::size_t index = 0; index < outputs.size(); ++index) {
        const TensorInfo& output = outputs[index];
        if (output.data_type != TensorDataType::float32) {
            return {
                .error = detail::make_adapter_error(
                    kComponent, ErrorCode::type_mismatch,
                    "Ultralytics segmentation outputs must be float32.",
                    ErrorContext{
                        .output_name = output.name,
                        .expected = std::string{"float32"},
                        .actual = detail::format_data_type(output.data_type),
                    })};
        }

        if (is_proto_tensor(output)) {
            proto_output = std::pair{index, output};
        }
        else {
            prediction_output = std::pair{index, output};
        }
    }

    if (!proto_output.has_value() || !prediction_output.has_value()) {
        return {.error = detail::make_adapter_error(
                    kComponent, ErrorCode::unsupported_model,
                    "Ultralytics segmentation could not find both prediction "
                    "and proto tensors.",
                    ErrorContext{
                        .expected = std::string{"one rank-3 prediction tensor "
                                                "and one rank-4 proto tensor"},
                    })};
    }

    const TensorInfo& proto = proto_output->second;
    const TensorInfo& prediction = prediction_output->second;
    const std::size_t mask_channels =
        static_cast<std::size_t>(*proto.shape.dims[1].value);

    SegmentationBindingSpec segmentation{};
    segmentation.mask_channel_count = mask_channels;
    if (prediction.shape.rank() == 3 &&
        prediction.shape.dims[0].value.has_value() &&
        *prediction.shape.dims[0].value == 1 &&
        prediction.shape.dims[1].value.has_value() &&
        prediction.shape.dims[2].value.has_value()) {
        const std::size_t first =
            static_cast<std::size_t>(*prediction.shape.dims[1].value);
        const std::size_t second =
            static_cast<std::size_t>(*prediction.shape.dims[2].value);
        if (second >= 4 + mask_channels + 1) {
            segmentation.layout = DetectionHeadLayout::xywh_class_scores_last;
            segmentation.proposal_count = first;
            segmentation.class_count =
                model.class_count.value_or(second - 4 - mask_channels);
        }
        else if (first >= 4 + mask_channels + 1) {
            segmentation.layout = DetectionHeadLayout::xywh_class_scores_first;
            segmentation.proposal_count = second;
            segmentation.class_count =
                model.class_count.value_or(first - 4 - mask_channels);
        }
        else {
            return {
                .error = detail::make_adapter_error(
                    kComponent, ErrorCode::unsupported_model,
                    "Ultralytics segmentation prediction output shape family "
                    "was not recognized.",
                    ErrorContext{
                        .output_name = prediction.name,
                        .expected = std::string{"[1,N,4+C+M] or [1,4+C+M,N]"},
                        .actual = detail::format_shape(prediction.shape),
                    })};
        }
    }
    else {
        return {.error = detail::make_adapter_error(
                    kComponent, ErrorCode::unsupported_model,
                    "Ultralytics segmentation prediction output shape family "
                    "was not recognized.",
                    ErrorContext{
                        .output_name = prediction.name,
                        .expected = std::string{"[1,N,4+C+M] or [1,4+C+M,N]"},
                        .actual = detail::format_shape(prediction.shape),
                    })};
    }

    if (model.class_count.has_value() &&
        segmentation.class_count != *model.class_count) {
        return {
            .error = detail::make_adapter_error(
                kComponent, ErrorCode::shape_mismatch,
                "Ultralytics segmentation class count does not match the "
                "declared ModelSpec.",
                ErrorContext{
                    .output_name = prediction.name,
                    .expected = std::to_string(*model.class_count) + " classes",
                    .actual =
                        std::to_string(segmentation.class_count) + " classes",
                })};
    }

    ModelSpec resolved = model;
    resolved.task = TaskKind::seg;
    resolved.adapter = std::string{kAdapterName};
    if (!resolved.input_size.has_value()) {
        resolved.input_size = *input_size;
    }
    if (!resolved.class_count.has_value()) {
        resolved.class_count = segmentation.class_count;
    }

    AdapterBindingSpec binding{};
    binding.model = std::move(resolved);
    binding.preprocess = make_detection_preprocess_policy(*input_size);
    binding.outputs.push_back(OutputBinding{
        .index = prediction_output->first,
        .name = prediction.name,
        .role = OutputRole::predictions,
        .data_type = prediction.data_type,
        .shape = prediction.shape,
    });
    binding.outputs.push_back(OutputBinding{
        .index = proto_output->first,
        .name = proto.name,
        .role = OutputRole::proto,
        .data_type = proto.data_type,
        .shape = proto.shape,
    });
    binding.segmentation = segmentation;

    return {.value = std::move(binding), .error = {}};
}

}  // namespace

Result<AdapterBindingSpec> probe_segmentation(
    const ModelSpec& model, const std::vector<TensorInfo>& inputs,
    const std::vector<TensorInfo>& outputs) {
    return build_segmentation_binding(model, inputs, outputs);
}

Result<AdapterBindingSpec> probe_segmentation_model(const ModelSpec& model,
                                                    SessionOptions session) {
    const auto description = detail::describe_model(model, session);
    if (!description.ok()) {
        return {.error = description.error};
    }

    return build_segmentation_binding(model, description.value->inputs,
                                      description.value->outputs);
}

}  // namespace yolo::adapters::ultralytics
