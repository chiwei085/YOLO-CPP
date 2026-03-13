#include "yolo/detail/detection_runtime.hpp"

#include <algorithm>
#include <string>

namespace yolo::detail
{
namespace
{

using adapters::ultralytics::DetectionBindingSpec;
using adapters::ultralytics::DetectionHeadLayout;
using adapters::ultralytics::OutputRole;

RectF xywh_to_rect(float cx, float cy, float w, float h) {
    return RectF{
        .x = cx - w * 0.5F,
        .y = cy - h * 0.5F,
        .width = w,
        .height = h,
    };
}

}  // namespace

Result<DetectionDecodeSpec> detection_decode_spec_from_binding(
    const adapters::ultralytics::AdapterBindingSpec& binding) {
    if (!binding.detection.has_value()) {
        return {.error = make_error(
                    ErrorCode::invalid_state,
                    "Detection runtime requires a detection binding spec.",
                    ErrorContext{.component = std::string{"detection"}})};
    }

    if (binding.outputs.empty()) {
        return {.error = make_error(
                    ErrorCode::invalid_state,
                    "Detection runtime requires at least one bound output.",
                    ErrorContext{.component = std::string{"detection"}})};
    }

    const DetectionBindingSpec& detection = *binding.detection;
    DetectionLayout layout = DetectionLayout::xywh_class_scores_last;
    switch (detection.layout) {
        case DetectionHeadLayout::xywh_class_scores_last:
            layout = DetectionLayout::xywh_class_scores_last;
            break;
        case DetectionHeadLayout::xywh_class_scores_first:
            layout = DetectionLayout::xywh_class_scores_first;
            break;
        case DetectionHeadLayout::xyxy_score_class:
            layout = DetectionLayout::xyxy_score_class;
            break;
    }

    std::size_t output_index = binding.outputs.front().index;
    for (const auto& output : binding.outputs) {
        if (output.role == OutputRole::predictions) {
            output_index = output.index;
            break;
        }
    }

    return {.value = DetectionDecodeSpec{
                .output_index = output_index,
                .layout = layout,
                .proposal_count = detection.proposal_count,
                .class_count = detection.class_count,
            },
            .error = {}};
}

Result<std::vector<DetectionCandidate>> decode_detections(
    const RawOutputTensors& outputs, const DetectionDecodeSpec& spec) {
    if (spec.output_index >= outputs.size()) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "Detection binding points to a missing output tensor.",
                    ErrorContext{
                        .component = std::string{"detection_decoder"},
                        .expected =
                            std::to_string(spec.output_index + 1) +
                            " output tensors",
                        .actual = std::to_string(outputs.size()) +
                                  " output tensors",
                    })};
    }

    const auto float_values_result =
        copy_float_tensor_data(outputs[spec.output_index], "detection_decoder");
    if (!float_values_result.ok()) {
        return {.error = float_values_result.error};
    }

    const std::vector<float>& values = *float_values_result.value;
    std::vector<DetectionCandidate> candidates{};
    candidates.reserve(spec.proposal_count);

    if (spec.layout == DetectionLayout::xywh_class_scores_last) {
        const std::size_t stride = 4 + spec.class_count;
        if (values.size() < spec.proposal_count * stride) {
            return {.error = make_error(
                        ErrorCode::shape_mismatch,
                        "Detection output tensor payload is smaller than its "
                        "decoded shape.",
                        ErrorContext{
                            .component = std::string{"detection_decoder"},
                            .output_name = outputs[spec.output_index].info.name,
                            .expected =
                                std::to_string(spec.proposal_count * stride),
                            .actual = std::to_string(values.size()),
                        })};
        }

        for (std::size_t i = 0; i < spec.proposal_count; ++i) {
            const float* row = values.data() + i * stride;
            auto class_begin = row + 4;
            auto class_end = row + stride;
            const auto class_it = std::max_element(class_begin, class_end);
            const float best_score = class_it == class_end ? 0.0F : *class_it;
            const ClassId class_id = class_it == class_end
                                         ? 0U
                                         : static_cast<ClassId>(std::distance(
                                               class_begin, class_it));
            candidates.push_back(DetectionCandidate{
                .bbox = xywh_to_rect(row[0], row[1], row[2], row[3]),
                .score = best_score,
                .class_id = class_id,
            });
        }
    }
    else if (spec.layout == DetectionLayout::xywh_class_scores_first) {
        const std::size_t proposal_stride = spec.proposal_count;
        const std::size_t required_values =
            proposal_stride * (4 + spec.class_count);
        if (values.size() < required_values) {
            return {.error = make_error(
                        ErrorCode::shape_mismatch,
                        "Detection output tensor payload is smaller than its "
                        "decoded shape.",
                        ErrorContext{
                            .component = std::string{"detection_decoder"},
                            .output_name = outputs[spec.output_index].info.name,
                            .expected = std::to_string(required_values),
                            .actual = std::to_string(values.size()),
                        })};
        }

        for (std::size_t i = 0; i < spec.proposal_count; ++i) {
            const float cx = values[i];
            const float cy = values[proposal_stride + i];
            const float w = values[proposal_stride * 2 + i];
            const float h = values[proposal_stride * 3 + i];

            float best_score = 0.0F;
            ClassId class_id = 0;
            for (std::size_t c = 0; c < spec.class_count; ++c) {
                const float score = values[proposal_stride * (4 + c) + i];
                if (score > best_score) {
                    best_score = score;
                    class_id = static_cast<ClassId>(c);
                }
            }

            candidates.push_back(DetectionCandidate{
                .bbox = xywh_to_rect(cx, cy, w, h),
                .score = best_score,
                .class_id = class_id,
            });
        }
    }
    else {
        const std::size_t stride = static_cast<std::size_t>(
            *outputs[spec.output_index].info.shape.dims[1].value);
        if (values.size() < spec.proposal_count * stride) {
            return {.error = make_error(
                        ErrorCode::shape_mismatch,
                        "Detection output tensor payload is smaller than its "
                        "decoded shape.",
                        ErrorContext{
                            .component = std::string{"detection_decoder"},
                            .output_name = outputs[spec.output_index].info.name,
                            .expected =
                                std::to_string(spec.proposal_count * stride),
                            .actual = std::to_string(values.size()),
                        })};
        }

        for (std::size_t i = 0; i < spec.proposal_count; ++i) {
            const float* row = values.data() + i * stride;
            candidates.push_back(DetectionCandidate{
                .bbox =
                    RectF{
                        .x = row[0],
                        .y = row[1],
                        .width = row[2] - row[0],
                        .height = row[3] - row[1],
                    },
                .score = row[4],
                .class_id = static_cast<ClassId>(std::max(0.0F, row[5])),
            });
        }
    }

    return {.value = std::move(candidates), .error = {}};
}

}  // namespace yolo::detail
