#include "yolo/tasks/segmentation.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/engine.hpp"
#include "yolo/detail/image_preprocess.hpp"
#include "yolo/detail/segmentation_runtime.hpp"
#include "yolo/detail/task_factory.hpp"
#include "yolo/detail/task_runtime_utils.hpp"

namespace yolo
{
namespace
{

using adapters::ultralytics::AdapterBindingSpec;
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

float intersection_over_union(const RectF& a, const RectF& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float w = std::max(0.0F, x2 - x1);
    const float h = std::max(0.0F, y2 - y1);
    const float inter = w * h;
    const float denom = a.area() + b.area() - inter;
    return denom > 0.0F ? inter / denom : 0.0F;
}

RectF unmap_rect(const RectF& rect, const PreprocessRecord& record) {
    RectF unmapped = rect;
    if (record.resize_mode == ResizeMode::letterbox) {
        unmapped.x = (unmapped.x - record.padding.left) / record.resize_scale.x;
        unmapped.y = (unmapped.y - record.padding.top) / record.resize_scale.y;
        unmapped.width /= record.resize_scale.x;
        unmapped.height /= record.resize_scale.y;
    }
    else if (record.resize_mode == ResizeMode::resize_crop && record.crop) {
        unmapped.x = (unmapped.x + record.crop->x) / record.resize_scale.x;
        unmapped.y = (unmapped.y + record.crop->y) / record.resize_scale.y;
        unmapped.width /= record.resize_scale.x;
        unmapped.height /= record.resize_scale.y;
    }
    else {
        unmapped.x /= record.resize_scale.x;
        unmapped.y /= record.resize_scale.y;
        unmapped.width /= record.resize_scale.x;
        unmapped.height /= record.resize_scale.y;
    }

    unmapped.x = std::clamp(unmapped.x, 0.0F,
                            static_cast<float>(record.source_size.width));
    unmapped.y = std::clamp(unmapped.y, 0.0F,
                            static_cast<float>(record.source_size.height));
    unmapped.width =
        std::clamp(unmapped.width, 0.0F,
                   static_cast<float>(record.source_size.width) - unmapped.x);
    unmapped.height =
        std::clamp(unmapped.height, 0.0F,
                   static_cast<float>(record.source_size.height) - unmapped.y);
    return unmapped;
}

std::optional<std::string> label_for(const ModelSpec& spec, ClassId class_id) {
    const std::size_t index = static_cast<std::size_t>(class_id);
    if (index < spec.labels.size()) {
        return spec.labels[index];
    }

    return std::nullopt;
}

float sigmoid(float value) { return 1.0F / (1.0F + std::exp(-value)); }

Point2f map_source_to_model(float source_x, float source_y,
                            const PreprocessRecord& record) {
    if (record.resize_mode == ResizeMode::letterbox) {
        return Point2f{
            .x = (source_x + 0.5F) * record.resize_scale.x - 0.5F +
                 static_cast<float>(record.padding.left),
            .y = (source_y + 0.5F) * record.resize_scale.y - 0.5F +
                 static_cast<float>(record.padding.top),
        };
    }

    if (record.resize_mode == ResizeMode::resize_crop && record.crop) {
        return Point2f{
            .x = (source_x + 0.5F) * record.resize_scale.x - 0.5F -
                 static_cast<float>(record.crop->x),
            .y = (source_y + 0.5F) * record.resize_scale.y - 0.5F -
                 static_cast<float>(record.crop->y),
        };
    }

    return Point2f{
        .x = (source_x + 0.5F) * record.resize_scale.x - 0.5F,
        .y = (source_y + 0.5F) * record.resize_scale.y - 0.5F,
    };
}

std::size_t proto_index(const detail::ProtoMaskTensor& proto,
                        std::size_t channel, int y, int x) {
    return (channel * static_cast<std::size_t>(proto.size.height) +
            static_cast<std::size_t>(y)) *
               static_cast<std::size_t>(proto.size.width) +
           static_cast<std::size_t>(x);
}

float bilinear_sample_plane(const std::vector<float>& plane, const Size2i& size,
                            float x, float y) {
    if (plane.empty() || size.empty()) {
        return 0.0F;
    }

    const float clamped_x =
        std::clamp(x, 0.0F, static_cast<float>(size.width - 1));
    const float clamped_y =
        std::clamp(y, 0.0F, static_cast<float>(size.height - 1));
    const int x0 = static_cast<int>(std::floor(clamped_x));
    const int y0 = static_cast<int>(std::floor(clamped_y));
    const int x1 = std::min(x0 + 1, size.width - 1);
    const int y1 = std::min(y0 + 1, size.height - 1);
    const float dx = clamped_x - static_cast<float>(x0);
    const float dy = clamped_y - static_cast<float>(y0);

    const auto value_at = [&](int px, int py) {
        return plane[static_cast<std::size_t>(py) *
                         static_cast<std::size_t>(size.width) +
                     static_cast<std::size_t>(px)];
    };

    const float top =
        value_at(x0, y0) * (1.0F - dx) + value_at(x1, y0) * dx;
    const float bottom =
        value_at(x0, y1) * (1.0F - dx) + value_at(x1, y1) * dx;
    return top * (1.0F - dy) + bottom * dy;
}

InferenceMetadata make_metadata(const detail::RuntimeEngine& engine,
                                const detail::PreprocessedImage& preprocessed,
                                const SessionOptions& session) {
    return detail::make_task_metadata(TaskKind::seg, engine, preprocessed,
                                      session);
}

class RuntimeSegmenter final : public Segmenter
{
public:
    RuntimeSegmenter(AdapterBindingSpec binding, SessionOptions session,
                     SegmentationOptions options,
                     std::shared_ptr<detail::RuntimeEngine> engine)
        : binding_(std::move(binding)),
          spec_(binding_.model),
          session_(std::move(session)),
          options_(std::move(options)),
          engine_(std::move(engine)) {
        if (!engine_) {
            init_error_ = make_error(
                ErrorCode::invalid_state,
                "Segmentation runtime requires a valid shared engine.",
                ErrorContext{.component = std::string{"segmentation"}});
            return;
        }

        auto decode_spec_result =
            detail::segmentation_decode_spec_from_binding(binding_);
        if (!decode_spec_result.ok()) {
            init_error_ = std::move(decode_spec_result.error);
            return;
        }

        decode_spec_ = *decode_spec_result.value;
    }

    RuntimeSegmenter(ModelSpec spec, SessionOptions session,
                     SegmentationOptions options, Error init_error)
        : spec_(std::move(spec)),
          session_(std::move(session)),
          options_(std::move(options)),
          init_error_(std::move(init_error)) {}

    const ModelSpec& model() const noexcept override { return spec_; }

    SegmentationResult run(const ImageView& image) const override {
        if (!init_error_.ok()) {
            return SegmentationResult{
                .instances = {},
                .metadata = InferenceMetadata{.task = TaskKind::seg},
                .error = init_error_,
            };
        }

        auto input_info_result =
            detail::select_primary_input(*engine_, "segmentation");
        if (!input_info_result.ok()) {
            return SegmentationResult{
                .instances = {},
                .metadata = InferenceMetadata{.task = TaskKind::seg},
                .error = input_info_result.error,
            };
        }

        auto preprocess_result = detail::preprocess_image(
            image, binding_.preprocess, input_info_result.value->name);
        if (!preprocess_result.ok()) {
            return SegmentationResult{
                .instances = {},
                .metadata = InferenceMetadata{.task = TaskKind::seg},
                .error = preprocess_result.error,
            };
        }

        detail::RawInputTensor input{
            .info = preprocess_result.value->tensor.info,
            .bytes = preprocess_result.value->tensor.bytes(),
        };
        auto outputs_result = engine_->run(input);
        if (!outputs_result.ok()) {
            return SegmentationResult{
                .instances = {},
                .metadata =
                    make_metadata(*engine_, *preprocess_result.value, session_),
                .error = outputs_result.error,
            };
        }

        auto decoded_result =
            detail::decode_segmentation(*outputs_result.value, decode_spec_);
        if (!decoded_result.ok()) {
            return SegmentationResult{
                .instances = {},
                .metadata =
                    make_metadata(*engine_, *preprocess_result.value, session_),
                .error = decoded_result.error,
            };
        }

        return SegmentationResult{
            .instances = detail::postprocess_segmentation(
                std::move(decoded_result.value->candidates),
                decoded_result.value->proto, preprocess_result.value->record,
                options_, spec_),
            .metadata =
                make_metadata(*engine_, *preprocess_result.value, session_),
            .error = {},
        };
    }

private:
    AdapterBindingSpec binding_{};
    ModelSpec spec_;
    SessionOptions session_;
    SegmentationOptions options_;
    std::shared_ptr<detail::RuntimeEngine> engine_{};
    detail::SegmentationDecodeSpec decode_spec_{};
    Error init_error_{};
};

}  // namespace

std::unique_ptr<Segmenter> create_segmenter(ModelSpec spec,
                                            SessionOptions session,
                                            SegmentationOptions options) {
    spec.task = TaskKind::seg;
    auto binding_result =
        adapters::ultralytics::probe_segmentation_model(spec, session);
    if (!binding_result.ok()) {
        return std::make_unique<RuntimeSegmenter>(
            std::move(spec), std::move(session), std::move(options),
            std::move(binding_result.error));
    }

    auto engine_result =
        detail::RuntimeEngine::create(binding_result.value->model, session);
    if (!engine_result.ok()) {
        return std::make_unique<RuntimeSegmenter>(
            binding_result.value->model, std::move(session), std::move(options),
            std::move(engine_result.error));
    }

    return std::make_unique<RuntimeSegmenter>(
        std::move(*binding_result.value), std::move(session),
        std::move(options),
        std::shared_ptr<detail::RuntimeEngine>(std::move(*engine_result.value)));
}

namespace detail
{

Result<SegmentationDecodeSpec> segmentation_decode_spec_from_binding(
    const adapters::ultralytics::AdapterBindingSpec& binding) {
    if (!binding.segmentation.has_value()) {
        return {.error = make_error(
                    ErrorCode::invalid_state,
                    "Segmentation runtime requires a segmentation binding "
                    "spec.",
                    ErrorContext{
                        .component = std::string{"segmentation"}})};
    }

    std::optional<std::size_t> prediction_index{};
    std::optional<std::size_t> proto_index{};
    std::optional<Size2i> proto_size{};
    for (const auto& output : binding.outputs) {
        if (output.role == OutputRole::predictions) {
            prediction_index = output.index;
        }
        else if (output.role == OutputRole::proto) {
            proto_index = output.index;
            if (output.shape.rank() == 4 &&
                output.shape.dims[2].value.has_value() &&
                output.shape.dims[3].value.has_value()) {
                proto_size = Size2i{
                    .width = static_cast<int>(*output.shape.dims[3].value),
                    .height = static_cast<int>(*output.shape.dims[2].value),
                };
            }
        }
    }

    if (!prediction_index.has_value() || !proto_index.has_value() ||
        !proto_size.has_value()) {
        return {.error = make_error(
                    ErrorCode::invalid_state,
                    "Segmentation binding must expose prediction and proto "
                    "outputs.",
                    ErrorContext{
                        .component = std::string{"segmentation"}})};
    }

    DetectionLayout layout = DetectionLayout::xywh_class_scores_last;
    switch (binding.segmentation->layout) {
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

    return {.value = SegmentationDecodeSpec{
                .output_index = *prediction_index,
                .proto_output_index = *proto_index,
                .layout = layout,
                .proposal_count = binding.segmentation->proposal_count,
                .class_count = binding.segmentation->class_count,
                .mask_channel_count = binding.segmentation->mask_channel_count,
                .proto_size = *proto_size,
            },
            .error = {}};
}

Result<DecodedSegmentation> decode_segmentation(const RawOutputTensors& outputs,
                                                const SegmentationDecodeSpec& spec) {
    if (spec.output_index >= outputs.size() ||
        spec.proto_output_index >= outputs.size()) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "Segmentation binding points to a missing output tensor.",
                    ErrorContext{
                        .component = std::string{"segmentation_decoder"},
                        .expected =
                            std::to_string(std::max(spec.output_index,
                                                    spec.proto_output_index) +
                                           1) +
                            " output tensors",
                        .actual = std::to_string(outputs.size()) +
                                  " output tensors",
                    })};
    }

    const auto proto_values_result = copy_float_tensor_data(
        outputs[spec.proto_output_index], "segmentation_decoder");
    if (!proto_values_result.ok()) {
        return {.error = proto_values_result.error};
    }

    const TensorInfo& proto_info = outputs[spec.proto_output_index].info;
    if (proto_info.shape.rank() != 4 ||
        !proto_info.shape.dims[1].value.has_value() ||
        !proto_info.shape.dims[2].value.has_value() ||
        !proto_info.shape.dims[3].value.has_value()) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "Segmentation proto tensor must have rank 4 with static "
                    "dimensions.",
                    ErrorContext{
                        .component = std::string{"segmentation_decoder"},
                        .output_name = proto_info.name,
                        .actual = format_shape(proto_info.shape),
                    })};
    }

    if (static_cast<std::size_t>(*proto_info.shape.dims[1].value) !=
        spec.mask_channel_count) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "Segmentation proto tensor channel count does not match "
                    "binding.",
                    ErrorContext{
                        .component = std::string{"segmentation_decoder"},
                        .output_name = proto_info.name,
                        .expected = std::to_string(spec.mask_channel_count),
                        .actual = std::to_string(
                            *proto_info.shape.dims[1].value),
                    })};
    }

    auto prediction_values_result = copy_float_tensor_data(
        outputs[spec.output_index], "segmentation_decoder");
    if (!prediction_values_result.ok()) {
        return {.error = prediction_values_result.error};
    }

    const std::vector<float>& values = *prediction_values_result.value;
    std::vector<SegmentationCandidate> candidates{};
    candidates.reserve(spec.proposal_count);
    const std::size_t stride = 4 + spec.class_count + spec.mask_channel_count;

    if (spec.layout == DetectionLayout::xywh_class_scores_last) {
        if (values.size() < spec.proposal_count * stride) {
            return {.error = make_error(
                        ErrorCode::shape_mismatch,
                        "Segmentation prediction tensor payload is smaller "
                        "than its decoded shape.",
                        ErrorContext{
                            .component = std::string{"segmentation_decoder"},
                            .output_name = outputs[spec.output_index].info.name,
                            .expected =
                                std::to_string(spec.proposal_count * stride),
                            .actual = std::to_string(values.size()),
                        })};
        }

        for (std::size_t i = 0; i < spec.proposal_count; ++i) {
            const float* row = values.data() + i * stride;
            auto class_begin = row + 4;
            auto class_end = row + 4 + spec.class_count;
            const auto class_it = std::max_element(class_begin, class_end);
            const ClassId class_id = static_cast<ClassId>(
                std::distance(class_begin, class_it));
            candidates.push_back(SegmentationCandidate{
                .bbox = xywh_to_rect(row[0], row[1], row[2], row[3]),
                .score = class_it == class_end ? 0.0F : *class_it,
                .class_id = class_id,
                .mask_coefficients = std::vector<float>(
                    row + 4 + spec.class_count, row + stride),
            });
        }
    }
    else if (spec.layout == DetectionLayout::xywh_class_scores_first) {
        const std::size_t proposal_stride = spec.proposal_count;
        const std::size_t required_values = proposal_stride * stride;
        if (values.size() < required_values) {
            return {.error = make_error(
                        ErrorCode::shape_mismatch,
                        "Segmentation prediction tensor payload is smaller "
                        "than its decoded shape.",
                        ErrorContext{
                            .component = std::string{"segmentation_decoder"},
                            .output_name = outputs[spec.output_index].info.name,
                            .expected = std::to_string(required_values),
                            .actual = std::to_string(values.size()),
                        })};
        }

        for (std::size_t i = 0; i < spec.proposal_count; ++i) {
            float best_score = 0.0F;
            ClassId class_id = 0;
            for (std::size_t c = 0; c < spec.class_count; ++c) {
                const float score = values[proposal_stride * (4 + c) + i];
                if (score > best_score) {
                    best_score = score;
                    class_id = static_cast<ClassId>(c);
                }
            }

            std::vector<float> coefficients(spec.mask_channel_count, 0.0F);
            for (std::size_t m = 0; m < spec.mask_channel_count; ++m) {
                coefficients[m] = values[proposal_stride *
                                             (4 + spec.class_count + m) +
                                         i];
            }

            candidates.push_back(SegmentationCandidate{
                .bbox = xywh_to_rect(values[i], values[proposal_stride + i],
                                     values[proposal_stride * 2 + i],
                                     values[proposal_stride * 3 + i]),
                .score = best_score,
                .class_id = class_id,
                .mask_coefficients = std::move(coefficients),
            });
        }
    }
    else {
        return {.error = make_error(
                    ErrorCode::unsupported_model,
                    "Segmentation runtime does not support this output layout.",
                    ErrorContext{
                        .component = std::string{"segmentation_decoder"}})};
    }

    return {.value = DecodedSegmentation{
                .candidates = std::move(candidates),
                .proto = ProtoMaskTensor{
                    .size = spec.proto_size,
                    .channel_count = spec.mask_channel_count,
                    .values = *proto_values_result.value,
                },
            },
            .error = {}};
}

SegmentationMask project_segmentation_mask(const SegmentationCandidate& candidate,
                                           const ProtoMaskTensor& proto,
                                           const PreprocessRecord& record,
                                           float threshold) {
    SegmentationMask mask{};
    mask.size = record.source_size;
    mask.data.assign(static_cast<std::size_t>(std::max(0, record.source_size.width)) *
                         static_cast<std::size_t>(std::max(0, record.source_size.height)),
                     static_cast<std::uint8_t>(0));

    if (mask.size.empty() || proto.size.empty() || proto.channel_count == 0 ||
        candidate.mask_coefficients.size() != proto.channel_count) {
        return mask;
    }

    std::vector<float> proto_logits(
        static_cast<std::size_t>(proto.size.width * proto.size.height), 0.0F);
    for (int y = 0; y < proto.size.height; ++y) {
        for (int x = 0; x < proto.size.width; ++x) {
            float value = 0.0F;
            for (std::size_t channel = 0; channel < proto.channel_count;
                 ++channel) {
                value += candidate.mask_coefficients[channel] *
                         proto.values[proto_index(proto, channel, y, x)];
            }
            proto_logits[static_cast<std::size_t>(y * proto.size.width + x)] =
                value;
        }
    }

    const RectF source_bbox = unmap_rect(candidate.bbox, record);
    const int x0 = std::max(0, static_cast<int>(std::floor(source_bbox.x)));
    const int y0 = std::max(0, static_cast<int>(std::floor(source_bbox.y)));
    const int x1 = std::min(mask.size.width,
                            static_cast<int>(std::ceil(source_bbox.x +
                                                       source_bbox.width)));
    const int y1 = std::min(mask.size.height,
                            static_cast<int>(std::ceil(source_bbox.y +
                                                       source_bbox.height)));

    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            const Point2f model_point = map_source_to_model(
                static_cast<float>(x), static_cast<float>(y), record);
            if (model_point.x < candidate.bbox.x ||
                model_point.y < candidate.bbox.y ||
                model_point.x >= candidate.bbox.x + candidate.bbox.width ||
                model_point.y >= candidate.bbox.y + candidate.bbox.height) {
                continue;
            }

            const float proto_x =
                ((model_point.x + 0.5F) * static_cast<float>(proto.size.width) /
                     static_cast<float>(record.target_size.width)) -
                0.5F;
            const float proto_y =
                ((model_point.y + 0.5F) *
                     static_cast<float>(proto.size.height) /
                     static_cast<float>(record.target_size.height)) -
                0.5F;
            const float value = bilinear_sample_plane(proto_logits, proto.size,
                                                      proto_x, proto_y);
            if (value > 0.0F || sigmoid(value) > threshold) {
                mask.data[static_cast<std::size_t>(y) *
                              static_cast<std::size_t>(mask.size.width) +
                          static_cast<std::size_t>(x)] = 1;
            }
        }
    }

    return mask;
}

std::vector<SegmentationInstance> postprocess_segmentation(
    std::vector<SegmentationCandidate> candidates, const ProtoMaskTensor& proto,
    const PreprocessRecord& record, const SegmentationOptions& options,
    const ModelSpec& spec) {
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [&](const SegmentationCandidate& candidate) {
                           return candidate.score <
                                      options.confidence_threshold ||
                                  candidate.bbox.width <= 0.0F ||
                                  candidate.bbox.height <= 0.0F;
                       }),
        candidates.end());

    std::sort(candidates.begin(), candidates.end(),
              [](const SegmentationCandidate& lhs,
                 const SegmentationCandidate& rhs) {
                  return lhs.score > rhs.score;
              });

    std::vector<SegmentationInstance> instances{};
    instances.reserve(std::min(options.max_detections, candidates.size()));
    std::vector<bool> suppressed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        instances.push_back(SegmentationInstance{
            .bbox = unmap_rect(candidates[i].bbox, record),
            .score = candidates[i].score,
            .class_id = candidates[i].class_id,
            .label = label_for(spec, candidates[i].class_id),
            .mask = project_segmentation_mask(candidates[i], proto, record),
        });
        if (instances.size() >= options.max_detections) {
            break;
        }

        for (std::size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j] ||
                candidates[i].class_id != candidates[j].class_id) {
                continue;
            }

            if (intersection_over_union(candidates[i].bbox,
                                        candidates[j].bbox) >=
                options.nms_iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return instances;
}

std::unique_ptr<Segmenter> create_segmenter_with_engine(
    AdapterBindingSpec binding, SessionOptions session,
    SegmentationOptions options, std::shared_ptr<RuntimeEngine> engine) {
    binding.model.task = TaskKind::seg;
    return std::make_unique<RuntimeSegmenter>(std::move(binding),
                                              std::move(session),
                                              std::move(options),
                                              std::move(engine));
}

}  // namespace detail

}  // namespace yolo
