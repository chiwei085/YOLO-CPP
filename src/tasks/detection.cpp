#include "yolo/tasks/detection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/engine.hpp"
#include "yolo/detail/image_preprocess.hpp"

namespace yolo
{
namespace
{
using adapters::ultralytics::AdapterBindingSpec;
using adapters::ultralytics::DetectionBindingSpec;
using adapters::ultralytics::DetectionHeadLayout;
using adapters::ultralytics::OutputRole;

struct DetectionCandidate
{
    RectF bbox{};
    float score{0.0F};
    ClassId class_id{0};
};

enum class DetectionLayout
{
    xywh_class_scores_last,
    xywh_class_scores_first,
    xyxy_score_class,
};

struct DetectionDecodeSpec
{
    std::size_t output_index{0};
    DetectionLayout layout{DetectionLayout::xywh_class_scores_last};
    std::size_t proposal_count{0};
    std::size_t class_count{0};
};

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

Result<TensorInfo> select_primary_input(const detail::RuntimeEngine& engine) {
    const auto& inputs = engine.description().inputs;
    if (inputs.empty()) {
        return {.error = make_error(
                    ErrorCode::invalid_state,
                    "Detection engine has no input tensor metadata.",
                    ErrorContext{.component = std::string{"detection"}})};
    }

    return {.value = inputs.front(), .error = {}};
}

Result<DetectionDecodeSpec> detection_decode_spec_from_binding(
    const AdapterBindingSpec& binding) {
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

RectF xywh_to_rect(float cx, float cy, float w, float h) {
    return RectF{
        .x = cx - w * 0.5F,
        .y = cy - h * 0.5F,
        .width = w,
        .height = h,
    };
}

Result<std::vector<DetectionCandidate>> decode_detections(
    const detail::RawOutputTensors& outputs, const DetectionDecodeSpec& spec) {
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

    const auto float_values_result = detail::copy_float_tensor_data(
        outputs[spec.output_index], "detection_decoder");
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
    else if (record.resize_mode == ResizeMode::resize_crop) {
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

std::vector<Detection> postprocess_detections(
    std::vector<DetectionCandidate> candidates, const PreprocessRecord& record,
    const DetectionOptions& options, const ModelSpec& spec) {
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [&](const DetectionCandidate& candidate) {
                           return candidate.score <
                                      options.confidence_threshold ||
                                  candidate.bbox.width <= 0.0F ||
                                  candidate.bbox.height <= 0.0F;
                       }),
        candidates.end());

    std::sort(candidates.begin(), candidates.end(),
              [](const DetectionCandidate& lhs, const DetectionCandidate& rhs) {
                  return lhs.score > rhs.score;
              });

    std::vector<Detection> detections{};
    detections.reserve(std::min(options.max_detections, candidates.size()));
    std::vector<bool> suppressed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        detections.push_back(Detection{
            .bbox = unmap_rect(candidates[i].bbox, record),
            .score = candidates[i].score,
            .class_id = candidates[i].class_id,
            .label = label_for(spec, candidates[i].class_id),
        });
        if (detections.size() >= options.max_detections) {
            break;
        }

        for (std::size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            if (!options.class_agnostic_nms &&
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

    return detections;
}

InferenceMetadata make_metadata(const detail::RuntimeEngine& engine,
                                const detail::PreprocessedImage& preprocessed,
                                const SessionOptions& session) {
    return InferenceMetadata{
        .task = TaskKind::detect,
        .model_name = engine.model().model_name,
        .adapter_name = engine.model().adapter,
        .provider_name = provider_name_from_options(session),
        .original_image_size = preprocessed.record.source_size,
        .preprocess = preprocessed.record,
        .outputs = engine.description().outputs,
        .latency_ms = std::nullopt,
    };
}

class RuntimeDetector final : public Detector
{
public:
    RuntimeDetector(AdapterBindingSpec binding, SessionOptions session,
                    DetectionOptions options,
                    std::shared_ptr<detail::RuntimeEngine> engine)
        : binding_(std::move(binding)),
          spec_(binding_.model),
          session_(std::move(session)),
          options_(std::move(options)),
          engine_(std::move(engine)) {
        if (!engine_) {
            init_error_ = make_error(
                ErrorCode::invalid_state,
                "Detection runtime requires a valid shared engine.",
                ErrorContext{.component = std::string{"detection"}});
            return;
        }

        auto decode_spec_result = detection_decode_spec_from_binding(binding_);
        if (!decode_spec_result.ok()) {
            init_error_ = std::move(decode_spec_result.error);
            return;
        }

        decode_spec_ = *decode_spec_result.value;
    }

    RuntimeDetector(ModelSpec spec, SessionOptions session,
                    DetectionOptions options, Error init_error)
        : spec_(std::move(spec)),
          session_(std::move(session)),
          options_(std::move(options)),
          init_error_(std::move(init_error)) {}

    const ModelSpec& model() const noexcept override { return spec_; }

    DetectionResult run(const ImageView& image) const override {
        if (!init_error_.ok()) {
            return DetectionResult{
                {}, InferenceMetadata{.task = TaskKind::detect}, init_error_};
        }

        auto input_info_result = select_primary_input(*engine_);
        if (!input_info_result.ok()) {
            return DetectionResult{{},
                                   InferenceMetadata{.task = TaskKind::detect},
                                   input_info_result.error};
        }

        auto preprocess_result = detail::preprocess_image(
            image, binding_.preprocess, input_info_result.value->name);
        if (!preprocess_result.ok()) {
            return DetectionResult{{},
                                   InferenceMetadata{.task = TaskKind::detect},
                                   preprocess_result.error};
        }

        detail::RawInputTensor input{
            .info = preprocess_result.value->tensor.info,
            .bytes = preprocess_result.value->tensor.bytes(),
        };
        auto outputs_result = engine_->run(input);
        if (!outputs_result.ok()) {
            return DetectionResult{
                {},
                make_metadata(*engine_, *preprocess_result.value, session_),
                outputs_result.error};
        }

        auto decoded_result =
            decode_detections(*outputs_result.value, decode_spec_);
        if (!decoded_result.ok()) {
            return DetectionResult{
                {},
                make_metadata(*engine_, *preprocess_result.value, session_),
                decoded_result.error};
        }

        return DetectionResult{
            .detections = postprocess_detections(
                std::move(*decoded_result.value),
                preprocess_result.value->record, options_, spec_),
            .metadata =
                make_metadata(*engine_, *preprocess_result.value, session_),
            .error = {},
        };
    }

private:
    AdapterBindingSpec binding_{};
    ModelSpec spec_;
    SessionOptions session_;
    DetectionOptions options_;
    std::shared_ptr<detail::RuntimeEngine> engine_{};
    DetectionDecodeSpec decode_spec_{};
    Error init_error_{};
};

}  // namespace

std::unique_ptr<Detector> create_detector(ModelSpec spec,
                                          SessionOptions session,
                                          DetectionOptions options) {
    spec.task = TaskKind::detect;
    auto binding_result =
        adapters::ultralytics::probe_detection_model(spec, session);
    if (!binding_result.ok()) {
        return std::make_unique<RuntimeDetector>(
            std::move(spec), std::move(session), std::move(options),
            std::move(binding_result.error));
    }

    auto engine_result =
        detail::RuntimeEngine::create(binding_result.value->model, session);
    if (!engine_result.ok()) {
        return std::make_unique<RuntimeDetector>(
            binding_result.value->model, std::move(session), std::move(options),
            std::move(engine_result.error));
    }

    return std::make_unique<RuntimeDetector>(
        std::move(*binding_result.value), std::move(session),
        std::move(options),
        std::shared_ptr<detail::RuntimeEngine>(std::move(*engine_result.value)));
}

namespace detail
{

std::unique_ptr<Detector> create_detector_with_engine(
    AdapterBindingSpec binding, SessionOptions session, DetectionOptions options,
    std::shared_ptr<RuntimeEngine> engine) {
    binding.model.task = TaskKind::detect;
    return std::make_unique<RuntimeDetector>(std::move(binding),
                                             std::move(session),
                                             std::move(options),
                                             std::move(engine));
}

}  // namespace detail

}  // namespace yolo
