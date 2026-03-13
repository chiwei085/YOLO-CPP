#include "yolo/tasks/detection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "yolo/detail/engine.hpp"
#include "yolo/detail/image_preprocess.hpp"

namespace yolo
{
namespace
{

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

bool is_channel_dimension(const TensorDimension& dim) {
    return dim.value.has_value() &&
           (*dim.value == 1 || *dim.value == 3 || *dim.value == 4);
}

Result<Size2i> resolve_input_size(const ModelSpec& spec,
                                  const TensorInfo& input) {
    if (spec.input_size.has_value()) {
        return {.value = *spec.input_size, .error = {}};
    }

    if (input.shape.rank() != 4) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "Detection input tensor must be rank-4.",
                    ErrorContext{
                        .component = std::string{"detection"},
                        .input_name = input.name,
                        .expected = std::string{"[N,C,H,W] or [N,H,W,C]"},
                        .actual = detail::format_shape(input.shape),
                    })};
    }

    const auto& dims = input.shape.dims;
    if (is_channel_dimension(dims[1]) && dims[2].value.has_value() &&
        dims[3].value.has_value()) {
        return {.value = Size2i{static_cast<int>(*dims[3].value),
                                static_cast<int>(*dims[2].value)},
                .error = {}};
    }

    if (dims[1].value.has_value() && dims[2].value.has_value() &&
        is_channel_dimension(dims[3])) {
        return {.value = Size2i{static_cast<int>(*dims[2].value),
                                static_cast<int>(*dims[1].value)},
                .error = {}};
    }

    return {.error = make_error(
                ErrorCode::shape_mismatch,
                "Detection input tensor has unsupported spatial layout.",
                ErrorContext{
                    .component = std::string{"detection"},
                    .input_name = input.name,
                    .expected = std::string{"static [N,C,H,W] or [N,H,W,C]"},
                    .actual = detail::format_shape(input.shape),
                })};
}

Result<DetectionDecodeSpec> infer_detection_decode_spec(
    const detail::RawOutputTensors& outputs, const ModelSpec& spec) {
    if (outputs.empty()) {
        return {
            .error = make_error(
                ErrorCode::shape_mismatch,
                "Detection decoder requires at least one output tensor.",
                ErrorContext{.component = std::string{"detection_decoder"}})};
    }

    const TensorInfo& info = outputs.front().info;
    if (info.shape.rank() == 3 && info.shape.dims[0].value.has_value() &&
        *info.shape.dims[0].value == 1 &&
        info.shape.dims[1].value.has_value() &&
        info.shape.dims[2].value.has_value()) {
        const std::size_t proposals =
            static_cast<std::size_t>(*info.shape.dims[1].value);
        const std::size_t attributes =
            static_cast<std::size_t>(*info.shape.dims[2].value);
        if (attributes >= 6) {
            const std::size_t class_count =
                spec.class_count.value_or(attributes - 4);
            return {.value =
                        DetectionDecodeSpec{
                            .output_index = 0,
                            .layout = DetectionLayout::xywh_class_scores_last,
                            .proposal_count = proposals,
                            .class_count = class_count,
                        },
                    .error = {}};
        }
    }

    if (info.shape.rank() == 3 && info.shape.dims[0].value.has_value() &&
        *info.shape.dims[0].value == 1 &&
        info.shape.dims[1].value.has_value() &&
        info.shape.dims[2].value.has_value()) {
        const std::size_t channels =
            static_cast<std::size_t>(*info.shape.dims[1].value);
        const std::size_t proposals =
            static_cast<std::size_t>(*info.shape.dims[2].value);
        if (channels >= 6) {
            const std::size_t class_count =
                spec.class_count.value_or(channels - 4);
            return {.value =
                        DetectionDecodeSpec{
                            .output_index = 0,
                            .layout = DetectionLayout::xywh_class_scores_first,
                            .proposal_count = proposals,
                            .class_count = class_count,
                        },
                    .error = {}};
        }
    }

    if (info.shape.rank() == 2 && info.shape.dims[0].value.has_value() &&
        info.shape.dims[1].value.has_value() &&
        *info.shape.dims[1].value >= 6) {
        return {.value =
                    DetectionDecodeSpec{
                        .output_index = 0,
                        .layout = DetectionLayout::xyxy_score_class,
                        .proposal_count =
                            static_cast<std::size_t>(*info.shape.dims[0].value),
                        .class_count = spec.class_count.value_or(0),
                    },
                .error = {}};
    }

    return {.error = make_error(
                ErrorCode::shape_mismatch,
                "No supported detection output shape family matched.",
                ErrorContext{
                    .component = std::string{"detection_decoder"},
                    .output_name = info.name,
                    .expected =
                        std::string{
                            "[1,N,4+C], [1,4+C,N], or [N,6+] detection output"},
                    .actual = detail::format_shape(info.shape),
                })};
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
    RuntimeDetector(ModelSpec spec, SessionOptions session,
                    DetectionOptions options)
        : spec_(std::move(spec)),
          session_(std::move(session)),
          options_(std::move(options)) {
        auto engine_result = detail::RuntimeEngine::create(spec_, session_);
        if (engine_result.ok()) {
            engine_ = std::shared_ptr<detail::RuntimeEngine>(
                std::move(*engine_result.value));
        }
        else {
            init_error_ = std::move(engine_result.error);
        }
    }

    RuntimeDetector(ModelSpec spec, SessionOptions session,
                    DetectionOptions options,
                    std::shared_ptr<detail::RuntimeEngine> engine)
        : spec_(std::move(spec)),
          session_(std::move(session)),
          options_(std::move(options)),
          engine_(std::move(engine)) {
        if (!engine_) {
            init_error_ = make_error(
                ErrorCode::invalid_state,
                "Detection runtime requires a valid shared engine.",
                ErrorContext{.component = std::string{"detection"}});
        }
    }

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

        auto input_size_result =
            resolve_input_size(spec_, *input_info_result.value);
        if (!input_size_result.ok()) {
            return DetectionResult{{},
                                   InferenceMetadata{.task = TaskKind::detect},
                                   input_size_result.error};
        }

        const PreprocessPolicy policy =
            make_detection_preprocess_policy(*input_size_result.value);
        auto preprocess_result = detail::preprocess_image(
            image, policy, input_info_result.value->name);
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

        auto decode_spec_result =
            infer_detection_decode_spec(*outputs_result.value, spec_);
        if (!decode_spec_result.ok()) {
            return DetectionResult{
                {},
                make_metadata(*engine_, *preprocess_result.value, session_),
                decode_spec_result.error};
        }

        auto decoded_result =
            decode_detections(*outputs_result.value, *decode_spec_result.value);
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
    ModelSpec spec_;
    SessionOptions session_;
    DetectionOptions options_;
    std::shared_ptr<detail::RuntimeEngine> engine_{};
    Error init_error_{};
};

}  // namespace

std::unique_ptr<Detector> create_detector(ModelSpec spec,
                                          SessionOptions session,
                                          DetectionOptions options) {
    spec.task = TaskKind::detect;
    return std::make_unique<RuntimeDetector>(
        std::move(spec), std::move(session), std::move(options));
}

namespace detail
{

std::unique_ptr<Detector> create_detector_with_engine(
    ModelSpec spec, SessionOptions session, DetectionOptions options,
    std::shared_ptr<RuntimeEngine> engine) {
    spec.task = TaskKind::detect;
    return std::make_unique<RuntimeDetector>(std::move(spec),
                                             std::move(session),
                                             std::move(options),
                                             std::move(engine));
}

}  // namespace detail

}  // namespace yolo
