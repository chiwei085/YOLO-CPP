#include "yolo/tasks/pose.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/engine.hpp"
#include "yolo/detail/image_preprocess.hpp"
#include "yolo/detail/pose_runtime.hpp"
#include "yolo/detail/task_factory.hpp"
#include "yolo/detail/task_runtime_utils.hpp"

namespace yolo
{
namespace
{

using adapters::ultralytics::AdapterBindingSpec;
using adapters::ultralytics::DetectionHeadLayout;
using adapters::ultralytics::OutputRole;
using adapters::ultralytics::PoseBindingSpec;
using adapters::ultralytics::PoseKeypointSemantic;

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

Point2f unmap_point(const Point2f& point, const PreprocessRecord& record) {
    Point2f unmapped = point;
    if (record.resize_mode == ResizeMode::letterbox) {
        unmapped.x = (unmapped.x - record.padding.left) / record.resize_scale.x;
        unmapped.y = (unmapped.y - record.padding.top) / record.resize_scale.y;
    }
    else if (record.resize_mode == ResizeMode::resize_crop && record.crop) {
        unmapped.x = (unmapped.x + record.crop->x) / record.resize_scale.x;
        unmapped.y = (unmapped.y + record.crop->y) / record.resize_scale.y;
    }
    else {
        unmapped.x /= record.resize_scale.x;
        unmapped.y /= record.resize_scale.y;
    }

    unmapped.x = std::clamp(unmapped.x, 0.0F,
                            static_cast<float>(record.source_size.width));
    unmapped.y = std::clamp(unmapped.y, 0.0F,
                            static_cast<float>(record.source_size.height));
    return unmapped;
}

std::optional<std::string> label_for(const ModelSpec& spec, ClassId class_id) {
    const std::size_t index = static_cast<std::size_t>(class_id);
    if (index < spec.labels.size()) {
        return spec.labels[index];
    }

    return std::nullopt;
}

InferenceMetadata make_metadata(const detail::RuntimeEngine& engine,
                                const detail::PreprocessedImage& preprocessed,
                                const SessionOptions& session) {
    return detail::make_task_metadata(TaskKind::pose, engine, preprocessed,
                                      session);
}

class RuntimePoseEstimator final : public PoseEstimator
{
public:
    RuntimePoseEstimator(AdapterBindingSpec binding, SessionOptions session,
                         PoseOptions options,
                         std::shared_ptr<detail::RuntimeEngine> engine)
        : binding_(std::move(binding)),
          spec_(binding_.model),
          session_(std::move(session)),
          options_(std::move(options)),
          engine_(std::move(engine)) {
        if (!engine_) {
            init_error_ =
                make_error(ErrorCode::invalid_state,
                           "Pose runtime requires a valid shared engine.",
                           ErrorContext{.component = std::string{"pose"}});
            return;
        }

        auto decode_spec_result =
            detail::pose_decode_spec_from_binding(binding_);
        if (!decode_spec_result.ok()) {
            init_error_ = std::move(decode_spec_result.error);
            return;
        }

        decode_spec_ = *decode_spec_result.value;
    }

    RuntimePoseEstimator(ModelSpec spec, SessionOptions session,
                         PoseOptions options, Error init_error)
        : spec_(std::move(spec)),
          session_(std::move(session)),
          options_(std::move(options)),
          init_error_(std::move(init_error)) {}

    const ModelSpec& model() const noexcept override { return spec_; }

    PoseResult run(const ImageView& image) const override {
        if (!init_error_.ok()) {
            return PoseResult{
                .poses = {},
                .metadata = InferenceMetadata{.task = TaskKind::pose},
                .error = init_error_,
            };
        }

        auto input_info_result = detail::select_primary_input(*engine_, "pose");
        if (!input_info_result.ok()) {
            return PoseResult{
                .poses = {},
                .metadata = InferenceMetadata{.task = TaskKind::pose},
                .error = input_info_result.error,
            };
        }

        auto preprocess_result = detail::preprocess_image(
            image, binding_.preprocess, input_info_result.value->name);
        if (!preprocess_result.ok()) {
            return PoseResult{
                .poses = {},
                .metadata = InferenceMetadata{.task = TaskKind::pose},
                .error = preprocess_result.error,
            };
        }

        detail::RawInputTensor input{
            .info = preprocess_result.value->tensor.info,
            .bytes = preprocess_result.value->tensor.bytes(),
        };
        auto outputs_result = engine_->run(input);
        if (!outputs_result.ok()) {
            return PoseResult{
                .poses = {},
                .metadata =
                    make_metadata(*engine_, *preprocess_result.value, session_),
                .error = outputs_result.error,
            };
        }

        auto decoded_result =
            detail::decode_poses(*outputs_result.value, decode_spec_);
        if (!decoded_result.ok()) {
            return PoseResult{
                .poses = {},
                .metadata =
                    make_metadata(*engine_, *preprocess_result.value, session_),
                .error = decoded_result.error,
            };
        }

        return PoseResult{
            .poses = detail::postprocess_poses(std::move(*decoded_result.value),
                                               preprocess_result.value->record,
                                               options_, spec_),
            .metadata =
                make_metadata(*engine_, *preprocess_result.value, session_),
            .error = {},
        };
    }

private:
    AdapterBindingSpec binding_{};
    ModelSpec spec_{};
    SessionOptions session_{};
    PoseOptions options_{};
    std::shared_ptr<detail::RuntimeEngine> engine_{};
    detail::PoseDecodeSpec decode_spec_{};
    Error init_error_{};
};

}  // namespace

namespace detail
{

Result<PoseDecodeSpec> pose_decode_spec_from_binding(
    const adapters::ultralytics::AdapterBindingSpec& binding) {
    if (!binding.pose.has_value()) {
        return {.error =
                    make_error(ErrorCode::invalid_state,
                               "Pose runtime requires a pose binding spec.",
                               ErrorContext{.component = std::string{"pose"}})};
    }

    if (binding.outputs.empty()) {
        return {.error = make_error(
                    ErrorCode::invalid_state,
                    "Pose runtime requires at least one bound output.",
                    ErrorContext{.component = std::string{"pose"}})};
    }

    const PoseBindingSpec& pose = *binding.pose;
    DetectionLayout layout = DetectionLayout::xywh_class_scores_last;
    switch (pose.layout) {
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

    return {.value =
                PoseDecodeSpec{
                    .output_index = output_index,
                    .layout = layout,
                    .proposal_count = pose.proposal_count,
                    .class_count = pose.class_count,
                    .keypoint_count = pose.keypoint_count,
                    .keypoint_dimension = pose.keypoint_dimension,
                    .keypoint_semantic = pose.keypoint_semantic,
                },
            .error = {}};
}

Result<std::vector<PoseCandidate>> decode_poses(const RawOutputTensors& outputs,
                                                const PoseDecodeSpec& spec) {
    if (spec.output_index >= outputs.size()) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "Pose binding points to a missing output tensor.",
                    ErrorContext{
                        .component = std::string{"pose_decoder"},
                        .expected = std::to_string(spec.output_index + 1) +
                                    " output tensors",
                        .actual =
                            std::to_string(outputs.size()) + " output tensors",
                    })};
    }

    const auto float_values_result =
        copy_float_tensor_data(outputs[spec.output_index], "pose_decoder");
    if (!float_values_result.ok()) {
        return {.error = float_values_result.error};
    }

    const std::size_t keypoint_values =
        spec.keypoint_count * spec.keypoint_dimension;
    const std::size_t stride = 4 + spec.class_count + keypoint_values;
    const std::vector<float>& values = *float_values_result.value;
    std::vector<PoseCandidate> candidates{};
    candidates.reserve(spec.proposal_count);

    if (spec.layout == DetectionLayout::xywh_class_scores_last) {
        if (values.size() < spec.proposal_count * stride) {
            return {
                .error = make_error(
                    ErrorCode::shape_mismatch,
                    "Pose output tensor payload is smaller than its decoded "
                    "shape.",
                    ErrorContext{
                        .component = std::string{"pose_decoder"},
                        .output_name = outputs[spec.output_index].info.name,
                        .expected =
                            std::to_string(spec.proposal_count * stride),
                        .actual = std::to_string(values.size()),
                    })};
        }

        for (std::size_t i = 0; i < spec.proposal_count; ++i) {
            const float* row = values.data() + i * stride;
            const float* class_begin = row + 4;
            const float* class_end = class_begin + spec.class_count;
            const auto class_it = std::max_element(class_begin, class_end);
            const float best_score = class_it == class_end ? 0.0F : *class_it;
            const ClassId class_id = class_it == class_end
                                         ? 0U
                                         : static_cast<ClassId>(std::distance(
                                               class_begin, class_it));

            std::vector<PoseKeypoint> keypoints{};
            keypoints.reserve(spec.keypoint_count);
            const float* keypoint_base = row + 4 + spec.class_count;
            for (std::size_t k = 0; k < spec.keypoint_count; ++k) {
                const float* kp = keypoint_base + k * spec.keypoint_dimension;
                const float kp_score =
                    spec.keypoint_dimension >= 3 ? kp[2] : 1.0F;
                keypoints.push_back(PoseKeypoint{
                    .score = kp_score,
                    .visible = spec.keypoint_semantic ==
                                       PoseKeypointSemantic::xyvisibility
                                   ? kp_score > 0.5F
                                   : kp_score > 0.0F,
                    .point = Point2f{.x = kp[0], .y = kp[1]},
                });
            }

            candidates.push_back(PoseCandidate{
                .bbox = xywh_to_rect(row[0], row[1], row[2], row[3]),
                .score = best_score,
                .class_id = class_id,
                .keypoints = std::move(keypoints),
            });
        }
    }
    else if (spec.layout == DetectionLayout::xywh_class_scores_first) {
        const std::size_t proposal_stride = spec.proposal_count;
        const std::size_t required_values = proposal_stride * stride;
        if (values.size() < required_values) {
            return {
                .error = make_error(
                    ErrorCode::shape_mismatch,
                    "Pose output tensor payload is smaller than its decoded "
                    "shape.",
                    ErrorContext{
                        .component = std::string{"pose_decoder"},
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

            std::vector<PoseKeypoint> keypoints{};
            keypoints.reserve(spec.keypoint_count);
            const std::size_t keypoint_base_channel = 4 + spec.class_count;
            for (std::size_t k = 0; k < spec.keypoint_count; ++k) {
                const std::size_t kp_channel =
                    keypoint_base_channel + k * spec.keypoint_dimension;
                const float kp_x = values[proposal_stride * kp_channel + i];
                const float kp_y =
                    values[proposal_stride * (kp_channel + 1) + i];
                const float kp_score =
                    spec.keypoint_dimension >= 3
                        ? values[proposal_stride * (kp_channel + 2) + i]
                        : 1.0F;
                keypoints.push_back(PoseKeypoint{
                    .score = kp_score,
                    .visible = spec.keypoint_semantic ==
                                       PoseKeypointSemantic::xyvisibility
                                   ? kp_score > 0.5F
                                   : kp_score > 0.0F,
                    .point = Point2f{.x = kp_x, .y = kp_y},
                });
            }

            candidates.push_back(PoseCandidate{
                .bbox = xywh_to_rect(values[i], values[proposal_stride + i],
                                     values[proposal_stride * 2 + i],
                                     values[proposal_stride * 3 + i]),
                .score = best_score,
                .class_id = class_id,
                .keypoints = std::move(keypoints),
            });
        }
    }
    else {
        return {.error = make_error(
                    ErrorCode::unsupported_model,
                    "Pose runtime does not support external-NMS export layout.",
                    ErrorContext{.component = std::string{"pose_decoder"}})};
    }

    return {.value = std::move(candidates), .error = {}};
}

std::vector<PoseDetection> postprocess_poses(
    std::vector<PoseCandidate> candidates, const PreprocessRecord& record,
    const PoseOptions& options, const ModelSpec& spec) {
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [&](const PoseCandidate& candidate) {
                           return candidate.score <
                                      options.confidence_threshold ||
                                  candidate.bbox.width <= 0.0F ||
                                  candidate.bbox.height <= 0.0F;
                       }),
        candidates.end());

    std::sort(candidates.begin(), candidates.end(),
              [](const PoseCandidate& lhs, const PoseCandidate& rhs) {
                  return lhs.score > rhs.score;
              });

    std::vector<PoseDetection> poses{};
    poses.reserve(std::min(options.max_detections, candidates.size()));
    std::vector<bool> suppressed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        std::vector<PoseKeypoint> keypoints{};
        keypoints.reserve(candidates[i].keypoints.size());
        for (const auto& keypoint : candidates[i].keypoints) {
            keypoints.push_back(PoseKeypoint{
                .score = keypoint.score,
                .visible = keypoint.visible,
                .point = unmap_point(keypoint.point, record),
            });
        }

        poses.push_back(PoseDetection{
            .bbox = unmap_rect(candidates[i].bbox, record),
            .score = candidates[i].score,
            .class_id = candidates[i].class_id,
            .label = label_for(spec, candidates[i].class_id),
            .keypoints = std::move(keypoints),
        });
        if (poses.size() >= options.max_detections) {
            break;
        }

        for (std::size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }

            if (intersection_over_union(candidates[i].bbox,
                                        candidates[j].bbox) >=
                options.nms_iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return poses;
}

std::unique_ptr<PoseEstimator> create_pose_estimator_with_engine(
    AdapterBindingSpec binding, SessionOptions session, PoseOptions options,
    std::shared_ptr<RuntimeEngine> engine) {
    binding.model.task = TaskKind::pose;
    return std::make_unique<RuntimePoseEstimator>(
        std::move(binding), std::move(session), std::move(options),
        std::move(engine));
}

}  // namespace detail

std::unique_ptr<PoseEstimator> create_pose_estimator(ModelSpec spec,
                                                     SessionOptions session,
                                                     PoseOptions options) {
    spec.task = TaskKind::pose;
    auto binding_result =
        adapters::ultralytics::probe_pose_model(spec, session);
    if (!binding_result.ok()) {
        return std::make_unique<RuntimePoseEstimator>(
            std::move(spec), std::move(session), std::move(options),
            std::move(binding_result.error));
    }

    auto engine_result =
        detail::RuntimeEngine::create(binding_result.value->model, session);
    if (!engine_result.ok()) {
        return std::make_unique<RuntimePoseEstimator>(
            binding_result.value->model, std::move(session), std::move(options),
            std::move(engine_result.error));
    }

    return std::make_unique<RuntimePoseEstimator>(
        std::move(*binding_result.value), std::move(session),
        std::move(options),
        std::shared_ptr<detail::RuntimeEngine>(
            std::move(*engine_result.value)));
}

}  // namespace yolo
