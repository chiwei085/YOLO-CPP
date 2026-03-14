#include "yolo/tasks/obb.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/engine.hpp"
#include "yolo/detail/image_preprocess.hpp"
#include "yolo/detail/obb_runtime.hpp"
#include "yolo/detail/task_factory.hpp"
#include "yolo/detail/task_runtime_utils.hpp"

namespace yolo
{
namespace
{

using adapters::ultralytics::AdapterBindingSpec;
using adapters::ultralytics::DetectionHeadLayout;
using adapters::ultralytics::ObbBindingSpec;
using adapters::ultralytics::OutputRole;

constexpr float kGeometryEpsilon = 1e-6F;

Point2f operator+(const Point2f& lhs, const Point2f& rhs) {
    return Point2f{.x = lhs.x + rhs.x, .y = lhs.y + rhs.y};
}

Point2f operator-(const Point2f& lhs, const Point2f& rhs) {
    return Point2f{.x = lhs.x - rhs.x, .y = lhs.y - rhs.y};
}

Point2f operator*(const Point2f& point, float scale) {
    return Point2f{.x = point.x * scale, .y = point.y * scale};
}

float cross(const Point2f& lhs, const Point2f& rhs) {
    return lhs.x * rhs.y - lhs.y * rhs.x;
}

float dot(const Point2f& lhs, const Point2f& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y;
}

float norm(const Point2f& point) {
    return std::sqrt(dot(point, point));
}

bool is_inside_edge(const Point2f& point, const Point2f& edge_start,
                    const Point2f& edge_end, float winding_sign) {
    return winding_sign * cross(edge_end - edge_start, point - edge_start) >=
           -kGeometryEpsilon;
}

Point2f line_intersection(const Point2f& segment_start,
                          const Point2f& segment_end, const Point2f& clip_start,
                          const Point2f& clip_end) {
    const Point2f segment = segment_end - segment_start;
    const Point2f clip = clip_end - clip_start;
    const float denominator = cross(segment, clip);
    if (std::abs(denominator) <= kGeometryEpsilon) {
        return segment_end;
    }

    const float t = cross(clip_start - segment_start, clip) / denominator;
    return segment_start + segment * t;
}

std::vector<Point2f> clip_polygon_against_edge(
    const std::vector<Point2f>& polygon, const Point2f& clip_start,
    const Point2f& clip_end, float winding_sign) {
    if (polygon.empty()) {
        return {};
    }

    std::vector<Point2f> output{};
    output.reserve(polygon.size() + 1);
    Point2f previous = polygon.back();
    bool previous_inside =
        is_inside_edge(previous, clip_start, clip_end, winding_sign);
    for (const Point2f& current : polygon) {
        const bool current_inside =
            is_inside_edge(current, clip_start, clip_end, winding_sign);
        if (current_inside != previous_inside) {
            output.push_back(
                line_intersection(previous, current, clip_start, clip_end));
        }
        if (current_inside) {
            output.push_back(current);
        }
        previous = current;
        previous_inside = current_inside;
    }

    return output;
}

float signed_polygon_area(const std::vector<Point2f>& polygon) {
    if (polygon.size() < 3) {
        return 0.0F;
    }

    float area = 0.0F;
    Point2f previous = polygon.back();
    for (const Point2f& current : polygon) {
        area += cross(previous, current);
        previous = current;
    }
    return area * 0.5F;
}

float polygon_area(const std::vector<Point2f>& polygon) {
    return std::abs(signed_polygon_area(polygon));
}

float rotated_iou(const OrientedBox& lhs, const OrientedBox& rhs) {
    const OrientedBox left = canonicalize_oriented_box(lhs);
    const OrientedBox right = canonicalize_oriented_box(rhs);
    const std::array<Point2f, 4> left_corners = left.corners();
    const std::array<Point2f, 4> right_corners = right.corners();
    std::vector<Point2f> intersection(left_corners.begin(), left_corners.end());
    const std::vector<Point2f> right_polygon(right_corners.begin(),
                                             right_corners.end());
    const float winding_sign =
        signed_polygon_area(right_polygon) >= 0.0F ? 1.0F : -1.0F;
    for (std::size_t i = 0; i < right_corners.size(); ++i) {
        const Point2f& edge_start = right_corners[i];
        const Point2f& edge_end = right_corners[(i + 1) % right_corners.size()];
        intersection = clip_polygon_against_edge(intersection, edge_start,
                                                 edge_end, winding_sign);
        if (intersection.empty()) {
            return 0.0F;
        }
    }

    const float left_area = left.size.width * left.size.height;
    const float right_area = right.size.width * right.size.height;
    const float intersection_area = polygon_area(intersection);
    const float union_area = left_area + right_area - intersection_area;
    return union_area > kGeometryEpsilon ? intersection_area / union_area
                                         : 0.0F;
}

Point2f unmap_point(const Point2f& point, const PreprocessRecord& record);

OrientedBox restore_box_to_source_space(const OrientedBox& box,
                                        const PreprocessRecord& record) {
    OrientedBox restored = box;
    restored.center = unmap_point(box.center, record);

    const float scale_x = record.resize_scale.x;
    const float scale_y = record.resize_scale.y;
    if (scale_x <= 0.0F || scale_y <= 0.0F) {
        restored.size = {0.0F, 0.0F};
        restored.angle_radians = 0.0F;
        return restored;
    }

    if (std::abs(scale_x - scale_y) <= kGeometryEpsilon) {
        restored.size = Size2f{
            .width = std::max(0.0F, box.size.width / scale_x),
            .height = std::max(0.0F, box.size.height / scale_y),
        };
        return canonicalize_oriented_box(restored);
    }

    // Non-uniform scaling cannot preserve a perfect rotated rectangle exactly.
    // We approximate by transforming the box axes independently, then
    // re-canonicalize into the closest xywhr representation.
    const float cos_value = std::cos(box.angle_radians);
    const float sin_value = std::sin(box.angle_radians);
    const Point2f width_axis{
        .x = (box.size.width * 0.5F * cos_value) / scale_x,
        .y = (box.size.width * 0.5F * sin_value) / scale_y,
    };
    const Point2f height_axis{
        .x = (-box.size.height * 0.5F * sin_value) / scale_x,
        .y = (box.size.height * 0.5F * cos_value) / scale_y,
    };
    restored.size = Size2f{
        .width = std::max(0.0F, norm(width_axis) * 2.0F),
        .height = std::max(0.0F, norm(height_axis) * 2.0F),
    };
    restored.angle_radians = std::atan2(width_axis.y, width_axis.x);
    return canonicalize_oriented_box(restored);
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
    return detail::make_task_metadata(TaskKind::obb, engine, preprocessed,
                                      session);
}

class RuntimeObbDetector final : public OrientedDetector
{
public:
    RuntimeObbDetector(AdapterBindingSpec binding, SessionOptions session,
                       ObbOptions options,
                       std::shared_ptr<detail::RuntimeEngine> engine)
        : binding_(std::move(binding)),
          spec_(binding_.model),
          session_(std::move(session)),
          options_(std::move(options)),
          engine_(std::move(engine)) {
        if (!engine_) {
            init_error_ =
                make_error(ErrorCode::invalid_state,
                           "OBB runtime requires a valid shared engine.",
                           ErrorContext{.component = std::string{"obb"}});
            return;
        }

        auto decode_spec_result =
            detail::obb_decode_spec_from_binding(binding_);
        if (!decode_spec_result.ok()) {
            init_error_ = std::move(decode_spec_result.error);
            return;
        }

        decode_spec_ = *decode_spec_result.value;
    }

    RuntimeObbDetector(ModelSpec spec, SessionOptions session,
                       ObbOptions options, Error init_error)
        : spec_(std::move(spec)),
          session_(std::move(session)),
          options_(std::move(options)),
          init_error_(std::move(init_error)) {}

    const ModelSpec& model() const noexcept override { return spec_; }

    ObbResult run(const ImageView& image) const override {
        if (!init_error_.ok()) {
            return ObbResult{
                .boxes = {},
                .metadata = InferenceMetadata{.task = TaskKind::obb},
                .error = init_error_,
            };
        }

        auto input_info_result = detail::select_primary_input(*engine_, "obb");
        if (!input_info_result.ok()) {
            return ObbResult{
                .boxes = {},
                .metadata = InferenceMetadata{.task = TaskKind::obb},
                .error = input_info_result.error,
            };
        }

        auto preprocess_result = detail::preprocess_image(
            image, binding_.preprocess, input_info_result.value->name);
        if (!preprocess_result.ok()) {
            return ObbResult{
                .boxes = {},
                .metadata = InferenceMetadata{.task = TaskKind::obb},
                .error = preprocess_result.error,
            };
        }

        detail::RawInputTensor input{
            .info = preprocess_result.value->tensor.info,
            .bytes = preprocess_result.value->tensor.bytes(),
        };
        auto outputs_result = engine_->run(input);
        if (!outputs_result.ok()) {
            return ObbResult{
                .boxes = {},
                .metadata =
                    make_metadata(*engine_, *preprocess_result.value, session_),
                .error = outputs_result.error,
            };
        }

        auto decoded_result =
            detail::decode_obb_candidates(*outputs_result.value, decode_spec_);
        if (!decoded_result.ok()) {
            return ObbResult{
                .boxes = {},
                .metadata =
                    make_metadata(*engine_, *preprocess_result.value, session_),
                .error = decoded_result.error,
            };
        }

        return ObbResult{
            .boxes = detail::postprocess_obb(std::move(*decoded_result.value),
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
    ObbOptions options_{};
    std::shared_ptr<detail::RuntimeEngine> engine_{};
    detail::ObbDecodeSpec decode_spec_{};
    Error init_error_{};
};

}  // namespace

namespace detail
{

Result<ObbDecodeSpec> obb_decode_spec_from_binding(
    const adapters::ultralytics::AdapterBindingSpec& binding) {
    if (!binding.obb.has_value()) {
        return {.error =
                    make_error(ErrorCode::invalid_state,
                               "OBB runtime requires an OBB binding spec.",
                               ErrorContext{.component = std::string{"obb"}})};
    }

    if (binding.outputs.empty()) {
        return {.error = make_error(
                    ErrorCode::invalid_state,
                    "OBB runtime requires at least one bound output.",
                    ErrorContext{.component = std::string{"obb"}})};
    }

    const ObbBindingSpec& obb = *binding.obb;
    DetectionLayout layout = DetectionLayout::xywh_class_scores_last;
    switch (obb.layout) {
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
                ObbDecodeSpec{
                    .output_index = output_index,
                    .layout = layout,
                    .proposal_count = obb.proposal_count,
                    .class_count = obb.class_count,
                    .box_coordinate_count = obb.box_coordinate_count,
                    .class_channel_offset = obb.class_channel_offset,
                    .angle_channel_offset = obb.angle_channel_offset,
                    .angle_is_radians = obb.angle_is_radians,
                },
            .error = {}};
}

Result<std::vector<ObbCandidate>> decode_obb_candidates(
    const RawOutputTensors& outputs, const ObbDecodeSpec& spec) {
    if (spec.output_index >= outputs.size()) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "OBB binding points to a missing output tensor.",
                    ErrorContext{
                        .component = std::string{"obb_decoder"},
                        .expected = std::to_string(spec.output_index + 1) +
                                    " output tensors",
                        .actual =
                            std::to_string(outputs.size()) + " output tensors",
                    })};
    }

    const auto float_values_result =
        copy_float_tensor_data(outputs[spec.output_index], "obb_decoder");
    if (!float_values_result.ok()) {
        return {.error = float_values_result.error};
    }

    const std::size_t stride = spec.box_coordinate_count + spec.class_count + 1;
    const std::vector<float>& values = *float_values_result.value;
    std::vector<ObbCandidate> candidates{};
    candidates.reserve(spec.proposal_count);

    if (spec.layout == DetectionLayout::xywh_class_scores_last) {
        if (values.size() < spec.proposal_count * stride) {
            return {.error = make_error(
                        ErrorCode::shape_mismatch,
                        "OBB output tensor payload is smaller than its decoded "
                        "shape.",
                        ErrorContext{
                            .component = std::string{"obb_decoder"},
                            .output_name = outputs[spec.output_index].info.name,
                            .expected =
                                std::to_string(spec.proposal_count * stride),
                            .actual = std::to_string(values.size()),
                        })};
        }

        for (std::size_t i = 0; i < spec.proposal_count; ++i) {
            const float* row = values.data() + i * stride;
            const float* class_begin = row + spec.class_channel_offset;
            const float* class_end = class_begin + spec.class_count;
            const auto class_it = std::max_element(class_begin, class_end);
            const float best_score = class_it == class_end ? 0.0F : *class_it;
            const ClassId class_id = class_it == class_end
                                         ? 0U
                                         : static_cast<ClassId>(std::distance(
                                               class_begin, class_it));

            const float angle =
                spec.angle_is_radians
                    ? row[spec.angle_channel_offset]
                    : row[spec.angle_channel_offset] * kObbPi / 180.0F;
            candidates.push_back(ObbCandidate{
                .box =
                    OrientedBox{
                        .center = Point2f{.x = row[0], .y = row[1]},
                        .size = Size2f{.width = row[2], .height = row[3]},
                        .angle_radians = angle,
                    },
                .score = best_score,
                .class_id = class_id,
            });
        }
    }
    else if (spec.layout == DetectionLayout::xywh_class_scores_first) {
        const std::size_t proposal_stride = spec.proposal_count;
        const std::size_t required_values = proposal_stride * stride;
        if (values.size() < required_values) {
            return {.error = make_error(
                        ErrorCode::shape_mismatch,
                        "OBB output tensor payload is smaller than its decoded "
                        "shape.",
                        ErrorContext{
                            .component = std::string{"obb_decoder"},
                            .output_name = outputs[spec.output_index].info.name,
                            .expected = std::to_string(required_values),
                            .actual = std::to_string(values.size()),
                        })};
        }

        for (std::size_t i = 0; i < spec.proposal_count; ++i) {
            float best_score = 0.0F;
            ClassId class_id = 0;
            for (std::size_t c = 0; c < spec.class_count; ++c) {
                const float score =
                    values[proposal_stride * (spec.class_channel_offset + c) +
                           i];
                if (score > best_score) {
                    best_score = score;
                    class_id = static_cast<ClassId>(c);
                }
            }

            const float angle =
                spec.angle_is_radians
                    ? values[proposal_stride * spec.angle_channel_offset + i]
                    : values[proposal_stride * spec.angle_channel_offset + i] *
                          kObbPi / 180.0F;
            candidates.push_back(ObbCandidate{
                .box =
                    OrientedBox{
                        .center = Point2f{.x = values[i],
                                          .y = values[proposal_stride + i]},
                        .size =
                            Size2f{.width = values[proposal_stride * 2 + i],
                                   .height = values[proposal_stride * 3 + i]},
                        .angle_radians = angle,
                    },
                .score = best_score,
                .class_id = class_id,
            });
        }
    }
    else {
        return {.error = make_error(
                    ErrorCode::unsupported_model,
                    "OBB runtime does not support external-NMS export layout.",
                    ErrorContext{.component = std::string{"obb_decoder"}})};
    }

    return {.value = std::move(candidates), .error = {}};
}

std::vector<OrientedDetection> postprocess_obb(
    std::vector<ObbCandidate> candidates, const PreprocessRecord& record,
    const ObbOptions& options, const ModelSpec& spec) {
    for (auto& candidate : candidates) {
        candidate.box = canonicalize_oriented_box(candidate.box);
    }

    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [&](const ObbCandidate& candidate) {
                           return candidate.score <
                                      options.confidence_threshold ||
                                  candidate.box.size.width <= 0.0F ||
                                  candidate.box.size.height <= 0.0F;
                       }),
        candidates.end());

    std::sort(candidates.begin(), candidates.end(),
              [](const ObbCandidate& lhs, const ObbCandidate& rhs) {
                  return lhs.score > rhs.score;
              });

    std::vector<OrientedDetection> boxes{};
    boxes.reserve(std::min(options.max_detections, candidates.size()));
    std::vector<bool> suppressed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        boxes.push_back(OrientedDetection{
            .box = restore_box_to_source_space(candidates[i].box, record),
            .score = candidates[i].score,
            .class_id = candidates[i].class_id,
            .label = label_for(spec, candidates[i].class_id),
        });
        if (boxes.size() >= options.max_detections) {
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

            if (rotated_iou(candidates[i].box, candidates[j].box) >=
                options.nms_iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return boxes;
}

std::unique_ptr<OrientedDetector> create_obb_detector_with_engine(
    AdapterBindingSpec binding, SessionOptions session, ObbOptions options,
    std::shared_ptr<RuntimeEngine> engine) {
    binding.model.task = TaskKind::obb;
    return std::make_unique<RuntimeObbDetector>(
        std::move(binding), std::move(session), std::move(options),
        std::move(engine));
}

}  // namespace detail

std::unique_ptr<OrientedDetector> create_obb_detector(ModelSpec spec,
                                                      SessionOptions session,
                                                      ObbOptions options) {
    spec.task = TaskKind::obb;
    auto binding_result = adapters::ultralytics::probe_obb_model(spec, session);
    if (!binding_result.ok()) {
        return std::make_unique<RuntimeObbDetector>(
            std::move(spec), std::move(session), std::move(options),
            std::move(binding_result.error));
    }

    auto engine_result =
        detail::RuntimeEngine::create(binding_result.value->model, session);
    if (!engine_result.ok()) {
        return std::make_unique<RuntimeObbDetector>(
            binding_result.value->model, std::move(session), std::move(options),
            std::move(engine_result.error));
    }

    return std::make_unique<RuntimeObbDetector>(
        std::move(*binding_result.value), std::move(session),
        std::move(options),
        std::shared_ptr<detail::RuntimeEngine>(
            std::move(*engine_result.value)));
}

}  // namespace yolo
