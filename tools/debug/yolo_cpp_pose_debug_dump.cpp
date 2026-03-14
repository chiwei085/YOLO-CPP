#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "example_image.hpp"
#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/engine.hpp"
#include "yolo/detail/image_preprocess.hpp"
#include "yolo/detail/pose_runtime.hpp"
#include "yolo/detail/task_runtime_utils.hpp"

namespace
{

using yolo::ClassId;
using yolo::ModelSpec;
using yolo::Padding2i;
using yolo::Point2f;
using yolo::PoseDetection;
using yolo::PoseKeypoint;
using yolo::PreprocessRecord;
using yolo::RectF;
using yolo::ResizeMode;
using yolo::SessionOptions;
using yolo::Size2i;
using yolo::TaskKind;
using yolo::adapters::ultralytics::AdapterBindingSpec;
using yolo::detail::PoseCandidate;

std::string escape_json(std::string_view value) {
    std::string escaped{};
    escaped.reserve(value.size() + 8);
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            default:
                escaped.push_back(ch);
                break;
        }
    }

    return escaped;
}

void write_float_list(std::ostream& stream, const std::vector<float>& values) {
    stream << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        stream << values[i];
    }
    stream << ']';
}

void write_int_list(std::ostream& stream, const std::vector<int>& values) {
    stream << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        stream << values[i];
    }
    stream << ']';
}

void write_size(std::ostream& stream, const Size2i& size) {
    stream << '[' << size.width << ',' << size.height << ']';
}

void write_padding(std::ostream& stream, const Padding2i& padding) {
    stream << "{\"left\":" << padding.left << ",\"top\":" << padding.top
           << ",\"right\":" << padding.right << ",\"bottom\":" << padding.bottom
           << '}';
}

void write_preprocess_record(std::ostream& stream,
                             const PreprocessRecord& record) {
    stream << "{\"source_size\":";
    write_size(stream, record.source_size);
    stream << ",\"target_size\":";
    write_size(stream, record.target_size);
    stream << ",\"resized_size\":";
    write_size(stream, record.resized_size);
    stream << ",\"resize_scale\":[" << record.resize_scale.x << ','
           << record.resize_scale.y << "],\"padding\":";
    write_padding(stream, record.padding);
    stream << '}';
}

void write_point(std::ostream& stream, const Point2f& point) {
    stream << '[' << point.x << ',' << point.y << ']';
}

void write_bbox_xywh(std::ostream& stream, const RectF& bbox) {
    stream << '[' << bbox.x << ',' << bbox.y << ',' << bbox.width << ','
           << bbox.height << ']';
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
        unmapped.x = (unmapped.x - static_cast<float>(record.padding.left)) /
                     record.resize_scale.x;
        unmapped.y = (unmapped.y - static_cast<float>(record.padding.top)) /
                     record.resize_scale.y;
        unmapped.width /= record.resize_scale.x;
        unmapped.height /= record.resize_scale.y;
    }
    else if (record.resize_mode == ResizeMode::resize_crop && record.crop) {
        unmapped.x = (unmapped.x + static_cast<float>(record.crop->x)) /
                     record.resize_scale.x;
        unmapped.y = (unmapped.y + static_cast<float>(record.crop->y)) /
                     record.resize_scale.y;
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
    Point2f restored = point;
    if (record.resize_mode == ResizeMode::letterbox) {
        restored.x = (restored.x - static_cast<float>(record.padding.left)) /
                     record.resize_scale.x;
        restored.y = (restored.y - static_cast<float>(record.padding.top)) /
                     record.resize_scale.y;
    }
    else if (record.resize_mode == ResizeMode::resize_crop && record.crop) {
        restored.x = (restored.x + static_cast<float>(record.crop->x)) /
                     record.resize_scale.x;
        restored.y = (restored.y + static_cast<float>(record.crop->y)) /
                     record.resize_scale.y;
    }
    else {
        restored.x /= record.resize_scale.x;
        restored.y /= record.resize_scale.y;
    }

    restored.x = std::clamp(restored.x, 0.0F,
                            static_cast<float>(record.source_size.width));
    restored.y = std::clamp(restored.y, 0.0F,
                            static_cast<float>(record.source_size.height));
    return restored;
}

struct CandidateSnapshot
{
    RectF bbox{};
    RectF source_bbox{};
    float score{0.0F};
    ClassId class_id{0};
    std::vector<PoseKeypoint> keypoints{};
    std::vector<PoseKeypoint> source_keypoints{};
};

std::vector<CandidateSnapshot> snapshot_candidates(
    const std::vector<PoseCandidate>& candidates,
    const PreprocessRecord& record) {
    std::vector<CandidateSnapshot> snapshots{};
    snapshots.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        std::vector<PoseKeypoint> source_keypoints{};
        source_keypoints.reserve(candidate.keypoints.size());
        for (const auto& keypoint : candidate.keypoints) {
            source_keypoints.push_back(PoseKeypoint{
                .score = keypoint.score,
                .visible = keypoint.visible,
                .point = unmap_point(keypoint.point, record),
            });
        }

        snapshots.push_back(CandidateSnapshot{
            .bbox = candidate.bbox,
            .source_bbox = unmap_rect(candidate.bbox, record),
            .score = candidate.score,
            .class_id = candidate.class_id,
            .keypoints = candidate.keypoints,
            .source_keypoints = std::move(source_keypoints),
        });
    }
    return snapshots;
}

std::vector<PoseCandidate> apply_confidence_filter(
    const std::vector<PoseCandidate>& input, float threshold) {
    std::vector<PoseCandidate> filtered{};
    filtered.reserve(input.size());
    for (const auto& candidate : input) {
        if (candidate.score < threshold || candidate.bbox.width <= 0.0F ||
            candidate.bbox.height <= 0.0F) {
            continue;
        }
        filtered.push_back(candidate);
    }

    std::sort(filtered.begin(), filtered.end(),
              [](const PoseCandidate& lhs, const PoseCandidate& rhs) {
                  return lhs.score > rhs.score;
              });
    return filtered;
}

std::vector<PoseCandidate> apply_nms(
    const std::vector<PoseCandidate>& candidates, float iou_threshold,
    std::size_t max_detections) {
    std::vector<PoseCandidate> kept{};
    kept.reserve(std::min(max_detections, candidates.size()));
    std::vector<bool> suppressed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        kept.push_back(candidates[i]);
        if (kept.size() >= max_detections) {
            break;
        }

        for (std::size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }

            if (intersection_over_union(candidates[i].bbox,
                                        candidates[j].bbox) >= iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return kept;
}

void write_keypoints(std::ostream& stream,
                     const std::vector<PoseKeypoint>& keypoints) {
    stream << '[';
    for (std::size_t i = 0; i < keypoints.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        const auto& keypoint = keypoints[i];
        stream << "{\"point\":";
        write_point(stream, keypoint.point);
        stream << ",\"score\":" << keypoint.score
               << ",\"visible\":" << (keypoint.visible ? "true" : "false")
               << '}';
    }
    stream << ']';
}

void write_candidate_list(std::ostream& stream,
                          const std::vector<CandidateSnapshot>& candidates) {
    stream << '[';
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        const auto& candidate = candidates[i];
        stream << "{\"class_id\":" << candidate.class_id
               << ",\"score\":" << candidate.score << ",\"bbox_xywh\":";
        write_bbox_xywh(stream, candidate.bbox);
        stream << ",\"source_bbox_xywh\":";
        write_bbox_xywh(stream, candidate.source_bbox);
        stream << ",\"keypoints\":";
        write_keypoints(stream, candidate.keypoints);
        stream << ",\"source_keypoints\":";
        write_keypoints(stream, candidate.source_keypoints);
        stream << '}';
    }
    stream << ']';
}

void write_final_poses(std::ostream& stream,
                       const std::vector<PoseDetection>& poses) {
    stream << '[';
    for (std::size_t i = 0; i < poses.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        const auto& pose = poses[i];
        stream << "{\"class_id\":" << pose.class_id
               << ",\"score\":" << pose.score << ",\"bbox_xywh\":";
        write_bbox_xywh(stream, pose.bbox);
        stream << ",\"keypoints\":";
        write_keypoints(stream, pose.keypoints);
        stream << '}';
    }
    stream << ']';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr
            << "usage: yolo_cpp_pose_debug_dump <model.onnx> <image.ppm>\n";
        return EXIT_FAILURE;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];

    auto binding_result = yolo::adapters::ultralytics::probe_pose_model(
        ModelSpec{.path = model_path, .task = TaskKind::pose},
        SessionOptions{});
    if (!binding_result.ok()) {
        return examples::print_error(binding_result.error);
    }

    auto engine_result = yolo::detail::RuntimeEngine::create(
        binding_result.value->model, SessionOptions{});
    if (!engine_result.ok()) {
        return examples::print_error(engine_result.error);
    }

    auto image_result = examples::load_ppm_image(image_path);
    if (!image_result.ok()) {
        return examples::print_error(image_result.error);
    }

    auto input_info_result =
        yolo::detail::select_primary_input(**engine_result.value, "pose");
    if (!input_info_result.ok()) {
        return examples::print_error(input_info_result.error);
    }

    auto preprocess_result = yolo::detail::preprocess_image(
        image_result.value->view(), binding_result.value->preprocess,
        input_info_result.value->name);
    if (!preprocess_result.ok()) {
        return examples::print_error(preprocess_result.error);
    }

    yolo::detail::RawInputTensor input{
        .info = preprocess_result.value->tensor.info,
        .bytes = preprocess_result.value->tensor.bytes(),
    };
    auto outputs_result = (**engine_result.value).run(input);
    if (!outputs_result.ok()) {
        return examples::print_error(outputs_result.error);
    }

    auto decode_spec_result =
        yolo::detail::pose_decode_spec_from_binding(*binding_result.value);
    if (!decode_spec_result.ok()) {
        return examples::print_error(decode_spec_result.error);
    }

    auto decoded_result = yolo::detail::decode_poses(*outputs_result.value,
                                                     *decode_spec_result.value);
    if (!decoded_result.ok()) {
        return examples::print_error(decoded_result.error);
    }

    yolo::PoseOptions options{};
    const auto confidence_filtered = apply_confidence_filter(
        *decoded_result.value, options.confidence_threshold);
    const auto nms_kept = apply_nms(
        confidence_filtered, options.nms_iou_threshold, options.max_detections);
    const auto final_poses = yolo::detail::postprocess_poses(
        std::move(*decoded_result.value), preprocess_result.value->record,
        options, binding_result.value->model);

    std::ostringstream json{};
    std::vector<int> preprocess_shape{};
    preprocess_shape.reserve(
        preprocess_result.value->tensor.info.shape.dims.size());
    for (const auto& dim : preprocess_result.value->tensor.info.shape.dims) {
        preprocess_shape.push_back(static_cast<int>(dim.value.value_or(0)));
    }

    const auto image_name =
        std::filesystem::path(image_path).filename().string();
    json << "{\"task\":\"pose\",\"image\":\"" << escape_json(image_name)
         << "\",\"preprocess\":{\"tensor_shape\":";
    write_int_list(json, preprocess_shape);
    json << ",\"tensor_values\":";
    write_float_list(json, preprocess_result.value->tensor.values);
    json << ",\"record\":";
    write_preprocess_record(json, preprocess_result.value->record);
    json << "},\"raw_outputs\":[";
    for (std::size_t i = 0; i < outputs_result.value->size(); ++i) {
        if (i > 0) {
            json << ',';
        }
        const auto& output = (*outputs_result.value)[i];
        const auto values_result =
            yolo::detail::copy_float_tensor_data(output, "pose_debug");
        if (!values_result.ok()) {
            return examples::print_error(values_result.error);
        }

        std::vector<int> shape{};
        shape.reserve(output.info.shape.dims.size());
        for (const auto& dim : output.info.shape.dims) {
            shape.push_back(static_cast<int>(dim.value.value_or(0)));
        }

        json << "{\"name\":\"" << escape_json(output.info.name)
             << "\",\"shape\":";
        write_int_list(json, shape);
        json << ",\"values\":";
        write_float_list(json, *values_result.value);
        json << '}';
    }
    json << "],\"decoded\":{\"candidate_count\":"
         << decoded_result.value->size() << ",\"candidates\":";
    write_candidate_list(json,
                         snapshot_candidates(*decoded_result.value,
                                             preprocess_result.value->record));
    json << "},\"confidence_filtered\":{\"candidate_count\":"
         << confidence_filtered.size() << ",\"candidates\":";
    write_candidate_list(json,
                         snapshot_candidates(confidence_filtered,
                                             preprocess_result.value->record));
    json << "},\"nms\":{\"candidate_count\":" << nms_kept.size()
         << ",\"candidates\":";
    write_candidate_list(
        json, snapshot_candidates(nms_kept, preprocess_result.value->record));
    json << "},\"final\":{\"poses\":";
    write_final_poses(json, final_poses);
    json << "}}\n";

    std::cout << json.str();
    return EXIT_SUCCESS;
}
