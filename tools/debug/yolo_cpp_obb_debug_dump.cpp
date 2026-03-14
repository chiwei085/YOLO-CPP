#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "ppm_loader/image_ppm.hpp"
#include "support/cli.hpp"
#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/engine.hpp"
#include "yolo/detail/image_preprocess.hpp"
#include "yolo/detail/obb_runtime.hpp"
#include "yolo/detail/task_runtime_utils.hpp"

namespace
{

using yolo::ClassId;
using yolo::ModelSpec;
using yolo::ObbOptions;
using yolo::OrientedBox;
using yolo::OrientedDetection;
using yolo::Padding2i;
using yolo::Point2f;
using yolo::PreprocessRecord;
using yolo::SessionOptions;
using yolo::Size2f;
using yolo::Size2i;
using yolo::TaskKind;
using yolo::detail::ObbCandidate;

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

void write_point(std::ostream& stream, const Point2f& point) {
    stream << '[' << point.x << ',' << point.y << ']';
}

void write_size(std::ostream& stream, const Size2f& size) {
    stream << '[' << size.width << ',' << size.height << ']';
}

void write_size_i(std::ostream& stream, const Size2i& size) {
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
    write_size_i(stream, record.source_size);
    stream << ",\"target_size\":";
    write_size_i(stream, record.target_size);
    stream << ",\"resized_size\":";
    write_size_i(stream, record.resized_size);
    stream << ",\"resize_scale\":[" << record.resize_scale.x << ','
           << record.resize_scale.y << "],\"padding\":";
    write_padding(stream, record.padding);
    stream << '}';
}

void write_corners(std::ostream& stream, const OrientedBox& box) {
    stream << '[';
    const auto corners = box.corners();
    for (std::size_t i = 0; i < corners.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        write_point(stream, corners[i]);
    }
    stream << ']';
}

void write_box(std::ostream& stream, const OrientedBox& box) {
    stream << "{\"center\":";
    write_point(stream, box.center);
    stream << ",\"size\":";
    write_size(stream, box.size);
    stream << ",\"angle_radians\":" << box.angle_radians << ",\"corners\":";
    write_corners(stream, box);
    stream << '}';
}

struct CandidateSnapshot
{
    ClassId class_id{0};
    float score{0.0F};
    OrientedBox box{};
};

void write_candidates(std::ostream& stream,
                      const std::vector<CandidateSnapshot>& candidates) {
    stream << '[';
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        stream << "{\"class_id\":" << candidates[i].class_id
               << ",\"score\":" << candidates[i].score << ",\"box\":";
        write_box(stream, candidates[i].box);
        stream << '}';
    }
    stream << ']';
}

std::vector<CandidateSnapshot> snapshot_candidates(
    const std::vector<ObbCandidate>& candidates) {
    std::vector<CandidateSnapshot> snapshot{};
    snapshot.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        snapshot.push_back(CandidateSnapshot{
            .class_id = candidate.class_id,
            .score = candidate.score,
            .box = candidate.box,
        });
    }
    return snapshot;
}

float cross(const Point2f& lhs, const Point2f& rhs) {
    return lhs.x * rhs.y - lhs.y * rhs.x;
}

Point2f operator-(const Point2f& lhs, const Point2f& rhs) {
    return Point2f{.x = lhs.x - rhs.x, .y = lhs.y - rhs.y};
}

Point2f operator+(const Point2f& lhs, const Point2f& rhs) {
    return Point2f{.x = lhs.x + rhs.x, .y = lhs.y + rhs.y};
}

Point2f operator*(const Point2f& point, float scale) {
    return Point2f{.x = point.x * scale, .y = point.y * scale};
}

bool is_inside(const Point2f& point, const Point2f& edge_start,
               const Point2f& edge_end, float winding_sign) {
    return winding_sign * cross(edge_end - edge_start, point - edge_start) >=
           -1e-6F;
}

Point2f line_intersection(const Point2f& a0, const Point2f& a1,
                          const Point2f& b0, const Point2f& b1) {
    const Point2f a = a1 - a0;
    const Point2f b = b1 - b0;
    const float denominator = cross(a, b);
    if (std::abs(denominator) <= 1e-6F) {
        return a1;
    }
    const float t = cross(b0 - a0, b) / denominator;
    return a0 + a * t;
}

std::vector<Point2f> clip_polygon(const std::vector<Point2f>& polygon,
                                  const Point2f& edge_start,
                                  const Point2f& edge_end, float winding_sign) {
    if (polygon.empty()) {
        return {};
    }

    std::vector<Point2f> output{};
    Point2f previous = polygon.back();
    bool previous_inside =
        is_inside(previous, edge_start, edge_end, winding_sign);
    for (const Point2f& current : polygon) {
        const bool current_inside =
            is_inside(current, edge_start, edge_end, winding_sign);
        if (current_inside != previous_inside) {
            output.push_back(
                line_intersection(previous, current, edge_start, edge_end));
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
    const auto left = yolo::canonicalize_oriented_box(lhs);
    const auto right = yolo::canonicalize_oriented_box(rhs);
    const auto left_corners = left.corners();
    const auto right_corners = right.corners();
    std::vector<Point2f> polygon(left_corners.begin(), left_corners.end());
    const std::vector<Point2f> right_polygon(right_corners.begin(),
                                             right_corners.end());
    const float winding_sign =
        signed_polygon_area(right_polygon) >= 0.0F ? 1.0F : -1.0F;
    for (std::size_t i = 0; i < right_corners.size(); ++i) {
        polygon = clip_polygon(polygon, right_corners[i],
                               right_corners[(i + 1) % right_corners.size()],
                               winding_sign);
        if (polygon.empty()) {
            return 0.0F;
        }
    }
    const float left_area = left.size.width * left.size.height;
    const float right_area = right.size.width * right.size.height;
    const float inter = polygon_area(polygon);
    const float uni = left_area + right_area - inter;
    return uni > 1e-6F ? inter / uni : 0.0F;
}

std::vector<ObbCandidate> canonicalize_candidates(
    const std::vector<ObbCandidate>& candidates) {
    std::vector<ObbCandidate> canonical{};
    canonical.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        canonical.push_back(ObbCandidate{
            .box = yolo::canonicalize_oriented_box(candidate.box),
            .score = candidate.score,
            .class_id = candidate.class_id,
        });
    }
    return canonical;
}

std::vector<ObbCandidate> apply_confidence_filter(
    const std::vector<ObbCandidate>& candidates, float threshold) {
    std::vector<ObbCandidate> filtered{};
    for (const auto& candidate : candidates) {
        if (candidate.score < threshold || candidate.box.size.width <= 0.0F ||
            candidate.box.size.height <= 0.0F) {
            continue;
        }
        filtered.push_back(candidate);
    }
    std::sort(filtered.begin(), filtered.end(),
              [](const ObbCandidate& lhs, const ObbCandidate& rhs) {
                  return lhs.score > rhs.score;
              });
    return filtered;
}

std::vector<ObbCandidate> apply_rotated_nms(
    const std::vector<ObbCandidate>& candidates, const ObbOptions& options) {
    std::vector<ObbCandidate> kept{};
    std::vector<bool> suppressed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        kept.push_back(candidates[i]);
        if (kept.size() >= options.max_detections) {
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
    return kept;
}

void write_final(std::ostream& stream,
                 const std::vector<OrientedDetection>& detections) {
    stream << '[';
    for (std::size_t i = 0; i < detections.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        stream << "{\"class_id\":" << detections[i].class_id
               << ",\"score\":" << detections[i].score << ",\"box\":";
        write_box(stream, detections[i].box);
        stream << '}';
    }
    stream << ']';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr
            << "usage: yolo_cpp_obb_debug_dump <model.onnx> <image.ppm>\n";
        return EXIT_FAILURE;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];

    auto binding_result = yolo::adapters::ultralytics::probe_obb_model(
        ModelSpec{.path = model_path, .task = TaskKind::obb}, SessionOptions{});
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
        yolo::detail::select_primary_input(**engine_result.value, "obb");
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

    auto spec_result =
        yolo::detail::obb_decode_spec_from_binding(*binding_result.value);
    if (!spec_result.ok()) {
        return examples::print_error(spec_result.error);
    }

    auto decoded_result = yolo::detail::decode_obb_candidates(
        *outputs_result.value, *spec_result.value);
    if (!decoded_result.ok()) {
        return examples::print_error(decoded_result.error);
    }

    const auto canonicalized = canonicalize_candidates(*decoded_result.value);
    const ObbOptions options{};
    const auto confidence_filtered =
        apply_confidence_filter(canonicalized, options.confidence_threshold);
    const auto nms_kept = apply_rotated_nms(confidence_filtered, options);
    const auto final_boxes = yolo::detail::postprocess_obb(
        *decoded_result.value, preprocess_result.value->record, options,
        binding_result.value->model);

    std::ostringstream json{};
    std::vector<int> preprocess_shape{};
    for (const auto& dim : preprocess_result.value->tensor.info.shape.dims) {
        preprocess_shape.push_back(static_cast<int>(dim.value.value_or(0)));
    }
    const auto image_name =
        std::filesystem::path(image_path).filename().string();
    json << "{\"task\":\"obb\",\"image\":\"" << escape_json(image_name)
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
            yolo::detail::copy_float_tensor_data(output, "obb_debug");
        if (!values_result.ok()) {
            return examples::print_error(values_result.error);
        }
        std::vector<int> shape{};
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
    write_candidates(json, snapshot_candidates(*decoded_result.value));
    json << "},\"canonicalized\":{\"candidate_count\":" << canonicalized.size()
         << ",\"candidates\":";
    write_candidates(json, snapshot_candidates(canonicalized));
    json << "},\"confidence_filtered\":{\"candidate_count\":"
         << confidence_filtered.size() << ",\"candidates\":";
    write_candidates(json, snapshot_candidates(confidence_filtered));
    json << "},\"nms\":{\"candidate_count\":" << nms_kept.size()
         << ",\"candidates\":";
    write_candidates(json, snapshot_candidates(nms_kept));
    json << "},\"final\":{\"boxes\":";
    write_final(json, final_boxes);
    json << "}}\n";

    std::cout << json.str();
    return EXIT_SUCCESS;
}
