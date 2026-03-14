#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "ppm_loader/image_ppm.hpp"
#include "support/cli.hpp"
#include "yolo/facade.hpp"

namespace
{

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

void write_bbox(std::ostream& stream, const yolo::RectF& bbox) {
    stream << '[' << bbox.x << ',' << bbox.y << ',' << bbox.width << ','
           << bbox.height << ']';
}

std::vector<std::size_t> encode_rle(const std::vector<std::uint8_t>& mask) {
    std::vector<std::size_t> runs{};
    if (mask.empty()) {
        return runs;
    }

    std::uint8_t current = 0;
    std::size_t count = 0;
    for (const std::uint8_t value : mask) {
        if (value == current) {
            ++count;
            continue;
        }

        runs.push_back(count);
        current = value;
        count = 1;
    }
    runs.push_back(count);
    return runs;
}

std::optional<yolo::TaskKind> parse_task(std::string_view value) {
    if (value == "detect") {
        return yolo::TaskKind::detect;
    }
    if (value == "classify") {
        return yolo::TaskKind::classify;
    }
    if (value == "seg") {
        return yolo::TaskKind::seg;
    }
    if (value == "pose") {
        return yolo::TaskKind::pose;
    }
    if (value == "obb") {
        return yolo::TaskKind::obb;
    }

    return std::nullopt;
}

void write_detection_result(std::ostream& stream,
                            const yolo::DetectionResult& result) {
    stream << "\"detections\":[";
    for (std::size_t i = 0; i < result.detections.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        const auto& detection = result.detections[i];
        stream << "{\"class_id\":" << detection.class_id
               << ",\"score\":" << detection.score << ",\"bbox\":";
        write_bbox(stream, detection.bbox);
        stream << '}';
    }
    stream << ']';
}

void write_classification_result(std::ostream& stream,
                                 const yolo::ClassificationResult& result) {
    stream << "\"classes\":[";
    for (std::size_t i = 0; i < result.classes.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        const auto& classification = result.classes[i];
        stream << "{\"class_id\":" << classification.class_id
               << ",\"score\":" << classification.score << '}';
    }
    stream << "],\"scores\":[";
    for (std::size_t i = 0; i < result.scores.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        stream << result.scores[i];
    }
    stream << ']';
}

void write_segmentation_result(std::ostream& stream,
                               const yolo::SegmentationResult& result) {
    stream << "\"instances\":[";
    for (std::size_t i = 0; i < result.instances.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        const auto& instance = result.instances[i];
        const auto rle = encode_rle(instance.mask.data);
        const std::size_t area = static_cast<std::size_t>(std::accumulate(
            instance.mask.data.begin(), instance.mask.data.end(), 0U));

        stream << "{\"class_id\":" << instance.class_id
               << ",\"score\":" << instance.score << ",\"bbox\":";
        write_bbox(stream, instance.bbox);
        stream << ",\"mask\":{\"size\":[" << instance.mask.size.width << ','
               << instance.mask.size.height << "],\"area\":" << area
               << ",\"rle\":[";
        for (std::size_t run_index = 0; run_index < rle.size(); ++run_index) {
            if (run_index > 0) {
                stream << ',';
            }
            stream << rle[run_index];
        }
        stream << "]}}";
    }
    stream << ']';
}

void write_point(std::ostream& stream, const yolo::Point2f& point) {
    stream << '[' << point.x << ',' << point.y << ']';
}

void write_pose_result(std::ostream& stream, const yolo::PoseResult& result) {
    stream << "\"poses\":[";
    for (std::size_t i = 0; i < result.poses.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        const auto& pose = result.poses[i];
        stream << "{\"class_id\":" << pose.class_id
               << ",\"score\":" << pose.score << ",\"bbox\":";
        write_bbox(stream, pose.bbox);
        stream << ",\"keypoints\":[";
        for (std::size_t keypoint_index = 0;
             keypoint_index < pose.keypoints.size(); ++keypoint_index) {
            if (keypoint_index > 0) {
                stream << ',';
            }
            const auto& keypoint = pose.keypoints[keypoint_index];
            stream << "{\"point\":";
            write_point(stream, keypoint.point);
            stream << ",\"score\":" << keypoint.score
                   << ",\"visible\":" << (keypoint.visible ? "true" : "false")
                   << '}';
        }
        stream << "]}";
    }
    stream << ']';
}

void write_size(std::ostream& stream, const yolo::Size2f& size) {
    stream << '[' << size.width << ',' << size.height << ']';
}

void write_obb_result(std::ostream& stream, const yolo::ObbResult& result) {
    stream << "\"boxes\":[";
    for (std::size_t i = 0; i < result.boxes.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        const auto& detection = result.boxes[i];
        stream << "{\"class_id\":" << detection.class_id
               << ",\"score\":" << detection.score << ",\"center\":";
        write_point(stream, detection.box.center);
        stream << ",\"size\":";
        write_size(stream, detection.box.size);
        stream << ",\"angle_radians\":" << detection.box.angle_radians
               << ",\"corners\":[";
        const auto corners = detection.box.corners();
        for (std::size_t corner_index = 0; corner_index < corners.size();
             ++corner_index) {
            if (corner_index > 0) {
                stream << ',';
            }
            write_point(stream, corners[corner_index]);
        }
        stream << "]}";
    }
    stream << ']';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: yolo_cpp_parity_dump "
                     "<detect|classify|seg|pose|obb> <model.onnx>"
                  << " <image1.ppm> [imageN.ppm...]\n";
        return EXIT_FAILURE;
    }

    const std::optional<yolo::TaskKind> task = parse_task(argv[1]);
    if (!task.has_value()) {
        std::cerr << "error: unsupported parity task '" << argv[1] << "'\n";
        return EXIT_FAILURE;
    }

    auto pipeline_result = yolo::create_pipeline(yolo::ModelSpec{
        .path = argv[2],
        .task = *task,
    });
    if (!pipeline_result.ok()) {
        return examples::print_error(pipeline_result.error);
    }

    const std::unique_ptr<yolo::Pipeline>& pipeline = *pipeline_result.value;

    std::ostringstream json{};
    json << "{\"task\":\"" << escape_json(argv[1]) << "\",\"images\":[";
    for (int index = 3; index < argc; ++index) {
        if (index > 3) {
            json << ',';
        }

        auto image_result = examples::load_ppm_image(argv[index]);
        if (!image_result.ok()) {
            return examples::print_error(image_result.error);
        }

        const auto image_name =
            std::filesystem::path(argv[index]).filename().string();
        json << "{\"image\":\"" << escape_json(image_name) << "\",";
        if (*task == yolo::TaskKind::detect) {
            const auto result = pipeline->detect(image_result.value->view());
            if (!result.ok()) {
                return examples::print_error(result.error);
            }
            write_detection_result(json, result);
        }
        else if (*task == yolo::TaskKind::classify) {
            const auto result = pipeline->classify(image_result.value->view());
            if (!result.ok()) {
                return examples::print_error(result.error);
            }
            write_classification_result(json, result);
        }
        else if (*task == yolo::TaskKind::seg) {
            const auto result = pipeline->segment(image_result.value->view());
            if (!result.ok()) {
                return examples::print_error(result.error);
            }
            write_segmentation_result(json, result);
        }
        else if (*task == yolo::TaskKind::pose) {
            const auto result =
                pipeline->estimate_pose(image_result.value->view());
            if (!result.ok()) {
                return examples::print_error(result.error);
            }
            write_pose_result(json, result);
        }
        else {
            const auto result =
                pipeline->detect_obb(image_result.value->view());
            if (!result.ok()) {
                return examples::print_error(result.error);
            }
            write_obb_result(json, result);
        }
        json << '}';
    }
    json << "]}\n";
    std::cout << json.str();
    return EXIT_SUCCESS;
}
