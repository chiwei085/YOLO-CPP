#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "example_image.hpp"
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
        stream << "{\"class_id\":" << detection.class_id << ",\"score\":"
               << detection.score << ",\"bbox\":";
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
        stream << "{\"class_id\":" << classification.class_id << ",\"score\":"
               << classification.score << '}';
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

        stream << "{\"class_id\":" << instance.class_id << ",\"score\":"
               << instance.score << ",\"bbox\":";
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr
            << "usage: yolo_cpp_parity_dump <detect|classify|seg> <model.onnx>"
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

    std::cout << "{\"task\":\"" << escape_json(argv[1]) << "\",\"images\":[";
    for (int index = 3; index < argc; ++index) {
        if (index > 3) {
            std::cout << ',';
        }

        auto image_result = examples::load_ppm_image(argv[index]);
        if (!image_result.ok()) {
            return examples::print_error(image_result.error);
        }

        std::cout << "{\"image\":\"" << escape_json(argv[index]) << "\",";
        if (*task == yolo::TaskKind::detect) {
            const auto result = pipeline->detect(image_result.value->view());
            if (!result.ok()) {
                return examples::print_error(result.error);
            }
            write_detection_result(std::cout, result);
        }
        else if (*task == yolo::TaskKind::classify) {
            const auto result = pipeline->classify(image_result.value->view());
            if (!result.ok()) {
                return examples::print_error(result.error);
            }
            write_classification_result(std::cout, result);
        }
        else {
            const auto result = pipeline->segment(image_result.value->view());
            if (!result.ok()) {
                return examples::print_error(result.error);
            }
            write_segmentation_result(std::cout, result);
        }
        std::cout << '}';
    }
    std::cout << "]}\n";
    return EXIT_SUCCESS;
}
