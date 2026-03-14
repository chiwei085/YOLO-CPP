#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "ppm_loader/image_ppm.hpp"
#include "support/cli.hpp"
#include "yolo/facade.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: detect_image <model.onnx> <image.ppm>\n";
        return 1;
    }

    auto image_result = examples::load_ppm_image(argv[2]);
    if (!image_result.ok()) {
        return examples::print_error(image_result.error);
    }

    auto pipeline_result = yolo::create_pipeline(yolo::ModelSpec{
        .path = argv[1], .model_name = std::string{"detect_image"}});
    if (!pipeline_result.ok()) {
        return examples::print_error(pipeline_result.error);
    }

    const std::unique_ptr<yolo::Pipeline>& pipeline = *pipeline_result.value;
    const auto& info = pipeline->info();
    examples::print_pipeline_summary(info.model.adapter.value_or("unknown"),
                                     "detect", info.inputs.size(),
                                     info.outputs.size());

    const yolo::DetectionResult result =
        pipeline->detect(image_result.value->view());
    if (!result.ok()) {
        return examples::print_error(result.error);
    }

    std::cout << "detections: " << result.detections.size() << '\n';
    const std::size_t preview =
        std::min<std::size_t>(result.detections.size(), 5);
    for (std::size_t i = 0; i < preview; ++i) {
        const auto& detection = result.detections[i];
        std::cout << i << ": class=" << detection.class_id
                  << " score=" << detection.score << " bbox=("
                  << detection.bbox.x << ", " << detection.bbox.y << ", "
                  << detection.bbox.width << ", " << detection.bbox.height
                  << ")\n";
    }

    return EXIT_SUCCESS;
}
