#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "example_image.hpp"
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
    std::cout << "adapter: "
              << pipeline->info().model.adapter.value_or("unknown") << '\n';
    std::cout << "task: detect\n";
    std::cout << "inputs: " << pipeline->info().inputs.size()
              << ", outputs: " << pipeline->info().outputs.size() << '\n';

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
