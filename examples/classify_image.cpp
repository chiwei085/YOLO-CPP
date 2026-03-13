#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "example_image.hpp"
#include "yolo/facade.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: classify_image <model.onnx> <image.ppm>\n";
        return 1;
    }

    auto image_result = examples::load_ppm_image(argv[2]);
    if (!image_result.ok()) {
        return examples::print_error(image_result.error);
    }

    auto pipeline_result = yolo::create_pipeline(yolo::ModelSpec{
        .path = argv[1],
        .task = yolo::TaskKind::classify,
        .model_name = std::string{"classify_image"},
    });
    if (!pipeline_result.ok()) {
        return examples::print_error(pipeline_result.error);
    }

    const std::unique_ptr<yolo::Pipeline>& pipeline = *pipeline_result.value;
    std::cout << "adapter: "
              << pipeline->info().model.adapter.value_or("unknown") << '\n';
    std::cout << "task: classify\n";
    std::cout << "inputs: " << pipeline->info().inputs.size()
              << ", outputs: " << pipeline->info().outputs.size() << '\n';

    const yolo::ClassificationResult result =
        pipeline->classify(image_result.value->view());
    if (!result.ok()) {
        return examples::print_error(result.error);
    }

    std::cout << "top classes: " << result.classes.size() << '\n';
    const std::size_t preview = std::min<std::size_t>(result.classes.size(), 5);
    for (std::size_t i = 0; i < preview; ++i) {
        const auto& classification = result.classes[i];
        std::cout << i << ": class=" << classification.class_id
                  << " score=" << classification.score;
        if (classification.label.has_value()) {
            std::cout << " label=" << *classification.label;
        }
        std::cout << '\n';
    }

    return EXIT_SUCCESS;
}
