#include <cstdlib>
#include <iostream>
#include <memory>

#include "support/cli.hpp"
#include "support/opencv_io.hpp"
#include "support/opencv_overlay.hpp"
#include "yolo/facade.hpp"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: pose_opencv_viz <model.onnx> <input.jpg|png> "
                     "<output.jpg|png>\n";
        return 1;
    }

    auto image_result = examples::load_cv_image(argv[2]);
    if (!image_result.ok()) {
        return examples::print_error(image_result.error);
    }

    auto pipeline_result = yolo::create_pipeline(yolo::ModelSpec{
        .path = argv[1],
        .task = yolo::TaskKind::pose,
        .model_name = std::string{"pose_opencv_viz"},
    });
    if (!pipeline_result.ok()) {
        return examples::print_error(pipeline_result.error);
    }

    const std::unique_ptr<yolo::Pipeline>& pipeline = *pipeline_result.value;
    const auto& info = pipeline->info();
    examples::print_pipeline_summary(info.model.adapter.value_or("unknown"),
                                     "pose", info.inputs.size(),
                                     info.outputs.size());

    const yolo::PoseResult result =
        pipeline->estimate_pose(examples::mat_view(*image_result.value));
    if (!result.ok()) {
        return examples::print_error(result.error);
    }

    cv::Mat annotated = image_result.value->clone();
    examples::draw_pose_detections(annotated, result.poses);

    if (const auto save_error = examples::save_cv_image(annotated, argv[3]);
        save_error.has_value()) {
        return examples::print_error(*save_error);
    }

    std::cout << "poses: " << result.poses.size() << '\n';
    std::cout << "saved: " << argv[3] << '\n';
    return EXIT_SUCCESS;
}
