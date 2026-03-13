#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "yolo/detail/detection_runtime.hpp"

namespace
{

TEST_CASE("detection postprocess applies confidence filtering and class-aware nms",
          "[component][detection]") {
    const std::vector<yolo::detail::DetectionCandidate> candidates = {
        {.bbox = {10.0F, 10.0F, 20.0F, 20.0F}, .score = 0.95F, .class_id = 1},
        {.bbox = {12.0F, 12.0F, 20.0F, 20.0F}, .score = 0.80F, .class_id = 1},
        {.bbox = {12.0F, 12.0F, 20.0F, 20.0F}, .score = 0.70F, .class_id = 2},
        {.bbox = {0.0F, 0.0F, 0.0F, 5.0F}, .score = 0.99F, .class_id = 0},
    };

    const auto detections = yolo::detail::postprocess_detections(
        candidates,
        yolo::PreprocessRecord{
            .source_size = {100, 100},
            .target_size = {100, 100},
            .resized_size = {100, 100},
            .resize_scale = {1.0F, 1.0F},
            .resize_mode = yolo::ResizeMode::direct,
        },
        yolo::DetectionOptions{
            .confidence_threshold = 0.5F,
            .nms_iou_threshold = 0.3F,
            .max_detections = 10,
            .class_agnostic_nms = false,
        },
        yolo::ModelSpec{
            .path = "unused.onnx",
            .task = yolo::TaskKind::detect,
            .labels = {"zero", "one", "two"},
        });

    REQUIRE(detections.size() == 2);
    CHECK(detections[0].class_id == 1);
    CHECK(detections[1].class_id == 2);
}

TEST_CASE("detection postprocess unmaps letterboxed boxes back to source image",
          "[component][detection]") {
    const auto detections = yolo::detail::postprocess_detections(
        {
            {.bbox = {2.0F, 2.0F, 4.0F, 2.0F}, .score = 0.9F, .class_id = 0},
        },
        yolo::PreprocessRecord{
            .source_size = {4, 2},
            .target_size = {4, 4},
            .resized_size = {4, 2},
            .resize_scale = {1.0F, 1.0F},
            .padding = {.left = 0, .top = 1, .right = 0, .bottom = 1},
            .resize_mode = yolo::ResizeMode::letterbox,
        },
        {},
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::detect});

    REQUIRE(detections.size() == 1);
    CHECK(detections[0].bbox.x == Catch::Approx(2.0F));
    CHECK(detections[0].bbox.y == Catch::Approx(1.0F));
    CHECK(detections[0].bbox.height == Catch::Approx(1.0F));
}

}  // namespace
