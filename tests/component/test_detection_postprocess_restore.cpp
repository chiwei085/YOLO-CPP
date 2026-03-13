#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "yolo/detail/detection_runtime.hpp"

namespace
{

TEST_CASE("detection postprocess restore unmapped letterbox boxes to source image",
          "[component][detection]") {
    const auto detections = yolo::detail::postprocess_detections(
        {
            {.bbox = {160.0F, 200.0F, 320.0F, 160.0F}, .score = 0.9F, .class_id = 0},
        },
        yolo::PreprocessRecord{
            .source_size = {1280, 720},
            .target_size = {640, 640},
            .resized_size = {640, 360},
            .resize_scale = {0.5F, 0.5F},
            .padding = {.left = 0, .top = 140, .right = 0, .bottom = 140},
            .resize_mode = yolo::ResizeMode::letterbox,
        },
        {},
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::detect});

    REQUIRE(detections.size() == 1);
    CHECK(detections[0].bbox.x == Catch::Approx(320.0F));
    CHECK(detections[0].bbox.y == Catch::Approx(120.0F));
    CHECK(detections[0].bbox.width == Catch::Approx(640.0F));
    CHECK(detections[0].bbox.height == Catch::Approx(320.0F));
}

TEST_CASE("detection postprocess restore clamps coordinates on non-square inputs",
          "[component][detection]") {
    const auto detections = yolo::detail::postprocess_detections(
        {
            {.bbox = {-10.0F, -20.0F, 260.0F, 180.0F}, .score = 0.9F, .class_id = 0},
        },
        yolo::PreprocessRecord{
            .source_size = {320, 180},
            .target_size = {640, 640},
            .resized_size = {640, 360},
            .resize_scale = {2.0F, 2.0F},
            .padding = {.left = 0, .top = 140, .right = 0, .bottom = 140},
            .resize_mode = yolo::ResizeMode::letterbox,
        },
        {},
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::detect});

    REQUIRE(detections.size() == 1);
    CHECK(detections[0].bbox.x == Catch::Approx(0.0F));
    CHECK(detections[0].bbox.y == Catch::Approx(0.0F));
    CHECK(detections[0].bbox.width == Catch::Approx(130.0F));
    CHECK(detections[0].bbox.height == Catch::Approx(90.0F));
}

}  // namespace
