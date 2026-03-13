#include <catch2/catch_test_macros.hpp>

#include "yolo/detail/detection_runtime.hpp"

namespace
{

yolo::PreprocessRecord direct_record() {
    return yolo::PreprocessRecord{
        .source_size = {100, 100},
        .target_size = {100, 100},
        .resized_size = {100, 100},
        .resize_scale = {1.0F, 1.0F},
        .resize_mode = yolo::ResizeMode::direct,
    };
}

yolo::ModelSpec detect_model() {
    return yolo::ModelSpec{
        .path = "unused.onnx",
        .task = yolo::TaskKind::detect,
        .labels = {"zero", "one", "two"},
    };
}

TEST_CASE("detection postprocess filtering handles empty and low-confidence inputs",
          "[component][detection]") {
    const auto empty = yolo::detail::postprocess_detections(
        {}, direct_record(),
        yolo::DetectionOptions{.confidence_threshold = 0.5F},
        detect_model());
    CHECK(empty.empty());

    const auto low = yolo::detail::postprocess_detections(
        {{.bbox = {0.0F, 0.0F, 10.0F, 10.0F}, .score = 0.2F, .class_id = 0}},
        direct_record(), yolo::DetectionOptions{.confidence_threshold = 0.5F},
        detect_model());
    CHECK(low.empty());
}

TEST_CASE("detection postprocess filtering keeps valid proposals by class",
          "[component][detection]") {
    const auto detections = yolo::detail::postprocess_detections(
        {
            {.bbox = {10.0F, 10.0F, 20.0F, 20.0F}, .score = 0.95F, .class_id = 1},
            {.bbox = {50.0F, 50.0F, 20.0F, 20.0F}, .score = 0.75F, .class_id = 2},
            {.bbox = {20.0F, 20.0F, 0.0F, 10.0F}, .score = 0.99F, .class_id = 0},
        },
        direct_record(),
        yolo::DetectionOptions{
            .confidence_threshold = 0.5F,
            .nms_iou_threshold = 0.5F,
            .max_detections = 10,
            .class_agnostic_nms = false,
        },
        detect_model());

    REQUIRE(detections.size() == 2);
    CHECK(detections[0].class_id == 1);
    CHECK(detections[1].class_id == 2);
}

}  // namespace
