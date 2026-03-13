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
    return yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::detect};
}

TEST_CASE("detection postprocess nms suppresses overlapping boxes of the same class",
          "[component][detection]") {
    const auto detections = yolo::detail::postprocess_detections(
        {
            {.bbox = {10.0F, 10.0F, 20.0F, 20.0F}, .score = 0.95F, .class_id = 1},
            {.bbox = {11.0F, 11.0F, 20.0F, 20.0F}, .score = 0.85F, .class_id = 1},
        },
        direct_record(),
        yolo::DetectionOptions{
            .confidence_threshold = 0.1F,
            .nms_iou_threshold = 0.3F,
            .max_detections = 10,
            .class_agnostic_nms = false,
        },
        detect_model());

    REQUIRE(detections.size() == 1);
    CHECK(detections.front().score == 0.95F);
}

TEST_CASE("detection postprocess nms keeps non-overlapping and cross-class boxes",
          "[component][detection]") {
    const auto detections = yolo::detail::postprocess_detections(
        {
            {.bbox = {10.0F, 10.0F, 20.0F, 20.0F}, .score = 0.95F, .class_id = 1},
            {.bbox = {60.0F, 60.0F, 15.0F, 15.0F}, .score = 0.75F, .class_id = 1},
            {.bbox = {11.0F, 11.0F, 20.0F, 20.0F}, .score = 0.90F, .class_id = 2},
        },
        direct_record(),
        yolo::DetectionOptions{
            .confidence_threshold = 0.1F,
            .nms_iou_threshold = 0.3F,
            .max_detections = 10,
            .class_agnostic_nms = false,
        },
        detect_model());

    REQUIRE(detections.size() == 3);
    CHECK(detections[0].class_id == 1);
    CHECK(detections[1].class_id == 2);
    CHECK(detections[2].class_id == 1);
}

TEST_CASE("detection postprocess nms handles identical boxes and iou boundary",
          "[component][detection]") {
    const auto detections = yolo::detail::postprocess_detections(
        {
            {.bbox = {10.0F, 10.0F, 20.0F, 20.0F}, .score = 0.95F, .class_id = 0},
            {.bbox = {10.0F, 10.0F, 20.0F, 20.0F}, .score = 0.90F, .class_id = 0},
            {.bbox = {30.0F, 10.0F, 20.0F, 20.0F}, .score = 0.80F, .class_id = 0},
        },
        direct_record(),
        yolo::DetectionOptions{
            .confidence_threshold = 0.1F,
            .nms_iou_threshold = 1.0F,
            .max_detections = 10,
            .class_agnostic_nms = false,
        },
        detect_model());

    REQUIRE(detections.size() == 2);
}

}  // namespace
