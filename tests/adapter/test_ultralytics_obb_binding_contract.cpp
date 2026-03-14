#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/adapters/ultralytics.hpp"

namespace
{

TEST_CASE(
    "ultralytics obb probe resolves prediction binding and angle metadata",
    "[adapter][obb]") {
    const auto result = yolo::adapters::ultralytics::probe_obb(
        yolo::ModelSpec{
            .path = "unused.onnx",
            .task = yolo::TaskKind::obb,
            .class_count = 15,
        },
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 3, 1024, 1024})},
        {yolo::test::make_tensor_info("output0", yolo::TensorDataType::float32,
                                      {1, 20, 21504})});

    REQUIRE(result.ok());
    REQUIRE(result.value->obb.has_value());
    CHECK(result.value->model.task == yolo::TaskKind::obb);
    CHECK(result.value->model.class_count == 15);
    REQUIRE(result.value->outputs.size() == 1);
    CHECK(result.value->obb->layout ==
          yolo::adapters::ultralytics::DetectionHeadLayout::
              xywh_class_scores_first);
    CHECK(result.value->obb->proposal_count == 21504);
    CHECK(result.value->obb->class_count == 15);
    CHECK(result.value->obb->box_coordinate_count == 4);
    CHECK(result.value->obb->class_channel_offset == 4);
    CHECK(result.value->obb->angle_channel_offset == 19);
    CHECK(result.value->obb->box_encoding ==
          yolo::adapters::ultralytics::ObbBoxEncoding::center_size_rotation);
    CHECK(result.value->obb->angle_is_radians);
}

TEST_CASE("ultralytics obb probe rejects detect-like outputs",
          "[adapter][obb]") {
    const auto result = yolo::adapters::ultralytics::probe_obb(
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::obb},
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 3, 1024, 1024})},
        {yolo::test::make_tensor_info("output0", yolo::TensorDataType::float32,
                                      {1, 84, 8400})});

    CHECK_FALSE(result.ok());
    CHECK(result.error.code == yolo::ErrorCode::unsupported_model);
}

TEST_CASE("ultralytics obb probe can infer class count from output width",
          "[adapter][obb]") {
    const auto result = yolo::adapters::ultralytics::probe_obb(
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::obb},
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 3, 1024, 1024})},
        {yolo::test::make_tensor_info("output0", yolo::TensorDataType::float32,
                                      {1, 20, 21504})});

    REQUIRE(result.ok());
    REQUIRE(result.value->obb.has_value());
    CHECK(result.value->obb->class_count == 15);
    CHECK(result.value->model.class_count == 15);
}

}  // namespace
