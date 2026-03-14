#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/adapters/ultralytics.hpp"

namespace
{

TEST_CASE(
    "ultralytics pose probe resolves prediction binding and keypoint metadata",
    "[adapter][pose]") {
    const auto result = yolo::adapters::ultralytics::probe_pose(
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::pose},
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 3, 640, 640})},
        {yolo::test::make_tensor_info("output0", yolo::TensorDataType::float32,
                                      {1, 56, 8400})});

    REQUIRE(result.ok());
    REQUIRE(result.value->pose.has_value());
    CHECK(result.value->model.task == yolo::TaskKind::pose);
    CHECK(result.value->model.class_count == 1);
    REQUIRE(result.value->outputs.size() == 1);
    CHECK(result.value->outputs.front().role ==
          yolo::adapters::ultralytics::OutputRole::predictions);
    CHECK(result.value->pose->layout ==
          yolo::adapters::ultralytics::DetectionHeadLayout::
              xywh_class_scores_first);
    CHECK(result.value->pose->proposal_count == 8400);
    CHECK(result.value->pose->class_count == 1);
    CHECK(result.value->pose->keypoint_count == 17);
    CHECK(result.value->pose->keypoint_dimension == 3);
    CHECK(result.value->pose->keypoint_semantic ==
          yolo::adapters::ultralytics::PoseKeypointSemantic::xyscore);
}

TEST_CASE("ultralytics pose probe rejects detect-like outputs",
          "[adapter][pose]") {
    const auto result = yolo::adapters::ultralytics::probe_pose(
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::pose},
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 3, 640, 640})},
        {yolo::test::make_tensor_info("output0", yolo::TensorDataType::float32,
                                      {1, 84, 8400})});

    CHECK_FALSE(result.ok());
    CHECK(result.error.code == yolo::ErrorCode::unsupported_model);
}

TEST_CASE("ultralytics pose probe honors declared class count",
          "[adapter][pose]") {
    const auto result = yolo::adapters::ultralytics::probe_pose(
        yolo::ModelSpec{
            .path = "unused.onnx",
            .task = yolo::TaskKind::pose,
            .class_count = 2,
        },
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 3, 640, 640})},
        {yolo::test::make_tensor_info("output0", yolo::TensorDataType::float32,
                                      {1, 57, 8400})});

    REQUIRE(result.ok());
    REQUIRE(result.value->pose.has_value());
    CHECK(result.value->pose->class_count == 2);
    CHECK(result.value->pose->keypoint_count == 17);
}

}  // namespace
