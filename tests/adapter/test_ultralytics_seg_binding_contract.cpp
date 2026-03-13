#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/adapters/ultralytics.hpp"

namespace
{

TEST_CASE("ultralytics segmentation probe resolves prediction and proto binding",
          "[adapter][segmentation]") {
    const auto result = yolo::adapters::ultralytics::probe_segmentation(
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::seg},
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 3, 640, 640})},
        {yolo::test::make_tensor_info("mask_proto",
                                      yolo::TensorDataType::float32,
                                      {1, 32, 160, 160}),
         yolo::test::make_tensor_info("predictions",
                                      yolo::TensorDataType::float32,
                                      {1, 116, 8400})});

    REQUIRE(result.ok());
    REQUIRE(result.value->segmentation.has_value());
    CHECK(result.value->model.task == yolo::TaskKind::seg);
    CHECK(result.value->model.class_count == 80);
    CHECK(result.value->outputs.size() == 2);
    CHECK(result.value->outputs[0].role ==
          yolo::adapters::ultralytics::OutputRole::predictions);
    CHECK(result.value->outputs[1].role ==
          yolo::adapters::ultralytics::OutputRole::proto);
    CHECK(result.value->segmentation->proposal_count == 8400);
    CHECK(result.value->segmentation->class_count == 80);
    CHECK(result.value->segmentation->mask_channel_count == 32);
}

TEST_CASE("ultralytics segmentation probe rejects missing proto tensor",
          "[adapter][segmentation]") {
    const auto result = yolo::adapters::ultralytics::probe_segmentation(
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::seg},
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 3, 640, 640})},
        {yolo::test::make_tensor_info("pred0", yolo::TensorDataType::float32,
                                      {1, 116, 8400}),
         yolo::test::make_tensor_info("pred1", yolo::TensorDataType::float32,
                                      {1, 116, 8400})});

    CHECK_FALSE(result.ok());
    CHECK(result.error.code == yolo::ErrorCode::unsupported_model);
}

TEST_CASE("ultralytics segmentation probe does not accept detect-like outputs",
          "[adapter][segmentation]") {
    const auto result = yolo::adapters::ultralytics::probe_segmentation(
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::seg},
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 3, 640, 640})},
        {yolo::test::make_tensor_info("predictions",
                                      yolo::TensorDataType::float32,
                                      {1, 84, 8400}),
         yolo::test::make_tensor_info("aux", yolo::TensorDataType::float32,
                                      {1, 32, 8400})});

    CHECK_FALSE(result.ok());
    CHECK(result.error.code == yolo::ErrorCode::unsupported_model);
}

}  // namespace
