#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/adapters/ultralytics.hpp"

namespace
{

TEST_CASE("ultralytics detect probe resolves head layout and class count",
          "[adapter][detect]") {
    const auto result = yolo::adapters::ultralytics::probe_detection(
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::detect},
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 3, 640, 640})},
        {yolo::test::make_tensor_info("output0", yolo::TensorDataType::float32,
                                      {1, 8400, 84})});

    REQUIRE(result.ok());
    REQUIRE(result.value->detection.has_value());
    CHECK(result.value->model.task == yolo::TaskKind::detect);
    CHECK(result.value->model.class_count == 80);
    CHECK(result.value->detection->proposal_count == 8400);
    CHECK(result.value->detection->class_count == 80);
}

TEST_CASE("ultralytics detect probe rejects invalid output family",
          "[adapter][detect]") {
    const auto result = yolo::adapters::ultralytics::probe_detection(
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::detect},
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 3, 640, 640})},
        {yolo::test::make_tensor_info("output0", yolo::TensorDataType::float32,
                                      {1, 5, 5})});

    CHECK_FALSE(result.ok());
    CHECK(result.error.code == yolo::ErrorCode::unsupported_model);
}

}  // namespace
