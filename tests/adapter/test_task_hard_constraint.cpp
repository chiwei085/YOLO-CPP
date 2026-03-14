#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/adapters/ultralytics.hpp"

namespace
{

TEST_CASE("pose binding rejects plain detection output contracts",
          "[adapter][constraints]") {
    const auto result = yolo::adapters::ultralytics::probe_pose(
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::pose},
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 3, 640, 640})},
        {yolo::test::make_tensor_info("output0", yolo::TensorDataType::float32,
                                      {1, 84, 8400})});

    CHECK_FALSE(result.ok());
    CHECK(result.error.code == yolo::ErrorCode::unsupported_model);
}

TEST_CASE("obb binding rejects plain detection output contracts",
          "[adapter][constraints]") {
    const auto result = yolo::adapters::ultralytics::probe_obb(
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::obb},
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 3, 640, 640})},
        {yolo::test::make_tensor_info("output0", yolo::TensorDataType::float32,
                                      {1, 84, 8400})});

    CHECK_FALSE(result.ok());
    CHECK(result.error.code == yolo::ErrorCode::unsupported_model);
}

}  // namespace
