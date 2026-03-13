#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "test_utils.hpp"
#include "yolo/detail/task_runtime_utils.hpp"

namespace
{

TEST_CASE("task common input selection returns the only input",
          "[unit][task-common]") {
    const std::vector<yolo::TensorInfo> inputs{
        yolo::test::make_tensor_info("tokens", yolo::TensorDataType::float32,
                                     {1, 128})};

    const auto result = yolo::detail::select_primary_input(inputs, "task_common");

    REQUIRE(result.ok());
    CHECK(result.value->name == "tokens");
}

TEST_CASE("task common input selection prefers image-like input",
          "[unit][task-common]") {
    const std::vector<yolo::TensorInfo> inputs{
        yolo::test::make_tensor_info("tokens", yolo::TensorDataType::float32,
                                     {1, 77}),
        yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                     {1, 3, 640, 640}),
        yolo::test::make_tensor_info("scale", yolo::TensorDataType::float32,
                                     {1})};

    const auto result = yolo::detail::select_primary_input(inputs, "task_common");

    REQUIRE(result.ok());
    CHECK(result.value->name == "images");
}

TEST_CASE("task common input selection fails when no image-like input exists",
          "[unit][task-common]") {
    const std::vector<yolo::TensorInfo> inputs{
        yolo::test::make_tensor_info("tokens", yolo::TensorDataType::float32,
                                     {1, 77}),
        yolo::test::make_tensor_info("mask", yolo::TensorDataType::uint8,
                                     {1, 256})};

    const auto result = yolo::detail::select_primary_input(inputs, "task_common");

    CHECK_FALSE(result.ok());
    CHECK(result.error.code == yolo::ErrorCode::unsupported_model);
    REQUIRE(result.error.context.has_value());
    CHECK(result.error.context->component == std::optional<std::string>{"task_common"});
}

TEST_CASE("task common input validation reports name shape and dtype failures",
          "[unit][task-common]") {
    auto dtype_mismatch = yolo::test::make_tensor_info(
        "images", yolo::TensorDataType::uint8, {1, 3, 640, 640});
    const auto dtype_result = yolo::detail::validate_primary_input(
        dtype_mismatch, "task_common", "images");
    CHECK_FALSE(dtype_result.ok());
    CHECK(dtype_result.error.code == yolo::ErrorCode::type_mismatch);

    auto rank_mismatch = yolo::test::make_tensor_info(
        "images", yolo::TensorDataType::float32, {1, 640, 640});
    const auto rank_result = yolo::detail::validate_primary_input(
        rank_mismatch, "task_common", "images");
    CHECK_FALSE(rank_result.ok());
    CHECK(rank_result.error.code == yolo::ErrorCode::shape_mismatch);

    auto name_mismatch = yolo::test::make_tensor_info(
        "pixels", yolo::TensorDataType::float32, {1, 3, 640, 640});
    const auto name_result = yolo::detail::validate_primary_input(
        name_mismatch, "task_common", "images");
    CHECK_FALSE(name_result.ok());
    CHECK(name_result.error.code == yolo::ErrorCode::invalid_argument);
}

}  // namespace
