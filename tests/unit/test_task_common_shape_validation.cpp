#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/detail/task_runtime_utils.hpp"

namespace
{

TEST_CASE("task common shape validation accepts positive static dimensions",
          "[unit][task-common]") {
    const auto input = yolo::test::make_tensor_info(
        "images", yolo::TensorDataType::float32, {1, 3, 640, 640});

    const auto height =
        yolo::detail::require_positive_dimension(input, 2, "task_common", "height");
    const auto width =
        yolo::detail::require_positive_dimension(input, 3, "task_common", "width");
    const auto class_count =
        yolo::detail::require_count(std::size_t{80}, "task_common", "class count");
    const auto proposal_count = yolo::detail::require_count(
        std::size_t{8400}, "task_common", "proposal count");
    const auto mask_coeff_count = yolo::detail::require_count(
        std::size_t{32}, "task_common", "mask coeff count");

    REQUIRE(height.ok());
    REQUIRE(width.ok());
    REQUIRE(class_count.ok());
    REQUIRE(proposal_count.ok());
    REQUIRE(mask_coeff_count.ok());
    CHECK(*height.value == 640);
    CHECK(*width.value == 640);
    CHECK(*class_count.value == 80);
    CHECK(*proposal_count.value == 8400);
    CHECK(*mask_coeff_count.value == 32);
}

TEST_CASE("task common shape validation rejects rank and dimension mismatches",
          "[unit][task-common]") {
    auto dynamic_rank = yolo::test::make_tensor_info(
        "images", yolo::TensorDataType::float32, {1, 3, 640, 640});
    dynamic_rank.shape.dims[2] = yolo::TensorDimension::dynamic();

    auto negative_dimension = yolo::test::make_tensor_info(
        "images", yolo::TensorDataType::float32, {1, 3, 640, 640});
    negative_dimension.shape.dims[3] = yolo::TensorDimension::fixed(-32);

    auto unexpected_rank = yolo::test::make_tensor_info(
        "images", yolo::TensorDataType::float32, {1, 3, 640});

    const auto dynamic_result = yolo::detail::require_positive_dimension(
        dynamic_rank, 2, "task_common", "height");
    const auto negative_result = yolo::detail::require_positive_dimension(
        negative_dimension, 3, "task_common", "width");
    const auto rank_result = yolo::detail::require_positive_dimension(
        unexpected_rank, 3, "task_common", "width");
    const auto dynamic_count =
        yolo::detail::require_count(std::nullopt, "task_common", "class count");
    const auto zero_count =
        yolo::detail::require_count(std::size_t{0}, "task_common", "proposal count");

    CHECK_FALSE(dynamic_result.ok());
    CHECK(dynamic_result.error.code == yolo::ErrorCode::shape_mismatch);
    CHECK_FALSE(negative_result.ok());
    CHECK(negative_result.error.code == yolo::ErrorCode::shape_mismatch);
    CHECK_FALSE(rank_result.ok());
    CHECK(rank_result.error.code == yolo::ErrorCode::shape_mismatch);
    CHECK_FALSE(dynamic_count.ok());
    CHECK(dynamic_count.error.code == yolo::ErrorCode::shape_mismatch);
    CHECK_FALSE(zero_count.ok());
    CHECK(zero_count.error.code == yolo::ErrorCode::shape_mismatch);
}

}  // namespace
