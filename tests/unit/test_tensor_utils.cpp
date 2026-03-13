#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "test_utils.hpp"
#include "yolo/detail/tensor_utils.hpp"

namespace
{

TEST_CASE("tensor utils compute element and dense byte counts",
          "[unit][tensor]") {
    CHECK(yolo::detail::tensor_element_size(yolo::TensorDataType::float32) == 4);
    CHECK(yolo::detail::tensor_element_size(yolo::TensorDataType::float16) == 2);

    const auto dense = yolo::detail::dense_byte_count(
        yolo::test::make_tensor_info("scores", yolo::TensorDataType::float32,
                                     {1, 3, 2}));
    REQUIRE(dense.has_value());
    CHECK(*dense == 24);
}

TEST_CASE("tensor utils return nullopt for dynamic dense byte count",
          "[unit][tensor]") {
    yolo::TensorInfo info{};
    info.name = "dynamic";
    info.data_type = yolo::TensorDataType::float32;
    info.shape.dims = {
        yolo::TensorDimension::fixed(1),
        yolo::TensorDimension::dynamic(),
    };

    CHECK_FALSE(yolo::detail::dense_byte_count(info).has_value());
}

TEST_CASE("tensor utils copy float tensor payload and validate type",
          "[unit][tensor]") {
    const auto copied = yolo::detail::copy_float_tensor_data(
        yolo::test::make_float_tensor("scores", {3}, {0.1F, 0.2F, 0.7F}),
        "tensor_test");

    REQUIRE(copied.ok());
    REQUIRE(copied.value->size() == 3);
    CHECK(copied.value->at(2) == Catch::Approx(0.7F));

    const auto bad_type = yolo::detail::copy_float_tensor_data(
        yolo::test::make_uint8_tensor("scores", {3}, {1, 2, 3}), "tensor_test");
    CHECK_FALSE(bad_type.ok());
    CHECK(bad_type.error.code == yolo::ErrorCode::type_mismatch);
}

}  // namespace
