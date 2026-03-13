#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "test_utils.hpp"
#include "yolo/detail/image_preprocess.hpp"

namespace
{

TEST_CASE("detection preprocess letterboxes and preserves pad metadata",
          "[unit][preprocess]") {
    const auto image = yolo::test::make_bgr_image(
        {2, 1},
        {
            10, 20, 30,
            40, 50, 60,
        });

    auto result = yolo::detail::preprocess_image(
        image.view(), yolo::make_detection_preprocess_policy({4, 4}), "images");

    REQUIRE(result.ok());
    CHECK(result.value->record.resize_mode == yolo::ResizeMode::letterbox);
    CHECK(result.value->record.padding.top == 1);
    CHECK(result.value->record.padding.bottom == 1);
    CHECK(result.value->tensor.info.shape.rank() == 4);
    REQUIRE(result.value->tensor.values.size() == 3 * 4 * 4);

    CHECK(result.value->tensor.values[0] == Catch::Approx(114.0F / 255.0F));
    CHECK(result.value->tensor.values[4] == Catch::Approx(30.0F / 255.0F));
    CHECK(result.value->tensor.values[16 + 4] == Catch::Approx(20.0F / 255.0F));
    CHECK(result.value->tensor.values[32 + 4] == Catch::Approx(10.0F / 255.0F));
}

TEST_CASE("classification preprocess uses resize-crop policy",
          "[unit][preprocess]") {
    const auto image = yolo::test::make_bgr_image(
        {2, 4},
        {
            10, 10, 10, 10, 10, 10,
            20, 20, 20, 20, 20, 20,
            30, 30, 30, 30, 30, 30,
            40, 40, 40, 40, 40, 40,
        });

    auto result = yolo::detail::preprocess_image(
        image.view(), yolo::make_classification_preprocess_policy({2, 2}),
        "images");

    REQUIRE(result.ok());
    CHECK(result.value->record.resize_mode == yolo::ResizeMode::resize_crop);
    REQUIRE(result.value->record.crop.has_value());
    CHECK(result.value->record.crop->y == 1);
    REQUIRE(result.value->tensor.values.size() == 3 * 2 * 2);

    const float expected = 20.0F / 255.0F;
    CHECK(result.value->tensor.values[0] == Catch::Approx(expected).margin(1e-4));
}

TEST_CASE("preprocess rejects empty images", "[unit][preprocess]") {
    const auto result = yolo::detail::preprocess_image(
        yolo::ImageView{}, yolo::make_detection_preprocess_policy({4, 4}));

    CHECK_FALSE(result.ok());
    CHECK(result.error.code == yolo::ErrorCode::invalid_argument);
}

}  // namespace
