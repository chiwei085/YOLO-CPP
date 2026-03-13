#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/detail/image_preprocess.hpp"

namespace
{

TEST_CASE("classification preprocess contract uses identity normalization",
          "[unit][classification][preprocess]") {
    const yolo::PreprocessPolicy policy =
        yolo::make_classification_preprocess_policy({224, 224});

    CHECK(policy.resize_mode == yolo::ResizeMode::resize_crop);
    CHECK(policy.color_conversion == yolo::ColorConversion::swap_rb);
    CHECK(policy.tensor_layout == yolo::TensorLayout::nchw);
    CHECK(policy.normalize.input_scale == Catch::Approx(255.0F));
    CHECK(policy.normalize.mean[0] == Catch::Approx(0.0F));
    CHECK(policy.normalize.mean[1] == Catch::Approx(0.0F));
    CHECK(policy.normalize.mean[2] == Catch::Approx(0.0F));
    CHECK(policy.normalize.std[0] == Catch::Approx(1.0F));
    CHECK(policy.normalize.std[1] == Catch::Approx(1.0F));
    CHECK(policy.normalize.std[2] == Catch::Approx(1.0F));

    CHECK_FALSE(policy.normalize.mean[0] == Catch::Approx(0.485F));
    CHECK_FALSE(policy.normalize.mean[1] == Catch::Approx(0.456F));
    CHECK_FALSE(policy.normalize.mean[2] == Catch::Approx(0.406F));
    CHECK_FALSE(policy.normalize.std[0] == Catch::Approx(0.229F));
    CHECK_FALSE(policy.normalize.std[1] == Catch::Approx(0.224F));
    CHECK_FALSE(policy.normalize.std[2] == Catch::Approx(0.225F));
}

TEST_CASE("classification preprocess contract record and tensor follow policy",
          "[unit][classification][preprocess]") {
    const auto image = yolo::test::make_bgr_image(
        {2, 4},
        {
            10, 20, 30, 40, 50, 60,
            70, 80, 90, 100, 110, 120,
            130, 140, 150, 160, 170, 180,
            190, 200, 210, 220, 230, 240,
        });

    const yolo::PreprocessPolicy policy =
        yolo::make_classification_preprocess_policy({2, 2});
    auto result = yolo::detail::preprocess_image(image.view(), policy, "images");

    REQUIRE(result.ok());
    CHECK(result.value->record.resize_mode == policy.resize_mode);
    CHECK(result.value->record.color_conversion == policy.color_conversion);
    CHECK(result.value->record.normalize.input_scale ==
          Catch::Approx(policy.normalize.input_scale));
    REQUIRE(result.value->record.crop.has_value());
    CHECK(result.value->record.crop->y == 1);
    CHECK(result.value->tensor.info.shape.rank() == 4);
    REQUIRE(result.value->tensor.values.size() == 3 * 2 * 2);

    // resize-crop + swap_rb + identity normalization => raw RGB / 255 in NCHW
    CHECK(result.value->tensor.values[0] ==
          Catch::Approx(90.0F / 255.0F).margin(1e-4));
    CHECK(result.value->tensor.values[4] ==
          Catch::Approx(80.0F / 255.0F).margin(1e-4));
    CHECK(result.value->tensor.values[8] ==
          Catch::Approx(70.0F / 255.0F).margin(1e-4));
}

}  // namespace
