#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/detail/classification_resize.hpp"

namespace
{

TEST_CASE("classification resize reference matches horizontal antialias case",
          "[component][classification][preprocess]") {
    const auto image = yolo::test::make_bgr_image(
        {4, 2},
        {
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
            15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125,
        });

    const auto resized = yolo::detail::resize_classification_image(
        image.view(), yolo::make_classification_preprocess_policy({2, 2}),
        {3, 2});

    REQUIRE(resized.ok());
    CHECK(resized.value->size.width == 3);
    CHECK(resized.value->size.height == 2);
    CHECK(resized.value->values ==
          std::vector<float>{39.0F, 29.0F, 19.0F, 75.0F, 65.0F, 55.0F,
                             111.0F, 101.0F, 91.0F, 44.0F, 34.0F, 24.0F,
                             80.0F, 70.0F, 60.0F, 116.0F, 106.0F, 96.0F});
}

TEST_CASE("classification resize reference matches vertical antialias case",
          "[component][classification][preprocess]") {
    const auto image = yolo::test::make_bgr_image(
        {2, 4},
        {
            5, 15, 25, 35, 45, 55,
            65, 75, 85, 95, 105, 115,
            125, 135, 145, 155, 165, 175,
            185, 195, 205, 215, 225, 235,
        });

    const auto resized = yolo::detail::resize_classification_image(
        image.view(), yolo::make_classification_preprocess_policy({2, 2}),
        {2, 3});

    REQUIRE(resized.ok());
    CHECK(resized.value->values ==
          std::vector<float>{43.0F, 33.0F, 23.0F, 73.0F, 63.0F, 53.0F,
                             115.0F, 105.0F, 95.0F, 145.0F, 135.0F, 125.0F,
                             187.0F, 177.0F, 167.0F, 217.0F, 207.0F, 197.0F});
}

TEST_CASE("classification resize reference matches odd-even antialias case",
          "[component][classification][preprocess]") {
    const auto image = yolo::test::make_bgr_image(
        {5, 3},
        {
            0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
            15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155,
            30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170,
        });

    const auto resized = yolo::detail::resize_classification_image(
        image.view(), yolo::make_classification_preprocess_policy({2, 2}),
        {4, 2});

    REQUIRE(resized.ok());
    CHECK(resized.value->values ==
          std::vector<float>{34.0F, 24.0F, 14.0F, 69.0F, 59.0F, 49.0F,
                             103.0F, 93.0F, 83.0F, 139.0F, 129.0F, 119.0F,
                             52.0F, 42.0F, 32.0F, 87.0F, 77.0F, 67.0F,
                             121.0F, 111.0F, 101.0F, 157.0F, 147.0F, 137.0F});
}

}  // namespace
