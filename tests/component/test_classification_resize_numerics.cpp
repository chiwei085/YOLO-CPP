#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/detail/image_preprocess.hpp"

namespace
{

TEST_CASE("classification resize numerics stay stable for synthetic input",
          "[component][classification][preprocess]") {
    const auto image = yolo::test::make_bgr_image(
        {2, 4},
        {
            10, 20, 30, 40, 50, 60,
            70, 80, 90, 100, 110, 120,
            130, 140, 150, 160, 170, 180,
            190, 200, 210, 220, 230, 240,
        });

    const auto trace = yolo::detail::trace_classification_preprocess(
        image.view(), yolo::make_classification_preprocess_policy({2, 2}),
        "images");

    REQUIRE(trace.ok());
    REQUIRE(trace.value->resized_image.values.size() == 2 * 4 * 3);
    REQUIRE(trace.value->cropped_image.values.size() == 2 * 2 * 3);

    CHECK(trace.value->resized_image.values[0] ==
          Catch::Approx(30.0F).margin(1e-4));
    CHECK(trace.value->resized_image.values[1] ==
          Catch::Approx(20.0F).margin(1e-4));
    CHECK(trace.value->resized_image.values[2] ==
          Catch::Approx(10.0F).margin(1e-4));

    CHECK(trace.value->cropped_image.values[0] ==
          Catch::Approx(90.0F).margin(1e-4));
    CHECK(trace.value->cropped_image.values[1] ==
          Catch::Approx(80.0F).margin(1e-4));
    CHECK(trace.value->cropped_image.values[2] ==
          Catch::Approx(70.0F).margin(1e-4));

    CHECK(trace.value->tensor.values[0] ==
          Catch::Approx(90.0F / 255.0F).margin(1e-4));
}

}  // namespace
