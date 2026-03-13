#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/detail/image_preprocess.hpp"

namespace
{

TEST_CASE("classification center crop geometry stays centered for even deltas",
          "[unit][classification][geometry]") {
    const auto image = yolo::test::make_bgr_image(
        {200, 400}, std::vector<std::uint8_t>(200 * 400 * 3, 10));

    const auto trace = yolo::detail::trace_classification_preprocess(
        image.view(), yolo::make_classification_preprocess_policy({224, 224}),
        "images");

    REQUIRE(trace.ok());
    REQUIRE(trace.value->crop.has_value());
    CHECK(trace.value->crop->x == 0);
    CHECK(trace.value->crop->y == 112);
    CHECK(trace.value->crop->width == 224);
    CHECK(trace.value->crop->height == 224);
}

TEST_CASE("classification center crop geometry floors odd center offsets",
          "[unit][classification][geometry]") {
    const auto image = yolo::test::make_bgr_image(
        {201, 333}, std::vector<std::uint8_t>(201 * 333 * 3, 10));

    const auto trace = yolo::detail::trace_classification_preprocess(
        image.view(), yolo::make_classification_preprocess_policy({224, 224}),
        "images");

    REQUIRE(trace.ok());
    REQUIRE(trace.value->crop.has_value());
    CHECK(trace.value->crop->x == 0);
    CHECK(trace.value->crop->y == (370 - 224) / 2);
}

}  // namespace
