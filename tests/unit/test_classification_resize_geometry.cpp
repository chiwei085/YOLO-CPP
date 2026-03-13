#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/detail/image_preprocess.hpp"

namespace
{

TEST_CASE("classification resize geometry preserves aspect ratio into target crop",
          "[unit][classification][geometry]") {
    const auto image = yolo::test::make_bgr_image(
        {400, 200}, std::vector<std::uint8_t>(400 * 200 * 3, 10));

    const auto trace = yolo::detail::trace_classification_preprocess(
        image.view(), yolo::make_classification_preprocess_policy({224, 224}),
        "images");

    REQUIRE(trace.ok());
    CHECK(trace.value->source_size.width == 400);
    CHECK(trace.value->source_size.height == 200);
    CHECK(trace.value->resized_size.width == 448);
    CHECK(trace.value->resized_size.height == 224);
    REQUIRE(trace.value->crop.has_value());
    CHECK(trace.value->crop->width == 224);
    CHECK(trace.value->crop->height == 224);
}

TEST_CASE("classification resize geometry floors scaled width for 4:3 inputs",
          "[unit][classification][geometry]") {
    const auto image = yolo::test::make_bgr_image(
        {768, 576}, std::vector<std::uint8_t>(768 * 576 * 3, 10));

    const auto trace = yolo::detail::trace_classification_preprocess(
        image.view(), yolo::make_classification_preprocess_policy({224, 224}),
        "images");

    REQUIRE(trace.ok());
    CHECK(trace.value->resized_size.width == 298);
    CHECK(trace.value->resized_size.height == 224);
    REQUIRE(trace.value->crop.has_value());
    CHECK(trace.value->crop->x == 37);
    CHECK(trace.value->crop->y == 0);
}

TEST_CASE("classification resize geometry floors odd dimensions consistently",
          "[unit][classification][geometry]") {
    const auto image = yolo::test::make_bgr_image(
        {333, 201}, std::vector<std::uint8_t>(333 * 201 * 3, 10));

    const auto trace = yolo::detail::trace_classification_preprocess(
        image.view(), yolo::make_classification_preprocess_policy({224, 224}),
        "images");

    REQUIRE(trace.ok());
    CHECK(trace.value->resized_size.width == 371);
    CHECK(trace.value->resized_size.height == 224);
    REQUIRE(trace.value->crop.has_value());
    CHECK(trace.value->crop->x == (371 - 224) / 2);
    CHECK(trace.value->crop->y == 0);
}

}  // namespace
