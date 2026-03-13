#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/detail/detection_runtime.hpp"

namespace
{

TEST_CASE("detection decode layouts honor predictions output binding",
          "[component][detection]") {
    const yolo::detail::RawOutputTensors outputs{
        yolo::test::make_float_tensor("aux", {1, 1, 1}, {7.0F}),
        yolo::test::make_float_tensor("pred", {1, 2, 6},
                                      {10.0F, 20.0F, 4.0F, 6.0F, 0.1F, 0.9F,
                                       30.0F, 40.0F, 8.0F, 10.0F, 0.8F, 0.2F}),
    };

    const auto result = yolo::detail::decode_detections(
        outputs,
        yolo::detail::DetectionDecodeSpec{
            .output_index = 1,
            .layout = yolo::detail::DetectionLayout::xywh_class_scores_last,
            .proposal_count = 2,
            .class_count = 2,
        });

    REQUIRE(result.ok());
    REQUIRE(result.value->size() == 2);
    CHECK(result.value->at(0).bbox.x == Catch::Approx(8.0F));
    CHECK(result.value->at(0).bbox.y == Catch::Approx(17.0F));
    CHECK(result.value->at(0).score == Catch::Approx(0.9F));
    CHECK(result.value->at(0).class_id == 1);
    CHECK(result.value->at(1).score == Catch::Approx(0.8F));
    CHECK(result.value->at(1).class_id == 0);
}

TEST_CASE("detection decode layouts honor proposal and class counts",
          "[component][detection]") {
    const auto result = yolo::detail::decode_detections(
        {yolo::test::make_float_tensor(
            "pred", {1, 7, 3},
            {10.0F, 30.0F, 50.0F, 20.0F, 40.0F, 60.0F, 4.0F, 8.0F, 12.0F,
             6.0F, 10.0F, 14.0F, 0.1F, 0.8F, 0.2F, 0.9F, 0.2F, 0.1F, 0.3F,
             0.4F, 0.95F})},
        yolo::detail::DetectionDecodeSpec{
            .output_index = 0,
            .layout = yolo::detail::DetectionLayout::xywh_class_scores_first,
            .proposal_count = 3,
            .class_count = 3,
        });

    REQUIRE(result.ok());
    REQUIRE(result.value->size() == 3);
    CHECK(result.value->at(0).class_id == 1);
    CHECK(result.value->at(1).class_id == 0);
    CHECK(result.value->at(2).class_id == 2);
    CHECK(result.value->at(2).score == Catch::Approx(0.95F));
}

}  // namespace
