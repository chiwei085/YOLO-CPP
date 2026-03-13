#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "test_utils.hpp"
#include "yolo/detail/detection_runtime.hpp"

namespace
{

TEST_CASE("detection decode supports xywh class-scores-last layout",
          "[component][detection]") {
    const auto result = yolo::detail::decode_detections(
        {yolo::test::make_float_tensor("pred", {1, 2, 6},
                                       {10.0F, 20.0F, 4.0F, 6.0F, 0.1F, 0.9F,
                                        30.0F, 40.0F, 8.0F, 10.0F, 0.8F,
                                        0.2F})},
        yolo::detail::DetectionDecodeSpec{
            .output_index = 0,
            .layout = yolo::detail::DetectionLayout::xywh_class_scores_last,
            .proposal_count = 2,
            .class_count = 2,
        });

    REQUIRE(result.ok());
    REQUIRE(result.value->size() == 2);
    CHECK(result.value->at(0).class_id == 1);
    CHECK(result.value->at(0).bbox.x == Catch::Approx(8.0F));
    CHECK(result.value->at(1).class_id == 0);
}

TEST_CASE("detection decode supports xywh class-scores-first layout",
          "[component][detection]") {
    const auto result = yolo::detail::decode_detections(
        {yolo::test::make_float_tensor(
            "pred", {1, 6, 2},
            {10.0F, 30.0F, 20.0F, 40.0F, 4.0F, 8.0F, 6.0F, 10.0F, 0.1F, 0.8F,
             0.9F, 0.2F})},
        yolo::detail::DetectionDecodeSpec{
            .output_index = 0,
            .layout = yolo::detail::DetectionLayout::xywh_class_scores_first,
            .proposal_count = 2,
            .class_count = 2,
        });

    REQUIRE(result.ok());
    REQUIRE(result.value->size() == 2);
    CHECK(result.value->at(0).class_id == 1);
    CHECK(result.value->at(1).class_id == 0);
}

TEST_CASE("detection decode validates payload size separately from postprocess",
          "[component][detection]") {
    const auto result = yolo::detail::decode_detections(
        {yolo::test::make_float_tensor("pred", {1, 2, 6}, {1.0F, 2.0F, 3.0F})},
        yolo::detail::DetectionDecodeSpec{
            .output_index = 0,
            .layout = yolo::detail::DetectionLayout::xywh_class_scores_last,
            .proposal_count = 2,
            .class_count = 2,
        });

    CHECK_FALSE(result.ok());
    CHECK(result.error.code == yolo::ErrorCode::shape_mismatch);
}

}  // namespace
