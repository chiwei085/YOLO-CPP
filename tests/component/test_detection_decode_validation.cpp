#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/detail/detection_runtime.hpp"

namespace
{

TEST_CASE("detection decode validation rejects output index and dtype mismatches",
          "[component][detection]") {
    const yolo::detail::RawOutputTensors outputs{
        yolo::test::make_uint8_tensor("pred", {1, 2, 6}, {1, 2, 3, 4}),
    };

    const auto index_result = yolo::detail::decode_detections(
        outputs,
        yolo::detail::DetectionDecodeSpec{
            .output_index = 3,
            .layout = yolo::detail::DetectionLayout::xywh_class_scores_last,
            .proposal_count = 2,
            .class_count = 2,
        });
    CHECK_FALSE(index_result.ok());
    CHECK(index_result.error.code == yolo::ErrorCode::shape_mismatch);

    const auto dtype_result = yolo::detail::decode_detections(
        outputs,
        yolo::detail::DetectionDecodeSpec{
            .output_index = 0,
            .layout = yolo::detail::DetectionLayout::xywh_class_scores_last,
            .proposal_count = 2,
            .class_count = 2,
        });
    CHECK_FALSE(dtype_result.ok());
    CHECK(dtype_result.error.code == yolo::ErrorCode::type_mismatch);
}

TEST_CASE("detection decode validation rejects rank and binding mismatches",
          "[component][detection]") {
    auto wrong_rank =
        yolo::test::make_float_tensor("pred", {1, 1, 2, 6}, {1.0F, 2.0F, 3.0F});
    auto wrong_shape =
        yolo::test::make_float_tensor("pred", {1, 2, 5},
                                      {10.0F, 20.0F, 4.0F, 6.0F, 0.9F,
                                       30.0F, 40.0F, 8.0F, 10.0F, 0.2F});
    auto empty_tensor = yolo::test::make_float_tensor("pred", {1, 0, 6}, {});

    const auto rank_result = yolo::detail::decode_detections(
        {wrong_rank},
        yolo::detail::DetectionDecodeSpec{
            .output_index = 0,
            .layout = yolo::detail::DetectionLayout::xywh_class_scores_last,
            .proposal_count = 2,
            .class_count = 2,
        });
    CHECK_FALSE(rank_result.ok());
    CHECK(rank_result.error.code == yolo::ErrorCode::shape_mismatch);

    const auto class_count_result = yolo::detail::decode_detections(
        {wrong_shape},
        yolo::detail::DetectionDecodeSpec{
            .output_index = 0,
            .layout = yolo::detail::DetectionLayout::xywh_class_scores_last,
            .proposal_count = 2,
            .class_count = 2,
        });
    CHECK_FALSE(class_count_result.ok());
    CHECK(class_count_result.error.code == yolo::ErrorCode::shape_mismatch);

    const auto empty_result = yolo::detail::decode_detections(
        {empty_tensor},
        yolo::detail::DetectionDecodeSpec{
            .output_index = 0,
            .layout = yolo::detail::DetectionLayout::xywh_class_scores_last,
            .proposal_count = 1,
            .class_count = 2,
        });
    CHECK_FALSE(empty_result.ok());
    CHECK(empty_result.error.code == yolo::ErrorCode::shape_mismatch);
}

}  // namespace
