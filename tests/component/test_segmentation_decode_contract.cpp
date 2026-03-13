#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/detail/segmentation_runtime.hpp"

namespace
{

yolo::adapters::ultralytics::AdapterBindingSpec make_binding() {
    yolo::adapters::ultralytics::AdapterBindingSpec binding{};
    binding.model = yolo::ModelSpec{
        .path = "unused.onnx",
        .task = yolo::TaskKind::seg,
        .class_count = 2,
    };
    binding.outputs = {
        yolo::adapters::ultralytics::OutputBinding{
            .index = 1,
            .name = "predictions",
            .role = yolo::adapters::ultralytics::OutputRole::predictions,
            .data_type = yolo::TensorDataType::float32,
            .shape = yolo::TensorShape{.dims = {yolo::TensorDimension::fixed(1),
                                                yolo::TensorDimension::fixed(2),
                                                yolo::TensorDimension::fixed(8)}}},
        yolo::adapters::ultralytics::OutputBinding{
            .index = 0,
            .name = "proto",
            .role = yolo::adapters::ultralytics::OutputRole::proto,
            .data_type = yolo::TensorDataType::float32,
            .shape = yolo::TensorShape{
                .dims = {yolo::TensorDimension::fixed(1),
                         yolo::TensorDimension::fixed(2),
                         yolo::TensorDimension::fixed(2),
                         yolo::TensorDimension::fixed(2)}}},
    };
    binding.segmentation = yolo::adapters::ultralytics::SegmentationBindingSpec{
        .layout = yolo::adapters::ultralytics::DetectionHeadLayout::
            xywh_class_scores_last,
        .proposal_count = 2,
        .class_count = 2,
        .mask_channel_count = 2,
        .has_proto = true,
    };
    return binding;
}

TEST_CASE("segmentation decode contract uses bound outputs and mask metadata",
          "[component][segmentation]") {
    const auto spec_result =
        yolo::detail::segmentation_decode_spec_from_binding(make_binding());
    REQUIRE(spec_result.ok());
    CHECK(spec_result.value->output_index == 1);
    CHECK(spec_result.value->proto_output_index == 0);
    CHECK(spec_result.value->proposal_count == 2);
    CHECK(spec_result.value->class_count == 2);
    CHECK(spec_result.value->mask_channel_count == 2);
    CHECK(spec_result.value->proto_size.width == 2);
    CHECK(spec_result.value->proto_size.height == 2);

    const yolo::detail::RawOutputTensors outputs{
        yolo::test::make_float_tensor("proto", {1, 2, 2, 2},
                                      {1.0F, 1.0F, 1.0F, 1.0F,
                                       0.0F, 0.0F, 0.0F, 0.0F}),
        yolo::test::make_float_tensor("predictions", {1, 2, 8},
                                      {3.0F, 3.0F, 2.0F, 2.0F, 0.2F, 0.9F,
                                       1.0F, 0.0F,
                                       1.0F, 1.0F, 2.0F, 2.0F, 0.8F, 0.1F,
                                       0.0F, 1.0F}),
    };

    const auto decoded =
        yolo::detail::decode_segmentation(outputs, *spec_result.value);
    REQUIRE(decoded.ok());
    REQUIRE(decoded.value->candidates.size() == 2);
    CHECK(decoded.value->candidates[0].class_id == 1);
    CHECK(decoded.value->candidates[0].score == Catch::Approx(0.9F));
    CHECK(decoded.value->candidates[0].mask_coefficients ==
          std::vector<float>{1.0F, 0.0F});
    CHECK(decoded.value->candidates[1].class_id == 0);
    CHECK(decoded.value->candidates[1].mask_coefficients ==
          std::vector<float>{0.0F, 1.0F});
}

TEST_CASE("segmentation decode contract rejects missing binding fields",
          "[component][segmentation]") {
    auto missing_proto_binding = make_binding();
    missing_proto_binding.outputs.pop_back();

    const auto spec_result =
        yolo::detail::segmentation_decode_spec_from_binding(
            missing_proto_binding);
    CHECK_FALSE(spec_result.ok());
    CHECK(spec_result.error.code == yolo::ErrorCode::invalid_state);
}

TEST_CASE("segmentation decode contract rejects dtype rank and size mismatches",
          "[component][segmentation]") {
    const auto spec_result =
        yolo::detail::segmentation_decode_spec_from_binding(make_binding());
    REQUIRE(spec_result.ok());

    SECTION("prediction dtype mismatch") {
        const auto decoded = yolo::detail::decode_segmentation(
            {yolo::test::make_float_tensor("proto", {1, 2, 2, 2},
                                          {1.0F, 1.0F, 1.0F, 1.0F,
                                           1.0F, 1.0F, 1.0F, 1.0F}),
             yolo::test::make_uint8_tensor("predictions", {1, 2, 8},
                                           {1, 2, 3, 4})},
            *spec_result.value);

        CHECK_FALSE(decoded.ok());
        CHECK(decoded.error.code == yolo::ErrorCode::type_mismatch);
    }

    SECTION("proto rank mismatch") {
        const auto decoded = yolo::detail::decode_segmentation(
            {yolo::test::make_float_tensor("proto", {1, 2, 4},
                                          {1.0F, 1.0F, 1.0F, 1.0F,
                                           1.0F, 1.0F, 1.0F, 1.0F}),
             yolo::test::make_float_tensor("predictions", {1, 2, 8},
                                           {3.0F, 3.0F, 2.0F, 2.0F, 0.2F,
                                            0.9F, 1.0F, 0.0F,
                                            1.0F, 1.0F, 2.0F, 2.0F, 0.8F,
                                            0.1F, 0.0F, 1.0F})},
            *spec_result.value);

        CHECK_FALSE(decoded.ok());
        CHECK(decoded.error.code == yolo::ErrorCode::shape_mismatch);
    }

    SECTION("prediction payload too small for binding") {
        const auto decoded = yolo::detail::decode_segmentation(
            {yolo::test::make_float_tensor("proto", {1, 2, 2, 2},
                                          {1.0F, 1.0F, 1.0F, 1.0F,
                                           1.0F, 1.0F, 1.0F, 1.0F}),
             yolo::test::make_float_tensor("predictions", {1, 2, 8},
                                           {3.0F, 3.0F, 2.0F, 2.0F})},
            *spec_result.value);

        CHECK_FALSE(decoded.ok());
        CHECK(decoded.error.code == yolo::ErrorCode::shape_mismatch);
    }
}

}  // namespace
