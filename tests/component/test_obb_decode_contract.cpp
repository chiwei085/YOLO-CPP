#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/detail/obb_runtime.hpp"

namespace
{

yolo::adapters::ultralytics::AdapterBindingSpec make_binding() {
    yolo::adapters::ultralytics::AdapterBindingSpec binding{};
    binding.model = yolo::ModelSpec{
        .path = "unused.onnx",
        .task = yolo::TaskKind::obb,
        .class_count = 2,
        .labels = {"a", "b"},
    };
    binding.outputs = {
        yolo::adapters::ultralytics::OutputBinding{
            .index = 0,
            .name = "predictions",
            .role = yolo::adapters::ultralytics::OutputRole::predictions,
            .data_type = yolo::TensorDataType::float32,
            .shape =
                yolo::TensorShape{.dims = {yolo::TensorDimension::fixed(1),
                                           yolo::TensorDimension::fixed(2),
                                           yolo::TensorDimension::fixed(7)}}},
    };
    binding.obb = yolo::adapters::ultralytics::ObbBindingSpec{
        .layout = yolo::adapters::ultralytics::DetectionHeadLayout::
            xywh_class_scores_last,
        .proposal_count = 2,
        .class_count = 2,
        .box_coordinate_count = 4,
        .class_channel_offset = 4,
        .angle_channel_offset = 6,
        .angle_is_radians = true,
    };
    return binding;
}

TEST_CASE("obb decode contract uses binding metadata and angle conversion",
          "[component][obb]") {
    const auto spec_result =
        yolo::detail::obb_decode_spec_from_binding(make_binding());
    REQUIRE(spec_result.ok());
    CHECK(spec_result.value->proposal_count == 2);
    CHECK(spec_result.value->class_count == 2);
    CHECK(spec_result.value->box_coordinate_count == 4);
    CHECK(spec_result.value->class_channel_offset == 4);
    CHECK(spec_result.value->angle_channel_offset == 6);

    const yolo::detail::RawOutputTensors outputs{
        yolo::test::make_float_tensor(
            "predictions", {1, 2, 7},
            {4.0F, 5.0F, 2.0F, 1.0F, 0.2F, 0.9F, 1.6707963F, 2.0F, 3.0F, 3.0F,
             2.0F, 0.8F, 0.1F, 0.0F}),
    };

    const auto decoded =
        yolo::detail::decode_obb_candidates(outputs, *spec_result.value);
    REQUIRE(decoded.ok());
    REQUIRE(decoded.value->size() == 2);
    CHECK(decoded.value->at(0).class_id == 1);
    CHECK(decoded.value->at(0).score == Catch::Approx(0.9F));
    CHECK(decoded.value->at(0).box.size.width == Catch::Approx(2.0F));
    CHECK(decoded.value->at(0).box.size.height == Catch::Approx(1.0F));
    CHECK(decoded.value->at(0).box.angle_radians ==
          Catch::Approx(1.6707963F).margin(1e-5));
}

TEST_CASE("obb postprocess restores center and size to source space",
          "[component][obb]") {
    const auto boxes = yolo::detail::postprocess_obb(
        {yolo::detail::ObbCandidate{
            .box =
                yolo::OrientedBox{
                    .center = {5.0F, 7.0F},
                    .size = {4.0F, 2.0F},
                    .angle_radians = yolo::kObbPi / 6.0F,
                },
            .score = 0.95F,
            .class_id = 1,
        }},
        yolo::PreprocessRecord{
            .source_size = {8, 8},
            .target_size = {16, 16},
            .resized_size = {16, 16},
            .resize_scale = {2.0F, 2.0F},
            .padding = {.left = 1, .top = 3, .right = 1, .bottom = 3},
            .resize_mode = yolo::ResizeMode::letterbox,
        },
        yolo::ObbOptions{.confidence_threshold = 0.25F},
        yolo::ModelSpec{
            .path = "unused.onnx",
            .task = yolo::TaskKind::obb,
            .labels = {"a", "b"},
        });

    REQUIRE(boxes.size() == 1);
    CHECK(boxes[0].box.center.x == Catch::Approx(2.0F));
    CHECK(boxes[0].box.center.y == Catch::Approx(2.0F));
    CHECK(boxes[0].box.size.width == Catch::Approx(2.0F));
    CHECK(boxes[0].box.size.height == Catch::Approx(1.0F));
    CHECK(boxes[0].box.angle_degrees() == Catch::Approx(30.0F));
    REQUIRE(boxes[0].label.has_value());
    CHECK(*boxes[0].label == "b");
}

TEST_CASE("obb canonicalization regularizes angle into half-pi range",
          "[component][obb]") {
    const auto canonical = yolo::canonicalize_oriented_box(yolo::OrientedBox{
        .center = {10.0F, 12.0F},
        .size = {6.0F, 2.0F},
        .angle_radians = yolo::kObbHalfPi + 0.1F,
    });

    CHECK(canonical.size.width == Catch::Approx(2.0F));
    CHECK(canonical.size.height == Catch::Approx(6.0F));
    CHECK(canonical.angle_radians == Catch::Approx(0.1F));
}

TEST_CASE(
    "obb postprocess uses angle-aware nms instead of axis-aligned overlap",
    "[component][obb]") {
    const auto boxes = yolo::detail::postprocess_obb(
        {
            yolo::detail::ObbCandidate{
                .box = yolo::canonicalize_oriented_box(yolo::OrientedBox{
                    .center = {50.0F, 50.0F},
                    .size = {60.0F, 8.0F},
                    .angle_radians = 0.0F,
                }),
                .score = 0.95F,
                .class_id = 0,
            },
            yolo::detail::ObbCandidate{
                .box = yolo::canonicalize_oriented_box(yolo::OrientedBox{
                    .center = {50.0F, 50.0F},
                    .size = {60.0F, 8.0F},
                    .angle_radians = yolo::kObbHalfPi - 0.01F,
                }),
                .score = 0.90F,
                .class_id = 0,
            },
        },
        yolo::PreprocessRecord{
            .source_size = {100, 100},
            .target_size = {100, 100},
            .resized_size = {100, 100},
            .resize_scale = {1.0F, 1.0F},
            .resize_mode = yolo::ResizeMode::letterbox,
        },
        yolo::ObbOptions{
            .confidence_threshold = 0.25F,
            .nms_iou_threshold = 0.3F,
            .max_detections = 10,
        },
        yolo::ModelSpec{
            .path = "unused.onnx",
            .task = yolo::TaskKind::obb,
        });

    REQUIRE(boxes.size() == 2);
}

}  // namespace
