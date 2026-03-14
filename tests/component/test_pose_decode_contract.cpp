#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/detail/pose_runtime.hpp"

namespace
{

yolo::adapters::ultralytics::AdapterBindingSpec make_binding() {
    yolo::adapters::ultralytics::AdapterBindingSpec binding{};
    binding.model = yolo::ModelSpec{
        .path = "unused.onnx",
        .task = yolo::TaskKind::pose,
        .class_count = 1,
        .labels = {"person"},
    };
    binding.outputs = {
        yolo::adapters::ultralytics::OutputBinding{
            .index = 0,
            .name = "aux",
            .role = yolo::adapters::ultralytics::OutputRole::predictions,
            .data_type = yolo::TensorDataType::float32,
            .shape =
                yolo::TensorShape{.dims = {yolo::TensorDimension::fixed(1),
                                           yolo::TensorDimension::fixed(2),
                                           yolo::TensorDimension::fixed(11)}}},
    };
    binding.pose = yolo::adapters::ultralytics::PoseBindingSpec{
        .layout = yolo::adapters::ultralytics::DetectionHeadLayout::
            xywh_class_scores_last,
        .proposal_count = 2,
        .class_count = 1,
        .keypoint_count = 2,
        .keypoint_dimension = 3,
        .keypoint_semantic =
            yolo::adapters::ultralytics::PoseKeypointSemantic::xyscore,
    };
    return binding;
}

TEST_CASE("pose decode contract uses binding metadata and keypoint layout",
          "[component][pose]") {
    const auto spec_result =
        yolo::detail::pose_decode_spec_from_binding(make_binding());
    REQUIRE(spec_result.ok());
    CHECK(spec_result.value->output_index == 0);
    CHECK(spec_result.value->proposal_count == 2);
    CHECK(spec_result.value->class_count == 1);
    CHECK(spec_result.value->keypoint_count == 2);
    CHECK(spec_result.value->keypoint_dimension == 3);

    const yolo::detail::RawOutputTensors outputs{
        yolo::test::make_float_tensor(
            "predictions", {1, 2, 11},
            {4.0F, 4.0F, 2.0F, 2.0F, 0.9F, 3.0F, 3.0F, 0.8F, 5.0F, 5.0F, 0.7F,
             2.0F, 2.0F, 1.0F, 1.0F, 0.4F, 1.5F, 1.5F, 0.5F, 2.5F, 2.5F, 0.4F}),
    };

    const auto decoded =
        yolo::detail::decode_poses(outputs, *spec_result.value);
    REQUIRE(decoded.ok());
    REQUIRE(decoded.value->size() == 2);
    CHECK(decoded.value->at(0).score == Catch::Approx(0.9F));
    CHECK(decoded.value->at(0).class_id == 0);
    REQUIRE(decoded.value->at(0).keypoints.size() == 2);
    CHECK(decoded.value->at(0).keypoints[0].point.x == Catch::Approx(3.0F));
    CHECK(decoded.value->at(0).keypoints[0].score == Catch::Approx(0.8F));
}

TEST_CASE("pose postprocess restores bbox and keypoints to source space",
          "[component][pose]") {
    const auto poses = yolo::detail::postprocess_poses(
        {yolo::detail::PoseCandidate{
            .bbox = {3.0F, 4.0F, 4.0F, 2.0F},
            .score = 0.95F,
            .class_id = 0,
            .keypoints =
                {
                    yolo::PoseKeypoint{
                        .score = 0.8F,
                        .visible = true,
                        .point = {5.0F, 6.0F},
                    },
                    yolo::PoseKeypoint{
                        .score = 0.7F,
                        .visible = true,
                        .point = {7.0F, 8.0F},
                    },
                },
        }},
        yolo::PreprocessRecord{
            .source_size = {8, 8},
            .target_size = {16, 16},
            .resized_size = {16, 16},
            .resize_scale = {2.0F, 2.0F},
            .padding = {.left = 1, .top = 2, .right = 1, .bottom = 2},
            .resize_mode = yolo::ResizeMode::letterbox,
        },
        yolo::PoseOptions{.confidence_threshold = 0.25F},
        yolo::ModelSpec{
            .path = "unused.onnx",
            .task = yolo::TaskKind::pose,
            .labels = {"person"},
        });

    REQUIRE(poses.size() == 1);
    CHECK(poses[0].bbox.x == Catch::Approx(1.0F));
    CHECK(poses[0].bbox.y == Catch::Approx(1.0F));
    CHECK(poses[0].bbox.width == Catch::Approx(2.0F));
    CHECK(poses[0].bbox.height == Catch::Approx(1.0F));
    REQUIRE(poses[0].label.has_value());
    CHECK(*poses[0].label == "person");
    REQUIRE(poses[0].keypoints.size() == 2);
    CHECK(poses[0].keypoints[0].point.x == Catch::Approx(2.0F));
    CHECK(poses[0].keypoints[0].point.y == Catch::Approx(2.0F));
}

}  // namespace
