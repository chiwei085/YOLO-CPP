#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "test_utils.hpp"
#include "yolo/facade.hpp"

namespace
{

std::unique_ptr<yolo::Pipeline> require_pose_pipeline() {
    auto pipeline_result = yolo::create_pipeline(yolo::ModelSpec{
        .path = YOLO_CPP_TEST_POSE_MODEL,
        .task = yolo::TaskKind::pose,
    });
    if (!pipeline_result.ok()) {
        SKIP(pipeline_result.error.message);
    }

    return std::move(*pipeline_result.value);
}

TEST_CASE("pose pipeline runs end-to-end when model asset is present",
          "[integration][pose]") {
    auto pipeline = require_pose_pipeline();
    REQUIRE(pipeline);
    REQUIRE(pipeline->info().adapter_binding.has_value());
    REQUIRE(pipeline->info().adapter_binding->pose.has_value());

    const auto image = yolo::test::make_bgr_image(
        {640, 640}, std::vector<std::uint8_t>(640 * 640 * 3, 118));
    const auto result = pipeline->estimate_pose(image.view());

    REQUIRE(result.ok());
    CHECK(result.metadata.task == yolo::TaskKind::pose);
    CHECK(result.metadata.adapter_name ==
          std::optional<std::string>{"ultralytics"});
    CHECK(result.metadata.outputs.empty() == false);
    if (!result.poses.empty()) {
        CHECK(result.poses.front().bbox.width >= 0.0F);
        CHECK(result.poses.front().bbox.height >= 0.0F);
        CHECK(result.poses.front().keypoints.size() ==
              pipeline->info().adapter_binding->pose->keypoint_count);
    }
}

}  // namespace
