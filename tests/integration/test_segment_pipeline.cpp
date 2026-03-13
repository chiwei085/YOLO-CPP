#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "test_utils.hpp"
#include "yolo/facade.hpp"

namespace
{

std::unique_ptr<yolo::Pipeline> require_segment_pipeline() {
    auto pipeline_result = yolo::create_pipeline(
        yolo::ModelSpec{
            .path = YOLO_CPP_TEST_SEGMENT_MODEL,
            .task = yolo::TaskKind::seg,
        });
    if (!pipeline_result.ok()) {
        SKIP(pipeline_result.error.message);
    }

    return std::move(*pipeline_result.value);
}

TEST_CASE("segment pipeline runs end-to-end when model asset is present",
          "[integration][segmentation]") {
    auto pipeline = require_segment_pipeline();
    REQUIRE(pipeline);
    REQUIRE(pipeline->info().adapter_binding.has_value());
    REQUIRE(pipeline->info().adapter_binding->segmentation.has_value());

    const auto image = yolo::test::make_bgr_image(
        {640, 640}, std::vector<std::uint8_t>(640 * 640 * 3, 110));
    const auto result = pipeline->segment(image.view());

    REQUIRE(result.ok());
    CHECK(result.metadata.task == yolo::TaskKind::seg);
    CHECK(result.metadata.adapter_name ==
          std::optional<std::string>{"ultralytics"});
    CHECK(result.metadata.outputs.size() >= 2);
    REQUIRE(result.instances.empty() == false);
    CHECK(result.instances.front().bbox.width >= 0.0F);
    CHECK(result.instances.front().mask.size.width == image.size.width);
    CHECK(result.instances.front().mask.size.height == image.size.height);
}

}  // namespace
