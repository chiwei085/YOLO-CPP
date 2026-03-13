#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "test_utils.hpp"
#include "yolo/facade.hpp"

namespace
{

std::unique_ptr<yolo::Pipeline> require_detect_pipeline() {
    auto pipeline_result = yolo::create_pipeline(
        yolo::ModelSpec{
            .path = YOLO_CPP_TEST_DETECT_MODEL,
            .task = yolo::TaskKind::detect,
        });
    if (!pipeline_result.ok()) {
        SKIP(pipeline_result.error.message);
    }

    return std::move(*pipeline_result.value);
}

TEST_CASE("detect pipeline runs end-to-end when model asset is present",
          "[integration][detect]") {
    auto pipeline = require_detect_pipeline();
    REQUIRE(pipeline);
    REQUIRE(pipeline->info().adapter_binding.has_value());

    const auto image = yolo::test::make_bgr_image(
        {640, 640}, std::vector<std::uint8_t>(640 * 640 * 3, 127));
    const auto result = pipeline->detect(image.view());

    REQUIRE(result.ok());
    CHECK(result.metadata.task == yolo::TaskKind::detect);
    CHECK(result.metadata.adapter_name == std::optional<std::string>{"ultralytics"});
    CHECK(result.metadata.outputs.empty() == false);
}

}  // namespace
