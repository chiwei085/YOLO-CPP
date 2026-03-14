#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

#include "test_utils.hpp"
#include "yolo/facade.hpp"

namespace
{

std::unique_ptr<yolo::Pipeline> require_obb_pipeline() {
    auto pipeline_result = yolo::create_pipeline(yolo::ModelSpec{
        .path = YOLO_CPP_TEST_OBB_MODEL,
        .task = yolo::TaskKind::obb,
    });
    if (!pipeline_result.ok()) {
        SKIP(pipeline_result.error.message);
    }

    return std::move(*pipeline_result.value);
}

TEST_CASE("obb pipeline runs end-to-end when model asset is present",
          "[integration][obb]") {
    auto pipeline = require_obb_pipeline();
    REQUIRE(pipeline);
    REQUIRE(pipeline->info().adapter_binding.has_value());
    REQUIRE(pipeline->info().adapter_binding->obb.has_value());

    const auto image = yolo::test::make_bgr_image(
        {1024, 1024}, std::vector<std::uint8_t>(1024 * 1024 * 3, 132));
    const auto result = pipeline->detect_obb(image.view());

    REQUIRE(result.ok());
    CHECK(result.metadata.task == yolo::TaskKind::obb);
    CHECK(result.metadata.adapter_name ==
          std::optional<std::string>{"ultralytics"});
    CHECK(result.metadata.outputs.empty() == false);
    if (!result.boxes.empty()) {
        CHECK(result.boxes.front().box.size.width >= 0.0F);
        CHECK(result.boxes.front().box.size.height >= 0.0F);
        CHECK(std::isfinite(result.boxes.front().box.angle_radians));
        CHECK(result.boxes.front().box.angle_radians >= 0.0F);
        CHECK(result.boxes.front().box.angle_radians < yolo::kObbHalfPi);
    }
}

}  // namespace
