#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "test_utils.hpp"
#include "yolo/facade.hpp"

namespace
{

std::unique_ptr<yolo::Pipeline> require_classify_pipeline() {
    auto pipeline_result = yolo::create_pipeline(
        yolo::ModelSpec{
            .path = YOLO_CPP_TEST_CLASSIFY_MODEL,
            .task = yolo::TaskKind::classify,
        });
    if (!pipeline_result.ok()) {
        SKIP(pipeline_result.error.message);
    }

    return std::move(*pipeline_result.value);
}

TEST_CASE("classify pipeline runs end-to-end when model asset is present",
          "[integration][classify]") {
    auto pipeline = require_classify_pipeline();
    REQUIRE(pipeline);
    REQUIRE(pipeline->info().adapter_binding.has_value());

    const auto image = yolo::test::make_bgr_image(
        {224, 224}, std::vector<std::uint8_t>(224 * 224 * 3, 96));
    const auto result = pipeline->classify(image.view());

    REQUIRE(result.ok());
    CHECK(result.metadata.task == yolo::TaskKind::classify);
    CHECK(result.metadata.classification_score_semantics ==
          std::optional<yolo::ClassificationScoreSemantics>{
              yolo::ClassificationScoreSemantics::probabilities});
    CHECK(result.scores.empty() == false);
    CHECK(result.classes.empty() == false);
}

}  // namespace
