#include <catch2/catch_test_macros.hpp>

#include "yolo/facade.hpp"

namespace
{

TEST_CASE("run_raw surfaces preprocess guard failures on empty image",
          "[integration][raw]") {
    auto pipeline_result = yolo::create_pipeline(
        yolo::ModelSpec{
            .path = YOLO_CPP_TEST_DETECT_MODEL,
            .task = yolo::TaskKind::detect,
        });
    if (!pipeline_result.ok()) {
        SKIP(pipeline_result.error.message);
    }

    const auto raw = (*pipeline_result.value)->run_raw(yolo::ImageView{});

    CHECK_FALSE(raw.ok());
    CHECK(raw.error.code == yolo::ErrorCode::invalid_argument);
    CHECK(raw.outputs.empty());
}

}  // namespace
