#include <catch2/catch_test_macros.hpp>

#include "yolo/facade.hpp"

namespace
{

TEST_CASE("pose auto-binding fails fast instead of falling back",
          "[adapter][constraints]") {
    const auto result = yolo::create_pipeline(
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::pose});

    CHECK_FALSE(result.ok());
    CHECK(result.error.code == yolo::ErrorCode::unsupported_task);
}

TEST_CASE("obb auto-binding fails fast instead of falling back",
          "[adapter][constraints]") {
    const auto result = yolo::create_pipeline(
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::obb});

    CHECK_FALSE(result.ok());
    CHECK(result.error.code == yolo::ErrorCode::unsupported_task);
}

}  // namespace
