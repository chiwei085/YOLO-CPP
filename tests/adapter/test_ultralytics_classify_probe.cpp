#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/adapters/ultralytics.hpp"

namespace
{

TEST_CASE("ultralytics classify probe infers logits semantics from output name",
          "[adapter][classify]") {
    const auto result = yolo::adapters::ultralytics::probe_classification(
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::classify},
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 3, 224, 224})},
        {yolo::test::make_tensor_info("class_logits",
                                      yolo::TensorDataType::float32, {1, 1000})});

    REQUIRE(result.ok());
    REQUIRE(result.value->classification.has_value());
    CHECK(result.value->model.task == yolo::TaskKind::classify);
    CHECK(result.value->model.class_count == 1000);
    CHECK(result.value->classification->score_kind ==
          yolo::adapters::ultralytics::ClassificationScoreKind::logits);
    CHECK(result.value->preprocess.resize_mode == yolo::ResizeMode::resize_crop);
    CHECK(result.value->preprocess.normalize.mean[0] == Catch::Approx(0.0F));
    CHECK(result.value->preprocess.normalize.mean[1] == Catch::Approx(0.0F));
    CHECK(result.value->preprocess.normalize.mean[2] == Catch::Approx(0.0F));
    CHECK(result.value->preprocess.normalize.std[0] == Catch::Approx(1.0F));
    CHECK(result.value->preprocess.normalize.std[1] == Catch::Approx(1.0F));
    CHECK(result.value->preprocess.normalize.std[2] == Catch::Approx(1.0F));
}

TEST_CASE("ultralytics classify probe rejects non-image inputs",
          "[adapter][classify]") {
    const auto result = yolo::adapters::ultralytics::probe_classification(
        yolo::ModelSpec{.path = "unused.onnx", .task = yolo::TaskKind::classify},
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 1, 224, 224})},
        {yolo::test::make_tensor_info("output0", yolo::TensorDataType::float32,
                                      {1, 1000})});

    CHECK_FALSE(result.ok());
    CHECK(result.error.code == yolo::ErrorCode::unsupported_model);
}

}  // namespace
