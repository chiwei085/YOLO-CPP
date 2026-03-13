#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/classification_runtime.hpp"

namespace
{

TEST_CASE("classification component flow maps binding to normalized top-k output",
          "[component][classification]") {
    auto binding_result = yolo::adapters::ultralytics::probe_classification(
        yolo::ModelSpec{
            .path = "unused.onnx",
            .task = yolo::TaskKind::classify,
            .labels = {"cat", "dog", "bird"},
        },
        {yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                      {1, 3, 224, 224})},
        {yolo::test::make_tensor_info("logits", yolo::TensorDataType::float32,
                                      {1, 3})});

    REQUIRE(binding_result.ok());

    auto decode_spec_result =
        yolo::detail::classification_decode_spec_from_binding(
            *binding_result.value);
    REQUIRE(decode_spec_result.ok());

    auto scores_result = yolo::detail::decode_classification_scores(
        {yolo::test::make_float_tensor("logits", {1, 3}, {1.0F, 4.0F, 2.0F})},
        *decode_spec_result.value);
    REQUIRE(scores_result.ok());

    const auto classes = yolo::detail::postprocess_classification(
        *scores_result.value, yolo::ClassificationOptions{.top_k = 2},
        binding_result.value->model);

    REQUIRE(classes.size() == 2);
    CHECK(classes[0].class_id == 1);
    REQUIRE(classes[0].label.has_value());
    CHECK(*classes[0].label == "dog");
    CHECK(classes[1].class_id == 2);
}

}  // namespace
