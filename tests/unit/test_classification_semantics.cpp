#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "test_utils.hpp"
#include "yolo/detail/classification_runtime.hpp"

namespace
{

using yolo::ClassificationOptions;
using yolo::ModelSpec;
using yolo::TaskKind;
using yolo::adapters::ultralytics::AdapterBindingSpec;
using yolo::adapters::ultralytics::ClassificationBindingSpec;
using yolo::adapters::ultralytics::ClassificationScoreKind;
using yolo::adapters::ultralytics::OutputBinding;
using yolo::adapters::ultralytics::OutputRole;

TEST_CASE("classification logits are normalized to probabilities",
          "[unit][classification]") {
    const auto result = yolo::detail::decode_classification_scores(
        yolo::detail::RawOutputTensors{
            yolo::test::make_float_tensor("logits", {1, 3}, {1.0F, 2.0F, 4.0F})},
        yolo::detail::ClassificationDecodeSpec{
            .output_index = 0,
            .class_count = 3,
            .score_kind = ClassificationScoreKind::logits,
        });

    REQUIRE(result.ok());
    REQUIRE(result.value->size() == 3);
    CHECK(result.value->at(0) == Catch::Approx(0.0420101F).margin(1e-5));
    CHECK(result.value->at(1) == Catch::Approx(0.114195F).margin(1e-5));
    CHECK(result.value->at(2) == Catch::Approx(0.843795F).margin(1e-5));
}

TEST_CASE("classification probabilities are left unchanged",
          "[unit][classification]") {
    const auto result = yolo::detail::decode_classification_scores(
        yolo::detail::RawOutputTensors{
            yolo::test::make_float_tensor("probabilities", {3},
                                          {0.2F, 0.3F, 0.5F})},
        yolo::detail::ClassificationDecodeSpec{
            .output_index = 0,
            .class_count = 3,
            .score_kind = ClassificationScoreKind::probabilities,
        });

    REQUIRE(result.ok());
    CHECK(result.value->at(0) == Catch::Approx(0.2F));
    CHECK(result.value->at(1) == Catch::Approx(0.3F));
    CHECK(result.value->at(2) == Catch::Approx(0.5F));
}

TEST_CASE("classification top-k preserves descending ranking and labels",
          "[unit][classification]") {
    const auto classes = yolo::detail::postprocess_classification(
        {0.1F, 0.7F, 0.2F}, ClassificationOptions{.top_k = 2},
        ModelSpec{
            .path = "unused.onnx",
            .task = TaskKind::classify,
            .labels = {"zero", "one", "two"},
        });

    REQUIRE(classes.size() == 2);
    CHECK(classes[0].class_id == 1);
    REQUIRE(classes[0].label.has_value());
    CHECK(*classes[0].label == "one");
    CHECK(classes[1].class_id == 2);
    REQUIRE(classes[1].label.has_value());
    CHECK(*classes[1].label == "two");
}

TEST_CASE("classification decode spec prefers prediction output binding",
          "[unit][classification]") {
    AdapterBindingSpec binding{};
    binding.outputs = {
        OutputBinding{.index = 1, .name = "aux", .role = OutputRole::proto},
        OutputBinding{.index = 0,
                      .name = "predictions",
                      .role = OutputRole::predictions},
    };
    binding.classification = ClassificationBindingSpec{
        .class_count = 3,
        .score_kind = ClassificationScoreKind::unknown,
    };

    const auto result =
        yolo::detail::classification_decode_spec_from_binding(binding);

    REQUIRE(result.ok());
    CHECK(result.value->output_index == 0);
    CHECK(result.value->class_count == 3);
    CHECK(result.value->score_kind == ClassificationScoreKind::unknown);
}

}  // namespace
