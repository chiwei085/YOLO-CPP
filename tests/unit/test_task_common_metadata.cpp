#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "test_utils.hpp"
#include "yolo/detail/task_runtime_utils.hpp"

namespace
{

yolo::PreprocessRecord make_record() {
    return yolo::PreprocessRecord{
        .source_size = {1280, 720},
        .target_size = {640, 640},
        .resized_size = {640, 360},
        .resize_scale = {0.5F, 0.5F},
        .padding = {.left = 0, .top = 140, .right = 0, .bottom = 140},
        .source_format = yolo::PixelFormat::bgr8,
        .output_format = yolo::PixelFormat::rgb8,
        .color_conversion = yolo::ColorConversion::swap_rb,
        .resize_mode = yolo::ResizeMode::letterbox,
        .tensor_layout = yolo::TensorLayout::nchw,
    };
}

std::vector<yolo::TensorInfo> make_outputs() {
    return {
        yolo::test::make_tensor_info("predictions", yolo::TensorDataType::float32,
                                     {1, 84, 8400}),
        yolo::test::make_tensor_info("proto", yolo::TensorDataType::float32,
                                     {1, 32, 160, 160}),
    };
}

TEST_CASE("task common metadata helper produces stable shared fields",
          "[unit][task-common]") {
    yolo::ModelSpec model{
        .path = "tests/assets/models/yolov8n.onnx",
        .task = yolo::TaskKind::detect,
        .model_name = std::string{"yolov8n"},
        .adapter = std::string{"ultralytics"},
    };
    yolo::SessionOptions session{};
    session.providers = {{yolo::ExecutionProvider::cuda, 0}};

    const auto metadata = yolo::detail::make_common_metadata(
        yolo::TaskKind::detect, model, make_record(), session, make_outputs());

    CHECK(metadata.task == yolo::TaskKind::detect);
    CHECK(metadata.model_name == std::optional<std::string>{"yolov8n"});
    CHECK(metadata.adapter_name == std::optional<std::string>{"ultralytics"});
    CHECK(metadata.provider_name == std::optional<std::string>{"cuda"});
    REQUIRE(metadata.preprocess.has_value());
    CHECK(metadata.preprocess->target_size.width == 640);
    REQUIRE(metadata.original_image_size.has_value());
    CHECK(metadata.original_image_size->width == 1280);
    CHECK(metadata.original_image_size->height == 720);
    REQUIRE(metadata.outputs.size() == 2);
    CHECK(metadata.outputs.front().name == "predictions");
}

TEST_CASE("task common metadata helper supports detect classify and segmentation",
          "[unit][task-common]") {
    yolo::ModelSpec classify_model{
        .path = "tests/assets/models/yolov8n-cls.onnx",
        .task = yolo::TaskKind::classify,
        .adapter = std::string{"ultralytics"},
    };
    yolo::ModelSpec seg_model{
        .path = "tests/assets/models/yolov8n-seg.onnx",
        .task = yolo::TaskKind::seg,
        .adapter = std::string{"ultralytics"},
    };

    const auto classify_metadata = yolo::detail::make_common_metadata(
        yolo::TaskKind::classify, classify_model, make_record(), {},
        make_outputs(), yolo::ClassificationScoreSemantics::probabilities,
        yolo::ClassificationScoreSemantics::logits);
    const auto seg_metadata = yolo::detail::make_common_metadata(
        yolo::TaskKind::seg, seg_model, make_record(), {}, make_outputs());

    CHECK(classify_metadata.task == yolo::TaskKind::classify);
    CHECK(classify_metadata.provider_name == std::optional<std::string>{"cpu"});
    CHECK(classify_metadata.classification_score_semantics ==
          std::optional<yolo::ClassificationScoreSemantics>{
              yolo::ClassificationScoreSemantics::probabilities});
    CHECK(classify_metadata.source_classification_score_semantics ==
          std::optional<yolo::ClassificationScoreSemantics>{
              yolo::ClassificationScoreSemantics::logits});

    CHECK(seg_metadata.task == yolo::TaskKind::seg);
    CHECK(seg_metadata.adapter_name == std::optional<std::string>{"ultralytics"});
    REQUIRE(seg_metadata.preprocess.has_value());
    CHECK(seg_metadata.preprocess->resize_mode == yolo::ResizeMode::letterbox);
}

}  // namespace
