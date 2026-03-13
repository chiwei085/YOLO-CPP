#include <catch2/catch_test_macros.hpp>

#include "test_utils.hpp"
#include "yolo/detail/pipeline_info_utils.hpp"

namespace
{

TEST_CASE("pipeline info exposes adapter binding and preprocess metadata",
          "[unit][pipeline]") {
    yolo::adapters::ultralytics::AdapterBindingSpec binding{};
    binding.model = yolo::ModelSpec{
        .path = "tests/assets/models/yolov8n.onnx",
        .task = yolo::TaskKind::detect,
        .adapter = std::string{"ultralytics"},
    };
    binding.preprocess = yolo::make_detection_preprocess_policy({640, 640});
    binding.outputs.push_back({
        .index = 0,
        .name = "output0",
        .role = yolo::adapters::ultralytics::OutputRole::predictions,
        .data_type = yolo::TensorDataType::float32,
        .shape = yolo::TensorShape{
            .dims = {yolo::TensorDimension::fixed(1),
                     yolo::TensorDimension::fixed(84),
                     yolo::TensorDimension::fixed(8400)}},
    });

    yolo::detail::SessionDescription description{};
    description.inputs.push_back(
        yolo::test::make_tensor_info("images", yolo::TensorDataType::float32,
                                     {1, 3, 640, 640}));
    description.outputs.push_back(
        yolo::test::make_tensor_info("output0", yolo::TensorDataType::float32,
                                     {1, 84, 8400}));

    const auto info = yolo::detail::make_pipeline_info(binding, description);

    REQUIRE(info.adapter_binding.has_value());
    CHECK(info.model.task == yolo::TaskKind::detect);
    REQUIRE(info.preprocess.has_value());
    CHECK(info.preprocess->target_size.width == 640);
    REQUIRE(info.adapter_binding->outputs.size() == 1);
    CHECK(info.adapter_binding->outputs.front().name == "output0");
  }

}  // namespace
