#include <catch2/catch_test_macros.hpp>

#include "yolo/detail/task_runtime_utils.hpp"

namespace
{

TEST_CASE("task common provider helper resolves cpu and defaults consistently",
          "[unit][task-common]") {
    yolo::SessionOptions default_options{};
    default_options.providers = {};

    yolo::SessionOptions cpu_options{};
    cpu_options.providers = {{yolo::ExecutionProvider::cpu, 0}};

    CHECK(yolo::detail::provider_name(yolo::ExecutionProvider::cpu) == "cpu");
    CHECK(yolo::detail::provider_name_from_options(default_options) == "cpu");
    CHECK(yolo::detail::provider_name_from_options(cpu_options) == "cpu");
}

TEST_CASE("task common provider helper resolves explicit providers consistently",
          "[unit][task-common]") {
    yolo::SessionOptions cuda_options{};
    cuda_options.providers = {{yolo::ExecutionProvider::cuda, 0}};

    yolo::SessionOptions tensorrt_options{};
    tensorrt_options.providers = {{yolo::ExecutionProvider::tensorrt, 0}};

    yolo::SessionOptions unknown_options{};
    unknown_options.providers = {
        {static_cast<yolo::ExecutionProvider>(99), 0}};

    CHECK(yolo::detail::provider_name(yolo::ExecutionProvider::cuda) == "cuda");
    CHECK(yolo::detail::provider_name(yolo::ExecutionProvider::tensorrt) ==
          "tensorrt");
    CHECK(yolo::detail::provider_name_from_options(cuda_options) == "cuda");
    CHECK(yolo::detail::provider_name_from_options(tensorrt_options) ==
          "tensorrt");
    CHECK(yolo::detail::provider_name_from_options(unknown_options) ==
          "unknown");
}

}  // namespace
