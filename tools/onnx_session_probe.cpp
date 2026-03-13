#include <algorithm>
#include <iostream>
#include <string_view>

#include "yolo/core/model_spec.hpp"
#include "yolo/core/session_options.hpp"
#include "yolo/detail/onnx_session.hpp"

namespace
{

int usage(std::string_view argv0) {
    std::cerr << "usage: " << argv0 << " <model.onnx> [<model.onnx> ...]\n";
    return 2;
}

void print_error(const yolo::Error& error) {
    std::cerr << error.message << '\n';
    if (!error.context.has_value()) {
        return;
    }

    const yolo::ErrorContext& context = *error.context;
    if (context.component.has_value()) {
        std::cerr << "component=" << *context.component << '\n';
    }
    if (context.input_name.has_value()) {
        std::cerr << "input=" << *context.input_name << '\n';
    }
    if (context.output_name.has_value()) {
        std::cerr << "output=" << *context.output_name << '\n';
    }
    if (context.expected.has_value()) {
        std::cerr << "expected=" << *context.expected << '\n';
    }
    if (context.actual.has_value()) {
        std::cerr << "actual=" << *context.actual << '\n';
    }
}

int probe_model(const char* model_path) {
    yolo::ModelSpec model{
        .path = model_path,
        .task = yolo::TaskKind::detect,
    };
    yolo::SessionOptions session{};

    const auto result = yolo::detail::OnnxSession::create(model, session);
    if (!result.ok()) {
        std::cerr << "FAIL " << model_path << '\n';
        print_error(result.error);
        return 1;
    }

    const auto& description = (*result.value)->description();
    std::cout << "OK  " << model_path << " inputs=" << description.inputs.size()
              << " outputs=" << description.outputs.size() << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        return usage(argc > 0 ? argv[0] : "onnx_session_probe");
    }

    int exit_code = 0;
    for (int index = 1; index < argc; ++index) {
        exit_code = std::max(exit_code, probe_model(argv[index]));
    }
    return exit_code;
}
