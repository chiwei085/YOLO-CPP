#include <onnxruntime/onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace
{

int usage(std::string_view argv0) {
    std::cerr << "usage: " << argv0 << " <model.onnx> [<model.onnx> ...]\n";
    return 2;
}

int probe_model(Ort::Env& env, const char* model_path) {
    try {
        Ort::SessionOptions options{};
        Ort::Session session{env, model_path, options};

        std::cout << "OK  " << model_path
                  << " inputs=" << session.GetInputCount()
                  << " outputs=" << session.GetOutputCount() << '\n';
        return 0;
    }
    catch (const Ort::Exception& exception) {
        std::cerr << "FAIL " << model_path << '\n';
        std::cerr << exception.what() << '\n';
        return 1;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        return usage(argc > 0 ? argv[0] : "ort_session_probe");
    }

    try {
        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ort_session_probe"};
        int exit_code = 0;
        for (int index = 1; index < argc; ++index) {
            exit_code = std::max(exit_code, probe_model(env, argv[index]));
        }
        return exit_code;
    }
    catch (const Ort::Exception& exception) {
        std::cerr << "FAIL env\n" << exception.what() << '\n';
        return 1;
    }
}
