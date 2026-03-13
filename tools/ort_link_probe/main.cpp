#include <onnxruntime/onnxruntime_cxx_api.h>

int main() {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ort_link_probe"};
    (void)env;
    return 0;
}
