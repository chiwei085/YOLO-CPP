#include "yolo/detail/onnx_session.hpp"

#include <array>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if YOLO_CPP_HAS_ONNXRUNTIME
#include <onnxruntime/onnxruntime_cxx_api.h>
#if YOLO_CPP_ORT_WITH_CUDA
#include <cuda_provider_factory.h>
#endif
#endif

namespace yolo::detail
{
namespace
{

std::string provider_name(ExecutionProvider provider) {
    switch (provider) {
        case ExecutionProvider::cpu:
            return "cpu";
        case ExecutionProvider::cuda:
            return "cuda";
        case ExecutionProvider::tensorrt:
            return "tensorrt";
    }

    return "unknown";
}

TensorDataType from_ort_type(
#if YOLO_CPP_HAS_ONNXRUNTIME
    ONNXTensorElementDataType data_type
#else
    int data_type
#endif
) {
#if YOLO_CPP_HAS_ONNXRUNTIME
    switch (data_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return TensorDataType::boolean;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return TensorDataType::uint8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return TensorDataType::int8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return TensorDataType::int16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return TensorDataType::int32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return TensorDataType::int64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return TensorDataType::float16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return TensorDataType::float32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return TensorDataType::float64;
        default:
            return TensorDataType::float32;
    }
#else
    static_cast<void>(data_type);
    return TensorDataType::float32;
#endif
}

#if YOLO_CPP_HAS_ONNXRUNTIME
TensorShape make_shape(const std::vector<std::int64_t>& dims) {
    TensorShape shape{};
    shape.dims.reserve(dims.size());
    for (std::int64_t dim : dims) {
        if (dim < 0) {
            shape.dims.push_back(TensorDimension::dynamic());
        }
        else {
            shape.dims.push_back(TensorDimension::fixed(dim));
        }
    }

    return shape;
}

TensorInfo make_tensor_info(Ort::Session& session,
                            Ort::AllocatorWithDefaultOptions& allocator,
                            std::size_t index, bool input) {
    Ort::AllocatedStringPtr name =
        input ? session.GetInputNameAllocated(index, allocator)
              : session.GetOutputNameAllocated(index, allocator);
    Ort::TypeInfo type_info = input ? session.GetInputTypeInfo(index)
                                    : session.GetOutputTypeInfo(index);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    return TensorInfo{
        .name = name.get(),
        .data_type = from_ort_type(tensor_info.GetElementType()),
        .shape = make_shape(tensor_info.GetShape()),
    };
}

::GraphOptimizationLevel to_ort_graph_optimization_level(
    GraphOptimizationLevel level) {
    switch (level) {
        case GraphOptimizationLevel::disable:
            return ORT_DISABLE_ALL;
        case GraphOptimizationLevel::basic:
            return ORT_ENABLE_BASIC;
        case GraphOptimizationLevel::extended:
            return ORT_ENABLE_EXTENDED;
        case GraphOptimizationLevel::all:
            return ORT_ENABLE_ALL;
    }

    return ORT_ENABLE_ALL;
}
#endif

Error unsupported_provider_error(ExecutionProvider provider) {
    return make_error(
        ErrorCode::backend_error,
        "Requested execution provider is not enabled in this build.",
        ErrorContext{
            .component = std::string{"onnx_session"},
            .expected =
                std::string{"cpu backend build or matching provider build"},
            .actual = provider_name(provider),
        });
}

class StubOnnxSession final : public OnnxSession
{
public:
    explicit StubOnnxSession(SessionDescription description)
        : description_(std::move(description)) {}

    const SessionDescription& description() const noexcept override {
        return description_;
    }

    Result<RawOutputTensors> run(const RawInputTensor&) const override {
        return {.error = make_error(
                    ErrorCode::backend_error,
                    "ONNX Runtime backend is not available on this build.",
                    ErrorContext{.component = std::string{"onnx_session"}})};
    }

private:
    SessionDescription description_{};
};

#if YOLO_CPP_HAS_ONNXRUNTIME
class OrtOnnxSession final : public OnnxSession
{
public:
    OrtOnnxSession(SessionDescription description, Ort::Env env,
                   Ort::Session session, Ort::MemoryInfo memory_info)
        : description_(std::move(description)),
          env_(std::move(env)),
          session_(std::move(session)),
          memory_info_(std::move(memory_info)) {}

    const SessionDescription& description() const noexcept override {
        return description_;
    }

    Result<RawOutputTensors> run(const RawInputTensor& input) const override {
        const std::optional<std::size_t> byte_count =
            dense_byte_count(input.info);
        if (!byte_count.has_value()) {
            return {.error =
                        make_error(ErrorCode::invalid_argument,
                                   "Input tensor must have a concrete byte "
                                   "size before inference.",
                                   ErrorContext{
                                       .component = std::string{"onnx_session"},
                                       .input_name = input.info.name,
                                       .expected = std::string{"static shape"},
                                       .actual = format_shape(input.info.shape),
                                   })};
        }

        if (input.bytes.size() != *byte_count) {
            return {
                .error = make_error(
                    ErrorCode::invalid_argument,
                    "Input tensor byte size does not match its declared shape.",
                    ErrorContext{
                        .component = std::string{"onnx_session"},
                        .input_name = input.info.name,
                        .expected = std::to_string(*byte_count),
                        .actual = std::to_string(input.bytes.size()),
                    })};
        }

        std::vector<std::int64_t> ort_shape{};
        ort_shape.reserve(input.info.shape.dims.size());
        for (const TensorDimension& dim : input.info.shape.dims) {
            if (!dim.value.has_value()) {
                return {.error = make_error(
                            ErrorCode::invalid_argument,
                            "Input tensor still contains dynamic dimensions.",
                            ErrorContext{
                                .component = std::string{"onnx_session"},
                                .input_name = input.info.name,
                            })};
            }

            ort_shape.push_back(*dim.value);
        }

        Ort::Value input_value = Ort::Value::CreateTensor(
            memory_info_,
            const_cast<void*>(static_cast<const void*>(input.bytes.data())),
            input.bytes.size(), ort_shape.data(), ort_shape.size(),
            static_cast<ONNXTensorElementDataType>(input.info.data_type));

        const char* input_name = description_.inputs.front().name.c_str();
        std::vector<const char*> output_names{};
        output_names.reserve(description_.outputs.size());
        for (const TensorInfo& output : description_.outputs) {
            output_names.push_back(output.name.c_str());
        }

        std::array<Ort::Value, 1> input_values{std::move(input_value)};
        std::vector<Ort::Value> output_values = session_.Run(
            Ort::RunOptions{nullptr}, &input_name, input_values.data(),
            input_values.size(), output_names.data(), output_names.size());

        RawOutputTensors outputs{};
        outputs.reserve(output_values.size());
        for (std::size_t i = 0; i < output_values.size(); ++i) {
            const Ort::Value& output_value = output_values[i];
            const Ort::TensorTypeAndShapeInfo tensor_info =
                output_value.GetTensorTypeAndShapeInfo();

            RawTensor raw_output{};
            raw_output.info = TensorInfo{
                .name = description_.outputs[i].name,
                .data_type = from_ort_type(tensor_info.GetElementType()),
                .shape = make_shape(tensor_info.GetShape()),
            };

            const std::size_t output_bytes =
                tensor_info.GetElementCount() *
                tensor_element_size(raw_output.info.data_type);
            raw_output.storage.resize(output_bytes);
            std::memcpy(raw_output.storage.data(),
                        output_value.GetTensorRawData(), output_bytes);
            outputs.push_back(std::move(raw_output));
        }

        return {.value = std::move(outputs), .error = {}};
    }

private:
    SessionDescription description_{};
    Ort::Env env_;
    mutable Ort::Session session_;
    Ort::MemoryInfo memory_info_;
};
#endif

}  // namespace

Result<std::unique_ptr<OnnxSession>> OnnxSession::create(
    const ModelSpec& spec, const SessionOptions& session_options) {
    if (spec.path.empty()) {
        return {.error = make_error(
                    ErrorCode::invalid_argument,
                    "Model path must not be empty.",
                    ErrorContext{.component = std::string{"onnx_session"}})};
    }

#if YOLO_CPP_HAS_ONNXRUNTIME
    try {
        for (const ExecutionProviderOptions& provider_option :
             session_options.providers) {
            if (provider_option.provider == ExecutionProvider::cpu) {
                continue;
            }
#if YOLO_CPP_ORT_WITH_CUDA
            if (provider_option.provider == ExecutionProvider::cuda) {
                continue;
            }
#endif
            return {.error =
                        unsupported_provider_error(provider_option.provider)};
        }

        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "yolo_cpp"};
        Ort::SessionOptions ort_options{};
        ort_options.SetIntraOpNumThreads(
            static_cast<int>(session_options.intra_op_threads));
        ort_options.SetInterOpNumThreads(
            static_cast<int>(session_options.inter_op_threads));
        ort_options.SetGraphOptimizationLevel(to_ort_graph_optimization_level(
            session_options.graph_optimization));
        if (!session_options.enable_memory_pattern) {
            ort_options.DisableMemPattern();
        }
        if (session_options.enable_profiling) {
            ort_options.EnableProfiling("yolo_cpp");
        }

#if YOLO_CPP_ORT_WITH_CUDA
        for (const ExecutionProviderOptions& provider_option :
             session_options.providers) {
            if (provider_option.provider != ExecutionProvider::cuda) {
                continue;
            }

            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = provider_option.device_index;
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(
                ort_options, &cuda_options));
        }
#endif

        Ort::Session session{env, spec.path.c_str(), ort_options};
        Ort::AllocatorWithDefaultOptions allocator{};

        SessionDescription description{};
        description.inputs.reserve(session.GetInputCount());
        description.outputs.reserve(session.GetOutputCount());
        for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
            description.inputs.push_back(
                make_tensor_info(session, allocator, i, true));
        }
        for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
            description.outputs.push_back(
                make_tensor_info(session, allocator, i, false));
        }

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        return Result<std::unique_ptr<OnnxSession>>{
            .value = std::unique_ptr<OnnxSession>(
                new OrtOnnxSession(std::move(description), std::move(env),
                                   std::move(session), std::move(memory_info))),
            .error = {},
        };
    }
    catch (const Ort::Exception& exception) {
        return {.error = make_error(
                    ErrorCode::backend_error, exception.what(),
                    ErrorContext{.component = std::string{"onnx_session"}})};
    }
#else
    static_cast<void>(session_options);
    return Result<std::unique_ptr<OnnxSession>>{
        .value = std::unique_ptr<OnnxSession>(
            new StubOnnxSession(SessionDescription{})),
        .error =
            make_error(ErrorCode::backend_error,
                       "ONNX Runtime backend is not available on this build.",
                       ErrorContext{.component = std::string{"onnx_session"}}),
    };
#endif
}

}  // namespace yolo::detail
