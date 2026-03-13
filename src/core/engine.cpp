#include "yolo/detail/engine.hpp"

#include <memory>
#include <utility>

namespace yolo::detail
{

RuntimeEngine::RuntimeEngine(ModelSpec spec, SessionOptions session_options,
                             std::unique_ptr<OnnxSession> session)
    : spec_(std::move(spec)),
      session_options_(std::move(session_options)),
      session_(std::move(session)) {}

Result<std::unique_ptr<RuntimeEngine>> RuntimeEngine::create(
    ModelSpec spec, SessionOptions session_options) {
    Result<std::unique_ptr<OnnxSession>> session_result =
        OnnxSession::create(spec, session_options);
    if (!session_result.ok()) {
        return {.error = std::move(session_result.error)};
    }

    return Result<std::unique_ptr<RuntimeEngine>>{
        .value = std::unique_ptr<RuntimeEngine>(
            new RuntimeEngine(std::move(spec), std::move(session_options),
                              std::move(*session_result.value))),
        .error = {},
    };
}

Result<RawOutputTensors> RuntimeEngine::run(const RawInputTensor& input) const {
    const SessionDescription& description = session_->description();
    if (description.inputs.empty()) {
        return {.error = make_error(
                    ErrorCode::invalid_state, "Model has no declared inputs.",
                    ErrorContext{.component = std::string{"runtime_engine"}})};
    }

    const TensorInfo& expected_input = description.inputs.front();
    if (expected_input.data_type != input.info.data_type) {
        return {.error = make_type_error("runtime_engine", expected_input.name,
                                         expected_input.data_type,
                                         input.info.data_type)};
    }

    if (expected_input.shape.rank() != input.info.shape.rank()) {
        return {.error =
                    make_shape_error("runtime_engine", expected_input.name,
                                     expected_input.shape, input.info.shape)};
    }

    for (std::size_t i = 0; i < expected_input.shape.dims.size(); ++i) {
        const TensorDimension& expected_dim = expected_input.shape.dims[i];
        const TensorDimension& actual_dim = input.info.shape.dims[i];
        if (expected_dim.value.has_value() && actual_dim.value.has_value() &&
            *expected_dim.value != *actual_dim.value) {
            return {.error = make_shape_error(
                        "runtime_engine", expected_input.name,
                        expected_input.shape, input.info.shape)};
        }
    }

    return session_->run(input);
}

}  // namespace yolo::detail
