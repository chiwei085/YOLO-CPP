#pragma once

#include <memory>

#include "yolo/core/error.hpp"
#include "yolo/core/model_spec.hpp"
#include "yolo/core/session_options.hpp"
#include "yolo/detail/onnx_session.hpp"

namespace yolo::detail
{

class RuntimeEngine
{
public:
    [[nodiscard]] static Result<std::unique_ptr<RuntimeEngine>> create(
        ModelSpec spec, SessionOptions session_options);

    [[nodiscard]] const ModelSpec& model() const noexcept { return spec_; }

    [[nodiscard]] const SessionDescription& description() const noexcept {
        return session_->description();
    }

    [[nodiscard]] Result<RawOutputTensors> run(
        const RawInputTensor& input) const;

private:
    RuntimeEngine(ModelSpec spec, SessionOptions session_options,
                  std::unique_ptr<OnnxSession> session);

    ModelSpec spec_;
    SessionOptions session_options_;
    std::unique_ptr<OnnxSession> session_;
};

}  // namespace yolo::detail
