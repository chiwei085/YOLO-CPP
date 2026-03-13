#pragma once

#include <memory>
#include <vector>

#include "yolo/core/error.hpp"
#include "yolo/core/model_spec.hpp"
#include "yolo/core/session_options.hpp"
#include "yolo/detail/tensor_utils.hpp"

namespace yolo::detail
{

struct SessionDescription
{
    std::vector<TensorInfo> inputs{};
    std::vector<TensorInfo> outputs{};
};

class OnnxSession
{
public:
    virtual ~OnnxSession() = default;

    [[nodiscard]] virtual const SessionDescription& description()
        const noexcept = 0;
    [[nodiscard]] virtual Result<RawOutputTensors> run(
        const RawInputTensor& input) const = 0;

    [[nodiscard]] static Result<std::unique_ptr<OnnxSession>> create(
        const ModelSpec& spec, const SessionOptions& session_options);
};

}  // namespace yolo::detail
