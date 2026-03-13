#pragma once

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/onnx_session.hpp"
#include "yolo/facade.hpp"

namespace yolo::detail
{

[[nodiscard]] PipelineInfo make_pipeline_info(
    const adapters::ultralytics::AdapterBindingSpec& binding,
    const SessionDescription& description);

}  // namespace yolo::detail
