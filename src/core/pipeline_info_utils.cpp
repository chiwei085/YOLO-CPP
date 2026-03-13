#include "yolo/detail/pipeline_info_utils.hpp"

namespace yolo::detail
{

PipelineInfo make_pipeline_info(
    const adapters::ultralytics::AdapterBindingSpec& binding,
    const SessionDescription& description) {
    return PipelineInfo{
        .model = binding.model,
        .inputs = description.inputs,
        .outputs = description.outputs,
        .preprocess = binding.preprocess,
        .adapter_binding = binding,
    };
}

}  // namespace yolo::detail
