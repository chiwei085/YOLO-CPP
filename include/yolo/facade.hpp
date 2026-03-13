#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/core/error.hpp"
#include "yolo/core/image.hpp"
#include "yolo/core/model_spec.hpp"
#include "yolo/core/result.hpp"
#include "yolo/core/session_options.hpp"
#include "yolo/core/tensor.hpp"
#include "yolo/tasks/classification.hpp"
#include "yolo/tasks/detection.hpp"
#include "yolo/tasks/obb.hpp"
#include "yolo/tasks/pose.hpp"
#include "yolo/tasks/segmentation.hpp"

namespace yolo
{

struct PipelineOptions
{
    DetectionOptions detection{};
    ClassificationOptions classification{};
    SegmentationOptions segmentation{};
    PoseOptions pose{};
    ObbOptions obb{};
};

struct RawOutputTensor
{
    TensorInfo info{};
    std::vector<std::byte> storage{};

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
        return storage;
    }
};

struct RawInferenceResult
{
    std::vector<RawOutputTensor> outputs{};
    InferenceMetadata metadata{};
    Error error{};

    [[nodiscard]] constexpr bool ok() const noexcept { return error.ok(); }
};

struct PipelineInfo
{
    ModelSpec model{};
    std::vector<TensorInfo> inputs{};
    std::vector<TensorInfo> outputs{};
    std::optional<PreprocessPolicy> preprocess{};
    // Advanced/debug introspection for adapter-derived runtime binding.
    std::optional<adapters::ultralytics::AdapterBindingSpec> adapter_binding{};
};

using InferenceResult = std::variant<DetectionResult, ClassificationResult,
                                     SegmentationResult, PoseResult, ObbResult>;

class Pipeline
{
public:
    virtual ~Pipeline() = default;

    [[nodiscard]] static Result<std::unique_ptr<Pipeline>> create(
        ModelSpec spec, SessionOptions session = {},
        PipelineOptions options = {});

    [[nodiscard]] virtual const PipelineInfo& info() const noexcept = 0;

    [[nodiscard]] virtual InferenceResult run(const ImageView& image) const = 0;
    [[nodiscard]] virtual RawInferenceResult run_raw(
        const ImageView& image) const = 0;

    [[nodiscard]] virtual DetectionResult detect(
        const ImageView& image) const = 0;
    [[nodiscard]] virtual ClassificationResult classify(
        const ImageView& image) const = 0;
    [[nodiscard]] virtual SegmentationResult segment(
        const ImageView& image) const = 0;
    [[nodiscard]] virtual PoseResult estimate_pose(
        const ImageView& image) const = 0;
    [[nodiscard]] virtual ObbResult detect_obb(
        const ImageView& image) const = 0;
};

[[nodiscard]] Result<std::unique_ptr<Pipeline>> create_pipeline(
    ModelSpec spec, SessionOptions session = {}, PipelineOptions options = {});

}  // namespace yolo
