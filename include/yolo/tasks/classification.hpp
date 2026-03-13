#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "yolo/core/error.hpp"
#include "yolo/core/image.hpp"
#include "yolo/core/model_spec.hpp"
#include "yolo/core/result.hpp"
#include "yolo/core/session_options.hpp"
#include "yolo/core/types.hpp"

namespace yolo
{

struct Classification
{
    ClassId class_id{0};
    float score{0.0F};
    std::optional<std::string> label{};
};

struct ClassificationOptions
{
    std::size_t top_k{5};
};

struct ClassificationResult
{
    std::vector<Classification> classes{};
    // Public scores are normalized probabilities for each class.
    std::vector<float> scores{};
    InferenceMetadata metadata{};
    Error error{};

    [[nodiscard]] constexpr bool ok() const noexcept { return error.ok(); }
};

class Classifier
{
public:
    virtual ~Classifier() = default;
    // RAII task handle with a stable task-specific interface.
    [[nodiscard]] virtual const ModelSpec& model() const noexcept = 0;
    virtual ClassificationResult run(const ImageView& image) const = 0;
};

[[nodiscard]] std::unique_ptr<Classifier> create_classifier(
    ModelSpec spec, SessionOptions session = {},
    ClassificationOptions options = {});

}  // namespace yolo
