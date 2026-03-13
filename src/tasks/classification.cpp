#include "yolo/tasks/classification.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "yolo/detail/engine.hpp"
#include "yolo/detail/image_preprocess.hpp"

namespace yolo
{
namespace
{

std::string provider_name_from_options(const SessionOptions& options) {
    if (options.providers.empty()) {
        return "cpu";
    }

    switch (options.providers.front().provider) {
        case ExecutionProvider::cpu:
            return "cpu";
        case ExecutionProvider::cuda:
            return "cuda";
        case ExecutionProvider::tensorrt:
            return "tensorrt";
    }

    return "unknown";
}

Result<TensorInfo> select_primary_input(const detail::RuntimeEngine& engine) {
    const auto& inputs = engine.description().inputs;
    if (inputs.empty()) {
        return {.error = make_error(
                    ErrorCode::invalid_state,
                    "Classification engine has no input tensor metadata.",
                    ErrorContext{.component = std::string{"classification"}})};
    }

    return {.value = inputs.front(), .error = {}};
}

bool is_channel_dimension(const TensorDimension& dim) {
    return dim.value.has_value() &&
           (*dim.value == 1 || *dim.value == 3 || *dim.value == 4);
}

Result<Size2i> resolve_input_size(const ModelSpec& spec,
                                  const TensorInfo& input) {
    if (spec.input_size.has_value()) {
        return {.value = *spec.input_size, .error = {}};
    }

    if (input.shape.rank() != 4) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "Classification input tensor must be rank-4.",
                    ErrorContext{
                        .component = std::string{"classification"},
                        .input_name = input.name,
                        .expected = std::string{"[N,C,H,W] or [N,H,W,C]"},
                        .actual = detail::format_shape(input.shape),
                    })};
    }

    const auto& dims = input.shape.dims;
    if (is_channel_dimension(dims[1]) && dims[2].value.has_value() &&
        dims[3].value.has_value()) {
        return {.value = Size2i{static_cast<int>(*dims[3].value),
                                static_cast<int>(*dims[2].value)},
                .error = {}};
    }

    if (dims[1].value.has_value() && dims[2].value.has_value() &&
        is_channel_dimension(dims[3])) {
        return {.value = Size2i{static_cast<int>(*dims[2].value),
                                static_cast<int>(*dims[1].value)},
                .error = {}};
    }

    return {.error = make_error(
                ErrorCode::shape_mismatch,
                "Classification input tensor has unsupported spatial layout.",
                ErrorContext{
                    .component = std::string{"classification"},
                    .input_name = input.name,
                    .expected = std::string{"static [N,C,H,W] or [N,H,W,C]"},
                    .actual = detail::format_shape(input.shape),
                })};
}

Result<std::vector<float>> decode_classification_scores(
    const detail::RawOutputTensors& outputs) {
    if (outputs.empty()) {
        return {
            .error = make_error(
                ErrorCode::shape_mismatch,
                "Classification decoder requires at least one output tensor.",
                ErrorContext{.component =
                                 std::string{"classification_decoder"}})};
    }

    const auto values_result = detail::copy_float_tensor_data(
        outputs.front(), "classification_decoder");
    if (!values_result.ok()) {
        return {.error = values_result.error};
    }

    const TensorInfo& info = outputs.front().info;
    std::size_t class_count = 0;
    if (info.shape.rank() == 1 && info.shape.dims[0].value.has_value()) {
        class_count = static_cast<std::size_t>(*info.shape.dims[0].value);
    }
    else if (info.shape.rank() == 2 && info.shape.dims[0].value.has_value() &&
             info.shape.dims[1].value.has_value() &&
             *info.shape.dims[0].value == 1) {
        class_count = static_cast<std::size_t>(*info.shape.dims[1].value);
    }
    else if (info.shape.rank() == 3 && info.shape.dims[0].value.has_value() &&
             info.shape.dims[1].value.has_value() &&
             info.shape.dims[2].value.has_value() &&
             *info.shape.dims[0].value == 1 && *info.shape.dims[1].value == 1) {
        class_count = static_cast<std::size_t>(*info.shape.dims[2].value);
    }
    else {
        return {
            .error = make_error(
                ErrorCode::shape_mismatch,
                "Classification decoder cannot interpret output tensor shape.",
                ErrorContext{
                    .component = std::string{"classification_decoder"},
                    .output_name = info.name,
                    .expected = std::string{"[C], [1,C], or [1,1,C]"},
                    .actual = detail::format_shape(info.shape),
                })};
    }

    if (class_count > values_result.value->size()) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "Classification output shape exceeds tensor payload.",
                    ErrorContext{
                        .component = std::string{"classification_decoder"},
                        .output_name = info.name,
                        .expected = std::to_string(class_count),
                        .actual =
                            std::to_string(values_result.value->size()),
                    })};
    }

    return {
        .value = std::vector<float>(values_result.value->begin(),
                                    values_result.value->begin() + class_count),
        .error = {}};
}

std::optional<std::string> label_for(const ModelSpec& spec, ClassId class_id) {
    const std::size_t index = static_cast<std::size_t>(class_id);
    if (index < spec.labels.size()) {
        return spec.labels[index];
    }

    return std::nullopt;
}

std::vector<Classification> postprocess_classification(
    const std::vector<float>& scores, const ClassificationOptions& options,
    const ModelSpec& spec) {
    std::vector<std::size_t> indices(scores.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    const std::size_t top_k = std::min(options.top_k, indices.size());
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                      [&](std::size_t lhs, std::size_t rhs) {
                          return scores[lhs] > scores[rhs];
                      });

    std::vector<Classification> classes{};
    classes.reserve(top_k);
    for (std::size_t i = 0; i < top_k; ++i) {
        classes.push_back(Classification{
            .class_id = static_cast<ClassId>(indices[i]),
            .score = scores[indices[i]],
            .label = label_for(spec, static_cast<ClassId>(indices[i])),
        });
    }

    return classes;
}

InferenceMetadata make_metadata(const detail::RuntimeEngine& engine,
                                const detail::PreprocessedImage& preprocessed,
                                const SessionOptions& session) {
    return InferenceMetadata{
        .task = TaskKind::classify,
        .model_name = engine.model().model_name,
        .adapter_name = engine.model().adapter,
        .provider_name = provider_name_from_options(session),
        .original_image_size = preprocessed.record.source_size,
        .preprocess = preprocessed.record,
        .outputs = engine.description().outputs,
        .latency_ms = std::nullopt,
    };
}

class RuntimeClassifier final : public Classifier
{
public:
    RuntimeClassifier(ModelSpec spec, SessionOptions session,
                      ClassificationOptions options)
        : spec_(std::move(spec)),
          session_(std::move(session)),
          options_(std::move(options)) {
        auto engine_result = detail::RuntimeEngine::create(spec_, session_);
        if (engine_result.ok()) {
            engine_ = std::shared_ptr<detail::RuntimeEngine>(
                std::move(*engine_result.value));
        }
        else {
            init_error_ = std::move(engine_result.error);
        }
    }

    RuntimeClassifier(ModelSpec spec, SessionOptions session,
                      ClassificationOptions options,
                      std::shared_ptr<detail::RuntimeEngine> engine)
        : spec_(std::move(spec)),
          session_(std::move(session)),
          options_(std::move(options)),
          engine_(std::move(engine)) {
        if (!engine_) {
            init_error_ = make_error(
                ErrorCode::invalid_state,
                "Classification runtime requires a valid shared engine.",
                ErrorContext{.component = std::string{"classification"}});
        }
    }

    const ModelSpec& model() const noexcept override { return spec_; }

    ClassificationResult run(const ImageView& image) const override {
        if (!init_error_.ok()) {
            return ClassificationResult{
                .classes = {},
                .scores = {},
                .metadata = InferenceMetadata{.task = TaskKind::classify},
                .error = init_error_,
            };
        }

        auto input_info_result = select_primary_input(*engine_);
        if (!input_info_result.ok()) {
            return ClassificationResult{
                .classes = {},
                .scores = {},
                .metadata = InferenceMetadata{.task = TaskKind::classify},
                .error = input_info_result.error,
            };
        }

        auto input_size_result =
            resolve_input_size(spec_, *input_info_result.value);
        if (!input_size_result.ok()) {
            return ClassificationResult{
                .classes = {},
                .scores = {},
                .metadata = InferenceMetadata{.task = TaskKind::classify},
                .error = input_size_result.error,
            };
        }

        const PreprocessPolicy policy =
            make_classification_preprocess_policy(*input_size_result.value);
        auto preprocess_result = detail::preprocess_image(
            image, policy, input_info_result.value->name);
        if (!preprocess_result.ok()) {
            return ClassificationResult{
                .classes = {},
                .scores = {},
                .metadata = InferenceMetadata{.task = TaskKind::classify},
                .error = preprocess_result.error,
            };
        }

        detail::RawInputTensor input{
            .info = preprocess_result.value->tensor.info,
            .bytes = preprocess_result.value->tensor.bytes(),
        };
        auto outputs_result = engine_->run(input);
        if (!outputs_result.ok()) {
            return ClassificationResult{
                .classes = {},
                .scores = {},
                .metadata =
                    make_metadata(*engine_, *preprocess_result.value, session_),
                .error = outputs_result.error,
            };
        }

        auto decoded_scores_result =
            decode_classification_scores(*outputs_result.value);
        if (!decoded_scores_result.ok()) {
            return ClassificationResult{
                .classes = {},
                .scores = {},
                .metadata =
                    make_metadata(*engine_, *preprocess_result.value, session_),
                .error = decoded_scores_result.error,
            };
        }

        return ClassificationResult{
            .classes = postprocess_classification(*decoded_scores_result.value,
                                                  options_, spec_),
            .scores = *decoded_scores_result.value,
            .metadata =
                make_metadata(*engine_, *preprocess_result.value, session_),
            .error = {},
        };
    }

private:
    ModelSpec spec_;
    SessionOptions session_;
    ClassificationOptions options_;
    std::shared_ptr<detail::RuntimeEngine> engine_{};
    Error init_error_{};
};

}  // namespace

std::unique_ptr<Classifier> create_classifier(ModelSpec spec,
                                              SessionOptions session,
                                              ClassificationOptions options) {
    spec.task = TaskKind::classify;
    return std::make_unique<RuntimeClassifier>(
        std::move(spec), std::move(session), std::move(options));
}

namespace detail
{

std::unique_ptr<Classifier> create_classifier_with_engine(
    ModelSpec spec, SessionOptions session, ClassificationOptions options,
    std::shared_ptr<RuntimeEngine> engine) {
    spec.task = TaskKind::classify;
    return std::make_unique<RuntimeClassifier>(std::move(spec),
                                               std::move(session),
                                               std::move(options),
                                               std::move(engine));
}

}  // namespace detail

}  // namespace yolo
