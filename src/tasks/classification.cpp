#include "yolo/tasks/classification.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/engine.hpp"
#include "yolo/detail/image_preprocess.hpp"

namespace yolo
{
namespace
{
using adapters::ultralytics::AdapterBindingSpec;
using adapters::ultralytics::ClassificationBindingSpec;
using adapters::ultralytics::ClassificationScoreKind;
using adapters::ultralytics::OutputRole;

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

struct ClassificationDecodeSpec
{
    std::size_t output_index{0};
    std::size_t class_count{0};
    ClassificationScoreKind score_kind{ClassificationScoreKind::unknown};
};

Result<ClassificationDecodeSpec> classification_decode_spec_from_binding(
    const AdapterBindingSpec& binding) {
    if (!binding.classification.has_value()) {
        return {.error = make_error(
                    ErrorCode::invalid_state,
                    "Classification runtime requires a classification binding "
                    "spec.",
                    ErrorContext{
                        .component = std::string{"classification"}})};
    }

    if (binding.outputs.empty()) {
        return {.error = make_error(
                    ErrorCode::invalid_state,
                    "Classification runtime requires at least one bound "
                    "output.",
                    ErrorContext{
                        .component = std::string{"classification"}})};
    }

    std::size_t output_index = binding.outputs.front().index;
    for (const auto& output : binding.outputs) {
        if (output.role == OutputRole::predictions) {
            output_index = output.index;
            break;
        }
    }

    const ClassificationBindingSpec& classification = *binding.classification;
    return {.value = ClassificationDecodeSpec{
                .output_index = output_index,
                .class_count = classification.class_count,
                .score_kind = classification.score_kind,
            },
            .error = {}};
}

void softmax_in_place(std::vector<float>& values) {
    if (values.empty()) {
        return;
    }

    const float max_value = *std::max_element(values.begin(), values.end());
    float sum = 0.0F;
    for (float& value : values) {
        value = std::exp(value - max_value);
        sum += value;
    }

    if (sum <= 0.0F) {
        return;
    }

    for (float& value : values) {
        value /= sum;
    }
}

Result<std::vector<float>> decode_classification_scores(
    const detail::RawOutputTensors& outputs,
    const ClassificationDecodeSpec& decode_spec) {
    if (decode_spec.output_index >= outputs.size()) {
        return {
            .error = make_error(
                ErrorCode::shape_mismatch,
                "Classification binding points to a missing output tensor.",
                ErrorContext{
                    .component = std::string{"classification_decoder"},
                    .expected =
                        std::to_string(decode_spec.output_index + 1) +
                        " output tensors",
                    .actual = std::to_string(outputs.size()) +
                              " output tensors",
                })};
    }

    const auto values_result = detail::copy_float_tensor_data(
        outputs[decode_spec.output_index], "classification_decoder");
    if (!values_result.ok()) {
        return {.error = values_result.error};
    }

    const TensorInfo& info = outputs[decode_spec.output_index].info;
    if (decode_spec.class_count > values_result.value->size()) {
        return {.error = make_error(
                    ErrorCode::shape_mismatch,
                    "Classification output shape exceeds tensor payload.",
                    ErrorContext{
                        .component = std::string{"classification_decoder"},
                        .output_name = info.name,
                        .expected = std::to_string(decode_spec.class_count),
                        .actual =
                            std::to_string(values_result.value->size()),
                    })};
    }

    std::vector<float> scores(values_result.value->begin(),
                              values_result.value->begin() +
                                  decode_spec.class_count);
    if (decode_spec.score_kind == ClassificationScoreKind::logits) {
        softmax_in_place(scores);
    }

    return {
        .value = std::move(scores),
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
    RuntimeClassifier(AdapterBindingSpec binding, SessionOptions session,
                      ClassificationOptions options,
                      std::shared_ptr<detail::RuntimeEngine> engine)
        : binding_(std::move(binding)),
          spec_(binding_.model),
          session_(std::move(session)),
          options_(std::move(options)),
          engine_(std::move(engine)) {
        if (!engine_) {
            init_error_ = make_error(
                ErrorCode::invalid_state,
                "Classification runtime requires a valid shared engine.",
                ErrorContext{.component = std::string{"classification"}});
            return;
        }

        auto decode_spec_result =
            classification_decode_spec_from_binding(binding_);
        if (!decode_spec_result.ok()) {
            init_error_ = std::move(decode_spec_result.error);
            return;
        }

        decode_spec_ = *decode_spec_result.value;
    }

    RuntimeClassifier(ModelSpec spec, SessionOptions session,
                      ClassificationOptions options, Error init_error)
        : spec_(std::move(spec)),
          session_(std::move(session)),
          options_(std::move(options)),
          init_error_(std::move(init_error)) {}

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

        auto preprocess_result = detail::preprocess_image(
            image, binding_.preprocess, input_info_result.value->name);
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
            decode_classification_scores(*outputs_result.value, decode_spec_);
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
    AdapterBindingSpec binding_{};
    ModelSpec spec_;
    SessionOptions session_;
    ClassificationOptions options_;
    std::shared_ptr<detail::RuntimeEngine> engine_{};
    ClassificationDecodeSpec decode_spec_{};
    Error init_error_{};
};

}  // namespace

std::unique_ptr<Classifier> create_classifier(ModelSpec spec,
                                              SessionOptions session,
                                              ClassificationOptions options) {
    spec.task = TaskKind::classify;
    auto binding_result =
        adapters::ultralytics::probe_classification_model(spec, session);
    if (!binding_result.ok()) {
        return std::make_unique<RuntimeClassifier>(
            std::move(spec), std::move(session), std::move(options),
            std::move(binding_result.error));
    }

    auto engine_result =
        detail::RuntimeEngine::create(binding_result.value->model, session);
    if (!engine_result.ok()) {
        return std::make_unique<RuntimeClassifier>(
            binding_result.value->model, std::move(session), std::move(options),
            std::move(engine_result.error));
    }

    return std::make_unique<RuntimeClassifier>(
        std::move(*binding_result.value), std::move(session),
        std::move(options),
        std::shared_ptr<detail::RuntimeEngine>(std::move(*engine_result.value)));
}

namespace detail
{

std::unique_ptr<Classifier> create_classifier_with_engine(
    AdapterBindingSpec binding, SessionOptions session,
    ClassificationOptions options,
    std::shared_ptr<RuntimeEngine> engine) {
    binding.model.task = TaskKind::classify;
    return std::make_unique<RuntimeClassifier>(std::move(binding),
                                               std::move(session),
                                               std::move(options),
                                               std::move(engine));
}

}  // namespace detail

}  // namespace yolo
