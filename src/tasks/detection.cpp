#include "yolo/tasks/detection.hpp"

#include <memory>
#include <utility>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/detection_runtime.hpp"
#include "yolo/detail/engine.hpp"
#include "yolo/detail/image_preprocess.hpp"
#include "yolo/detail/task_runtime_utils.hpp"

namespace yolo
{
namespace
{
using adapters::ultralytics::AdapterBindingSpec;
InferenceMetadata make_metadata(const detail::RuntimeEngine& engine,
                                const detail::PreprocessedImage& preprocessed,
                                const SessionOptions& session) {
    return detail::make_task_metadata(TaskKind::detect, engine, preprocessed,
                                      session);
}

class RuntimeDetector final : public Detector
{
public:
    RuntimeDetector(AdapterBindingSpec binding, SessionOptions session,
                    DetectionOptions options,
                    std::shared_ptr<detail::RuntimeEngine> engine)
        : binding_(std::move(binding)),
          spec_(binding_.model),
          session_(std::move(session)),
          options_(std::move(options)),
          engine_(std::move(engine)) {
        if (!engine_) {
            init_error_ = make_error(
                ErrorCode::invalid_state,
                "Detection runtime requires a valid shared engine.",
                ErrorContext{.component = std::string{"detection"}});
            return;
        }

        auto decode_spec_result =
            detail::detection_decode_spec_from_binding(binding_);
        if (!decode_spec_result.ok()) {
            init_error_ = std::move(decode_spec_result.error);
            return;
        }

        decode_spec_ = *decode_spec_result.value;
    }

    RuntimeDetector(ModelSpec spec, SessionOptions session,
                    DetectionOptions options, Error init_error)
        : spec_(std::move(spec)),
          session_(std::move(session)),
          options_(std::move(options)),
          init_error_(std::move(init_error)) {}

    const ModelSpec& model() const noexcept override { return spec_; }

    DetectionResult run(const ImageView& image) const override {
        if (!init_error_.ok()) {
            return DetectionResult{
                {}, InferenceMetadata{.task = TaskKind::detect}, init_error_};
        }

        auto input_info_result =
            detail::select_primary_input(*engine_, "detection");
        if (!input_info_result.ok()) {
            return DetectionResult{{},
                                   InferenceMetadata{.task = TaskKind::detect},
                                   input_info_result.error};
        }

        auto preprocess_result = detail::preprocess_image(
            image, binding_.preprocess, input_info_result.value->name);
        if (!preprocess_result.ok()) {
            return DetectionResult{{},
                                   InferenceMetadata{.task = TaskKind::detect},
                                   preprocess_result.error};
        }

        detail::RawInputTensor input{
            .info = preprocess_result.value->tensor.info,
            .bytes = preprocess_result.value->tensor.bytes(),
        };
        auto outputs_result = engine_->run(input);
        if (!outputs_result.ok()) {
            return DetectionResult{
                {},
                make_metadata(*engine_, *preprocess_result.value, session_),
                outputs_result.error};
        }

        auto decoded_result =
            detail::decode_detections(*outputs_result.value, decode_spec_);
        if (!decoded_result.ok()) {
            return DetectionResult{
                {},
                make_metadata(*engine_, *preprocess_result.value, session_),
                decoded_result.error};
        }

        return DetectionResult{
            .detections = detail::postprocess_detections(
                std::move(*decoded_result.value),
                preprocess_result.value->record, options_, spec_),
            .metadata =
                make_metadata(*engine_, *preprocess_result.value, session_),
            .error = {},
        };
    }

private:
    AdapterBindingSpec binding_{};
    ModelSpec spec_;
    SessionOptions session_;
    DetectionOptions options_;
    std::shared_ptr<detail::RuntimeEngine> engine_{};
    detail::DetectionDecodeSpec decode_spec_{};
    Error init_error_{};
};

}  // namespace

std::unique_ptr<Detector> create_detector(ModelSpec spec,
                                          SessionOptions session,
                                          DetectionOptions options) {
    spec.task = TaskKind::detect;
    auto binding_result =
        adapters::ultralytics::probe_detection_model(spec, session);
    if (!binding_result.ok()) {
        return std::make_unique<RuntimeDetector>(
            std::move(spec), std::move(session), std::move(options),
            std::move(binding_result.error));
    }

    auto engine_result =
        detail::RuntimeEngine::create(binding_result.value->model, session);
    if (!engine_result.ok()) {
        return std::make_unique<RuntimeDetector>(
            binding_result.value->model, std::move(session), std::move(options),
            std::move(engine_result.error));
    }

    return std::make_unique<RuntimeDetector>(
        std::move(*binding_result.value), std::move(session),
        std::move(options),
        std::shared_ptr<detail::RuntimeEngine>(std::move(*engine_result.value)));
}

namespace detail
{

std::unique_ptr<Detector> create_detector_with_engine(
    AdapterBindingSpec binding, SessionOptions session, DetectionOptions options,
    std::shared_ptr<RuntimeEngine> engine) {
    binding.model.task = TaskKind::detect;
    return std::make_unique<RuntimeDetector>(std::move(binding),
                                             std::move(session),
                                             std::move(options),
                                             std::move(engine));
}

}  // namespace detail

}  // namespace yolo
