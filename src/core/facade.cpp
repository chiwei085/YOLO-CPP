#include "yolo/facade.hpp"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "yolo/detail/engine.hpp"
#include "yolo/detail/image_preprocess.hpp"
#include "yolo/detail/pipeline_info_utils.hpp"
#include "yolo/detail/task_factory.hpp"
#include "yolo/detail/task_runtime_utils.hpp"

namespace yolo
{
namespace
{

Error unsupported_task_error(TaskKind requested, TaskKind actual,
                             std::string_view component) {
    return make_error(
        ErrorCode::unsupported_task,
        "The loaded pipeline is bound to a different task.",
        ErrorContext{
            .component = std::string{component},
            .expected = std::to_string(static_cast<int>(requested)),
            .actual = std::to_string(static_cast<int>(actual)),
        });
}

struct ProbeOutcome
{
    adapters::ultralytics::AdapterBindingSpec binding{};
};

std::vector<TaskKind> probe_order(TaskKind hint) {
    if (hint == TaskKind::detect) {
        return {TaskKind::detect, TaskKind::seg, TaskKind::classify};
    }
    if (hint == TaskKind::classify) {
        return {TaskKind::classify, TaskKind::detect, TaskKind::seg};
    }
    if (hint == TaskKind::seg) {
        return {TaskKind::seg, TaskKind::detect, TaskKind::classify};
    }
    if (hint == TaskKind::pose) {
        return {TaskKind::pose};
    }
    if (hint == TaskKind::obb) {
        return {TaskKind::obb};
    }

    return {TaskKind::seg, TaskKind::classify, TaskKind::detect};
}

Result<ProbeOutcome> probe_ultralytics_binding(const ModelSpec& spec,
                                               const SessionOptions& session) {
    Error last_error =
        make_error(ErrorCode::unsupported_model,
                   "No supported adapter recognized this model.",
                   ErrorContext{.component = std::string{"pipeline_probe"}});

    for (const TaskKind candidate : probe_order(spec.task)) {
        Result<adapters::ultralytics::AdapterBindingSpec> result{};
        if (candidate == TaskKind::detect) {
            result =
                adapters::ultralytics::probe_detection_model(spec, session);
        }
        else if (candidate == TaskKind::classify) {
            result = adapters::ultralytics::probe_classification_model(spec,
                                                                       session);
        }
        else if (candidate == TaskKind::seg) {
            result =
                adapters::ultralytics::probe_segmentation_model(spec, session);
        }
        else {
            result.error = make_error(
                ErrorCode::unsupported_task,
                "The requested auto-binding task is not supported yet.",
                ErrorContext{.component = std::string{"pipeline_probe"}});
        }

        if (result.ok()) {
            return {.value = ProbeOutcome{.binding = *result.value},
                    .error = {}};
        }

        last_error = std::move(result.error);
    }

    return {.error = std::move(last_error)};
}

std::vector<RawOutputTensor> to_public_raw_outputs(
    const detail::RawOutputTensors& outputs) {
    std::vector<RawOutputTensor> public_outputs{};
    public_outputs.reserve(outputs.size());
    for (const auto& output : outputs) {
        public_outputs.push_back(
            RawOutputTensor{.info = output.info, .storage = output.storage});
    }

    return public_outputs;
}

class BoundPipeline final : public Pipeline
{
public:
    BoundPipeline(PipelineInfo info, SessionOptions session,
                  PipelineOptions options,
                  std::shared_ptr<detail::RuntimeEngine> raw_engine,
                  std::unique_ptr<Detector> detector,
                  std::unique_ptr<Classifier> classifier,
                  std::unique_ptr<Segmenter> segmenter,
                  std::unique_ptr<PoseEstimator> pose_estimator,
                  std::unique_ptr<OrientedDetector> obb_detector)
        : info_(std::move(info)),
          session_(std::move(session)),
          options_(std::move(options)),
          raw_engine_(std::move(raw_engine)),
          detector_(std::move(detector)),
          classifier_(std::move(classifier)),
          segmenter_(std::move(segmenter)),
          pose_estimator_(std::move(pose_estimator)),
          obb_detector_(std::move(obb_detector)) {}

    const PipelineInfo& info() const noexcept override { return info_; }

    InferenceResult run(const ImageView& image) const override {
        if (info_.model.task == TaskKind::detect) {
            return detector_->run(image);
        }
        if (info_.model.task == TaskKind::classify) {
            return classifier_->run(image);
        }
        if (info_.model.task == TaskKind::seg) {
            return segmenter_->run(image);
        }
        if (info_.model.task == TaskKind::pose) {
            return pose_estimator_->run(image);
        }
        if (info_.model.task == TaskKind::obb) {
            return obb_detector_->run(image);
        }

        return DetectionResult{
            .detections = {},
            .metadata = InferenceMetadata{.task = info_.model.task},
            .error = unsupported_task_error(info_.model.task, info_.model.task,
                                            "pipeline_run"),
        };
    }

    RawInferenceResult run_raw(const ImageView& image) const override {
        if (!raw_engine_) {
            return RawInferenceResult{
                .outputs = {},
                .metadata = InferenceMetadata{.task = info_.model.task},
                .error = make_error(
                    ErrorCode::invalid_state,
                    "Pipeline raw inference is missing a runtime engine.",
                    ErrorContext{.component = std::string{"pipeline_raw"}}),
            };
        }

        if (!info_.preprocess.has_value() || info_.inputs.empty()) {
            return RawInferenceResult{
                .outputs = {},
                .metadata = InferenceMetadata{.task = info_.model.task},
                .error = make_error(
                    ErrorCode::invalid_state,
                    "Pipeline raw inference metadata is incomplete.",
                    ErrorContext{
                        .component = std::string{"pipeline_raw"},
                        .expected = std::string{
                            "preprocess policy and at least one input"},
                    }),
            };
        }

        auto preprocess_result = detail::preprocess_image(
            image, *info_.preprocess, info_.inputs.front().name);
        if (!preprocess_result.ok()) {
            return RawInferenceResult{
                .outputs = {},
                .metadata = InferenceMetadata{.task = info_.model.task},
                .error = preprocess_result.error,
            };
        }

        detail::RawInputTensor input{
            .info = preprocess_result.value->tensor.info,
            .bytes = preprocess_result.value->tensor.bytes(),
        };
        auto outputs_result = raw_engine_->run(input);
        if (!outputs_result.ok()) {
            return RawInferenceResult{
                .outputs = {},
                .metadata = detail::make_raw_metadata(info_.model,
                                                      *preprocess_result.value,
                                                      session_, {}),
                .error = outputs_result.error,
            };
        }

        return RawInferenceResult{
            .outputs = to_public_raw_outputs(*outputs_result.value),
            .metadata =
                detail::make_raw_metadata(info_.model, *preprocess_result.value,
                                          session_, *outputs_result.value),
            .error = {},
        };
    }

    DetectionResult detect(const ImageView& image) const override {
        if (!detector_) {
            return DetectionResult{
                .detections = {},
                .metadata = InferenceMetadata{.task = TaskKind::detect},
                .error = unsupported_task_error(
                    TaskKind::detect, info_.model.task, "pipeline_detect"),
            };
        }

        return detector_->run(image);
    }

    ClassificationResult classify(const ImageView& image) const override {
        if (!classifier_) {
            return ClassificationResult{
                .classes = {},
                .scores = {},
                .metadata = InferenceMetadata{.task = TaskKind::classify},
                .error = unsupported_task_error(
                    TaskKind::classify, info_.model.task, "pipeline_classify"),
            };
        }

        return classifier_->run(image);
    }

    SegmentationResult segment(const ImageView& image) const override {
        if (!segmenter_) {
            return SegmentationResult{
                .instances = {},
                .metadata = InferenceMetadata{.task = TaskKind::seg},
                .error = unsupported_task_error(TaskKind::seg, info_.model.task,
                                                "pipeline_segment"),
            };
        }

        return segmenter_->run(image);
    }

    PoseResult estimate_pose(const ImageView& image) const override {
        if (!pose_estimator_) {
            return PoseResult{
                .poses = {},
                .metadata = InferenceMetadata{.task = TaskKind::pose},
                .error = unsupported_task_error(
                    TaskKind::pose, info_.model.task, "pipeline_pose"),
            };
        }

        return pose_estimator_->run(image);
    }

    ObbResult detect_obb(const ImageView& image) const override {
        if (!obb_detector_) {
            return ObbResult{
                .boxes = {},
                .metadata = InferenceMetadata{.task = TaskKind::obb},
                .error = unsupported_task_error(TaskKind::obb, info_.model.task,
                                                "pipeline_obb"),
            };
        }

        return obb_detector_->run(image);
    }

private:
    PipelineInfo info_{};
    SessionOptions session_{};
    PipelineOptions options_{};
    std::shared_ptr<detail::RuntimeEngine> raw_engine_{};
    std::unique_ptr<Detector> detector_{};
    std::unique_ptr<Classifier> classifier_{};
    std::unique_ptr<Segmenter> segmenter_{};
    std::unique_ptr<PoseEstimator> pose_estimator_{};
    std::unique_ptr<OrientedDetector> obb_detector_{};
};

}  // namespace

Result<std::unique_ptr<Pipeline>> Pipeline::create(ModelSpec spec,
                                                   SessionOptions session,
                                                   PipelineOptions options) {
    return create_pipeline(std::move(spec), std::move(session),
                           std::move(options));
}

Result<std::unique_ptr<Pipeline>> create_pipeline(ModelSpec spec,
                                                  SessionOptions session,
                                                  PipelineOptions options) {
    const auto probe_result = probe_ultralytics_binding(spec, session);
    if (!probe_result.ok()) {
        return {.error = probe_result.error};
    }

    const auto& binding = probe_result.value->binding;
    auto engine_result = detail::RuntimeEngine::create(binding.model, session);
    if (!engine_result.ok()) {
        return {.error = engine_result.error};
    }

    auto shared_engine =
        std::shared_ptr<detail::RuntimeEngine>(std::move(*engine_result.value));

    PipelineInfo info =
        detail::make_pipeline_info(binding, shared_engine->description());

    std::unique_ptr<Detector> detector{};
    std::unique_ptr<Classifier> classifier{};
    std::unique_ptr<Segmenter> segmenter{};
    std::unique_ptr<PoseEstimator> pose_estimator{};
    std::unique_ptr<OrientedDetector> obb_detector{};

    if (binding.model.task == TaskKind::detect) {
        detector = detail::create_detector_with_engine(
            binding, session, options.detection, shared_engine);
    }
    else if (binding.model.task == TaskKind::classify) {
        classifier = detail::create_classifier_with_engine(
            binding, session, options.classification, shared_engine);
    }
    else if (binding.model.task == TaskKind::seg) {
        segmenter =
            create_segmenter(binding.model, session, options.segmentation);
    }
    else if (binding.model.task == TaskKind::pose) {
        pose_estimator =
            create_pose_estimator(binding.model, session, options.pose);
    }
    else if (binding.model.task == TaskKind::obb) {
        obb_detector = create_obb_detector(binding.model, session, options.obb);
    }

    return {
        .value = std::unique_ptr<Pipeline>(new BoundPipeline(
            std::move(info), std::move(session), std::move(options),
            std::move(shared_engine), std::move(detector),
            std::move(classifier), std::move(segmenter),
            std::move(pose_estimator), std::move(obb_detector))),
        .error = {},
    };
}

}  // namespace yolo
