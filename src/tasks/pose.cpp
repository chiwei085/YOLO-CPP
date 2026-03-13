#include "yolo/tasks/pose.hpp"

#include <memory>
#include <utility>

namespace yolo
{
namespace
{

class StubPoseEstimator final : public PoseEstimator
{
public:
    StubPoseEstimator(ModelSpec spec, SessionOptions session,
                      PoseOptions options)
        : spec_(std::move(spec)),
          session_(std::move(session)),
          options_(std::move(options)) {}

    const ModelSpec& model() const noexcept override { return spec_; }

    PoseResult run(const ImageView&) const override {
        return PoseResult{
            {},
            InferenceMetadata{.task = TaskKind::pose},
            make_error(ErrorCode::not_implemented,
                       "Pose pipeline is not implemented yet."),
        };
    }

private:
    ModelSpec spec_;
    SessionOptions session_;
    PoseOptions options_;
};

}  // namespace

std::unique_ptr<PoseEstimator> create_pose_estimator(ModelSpec spec,
                                                     SessionOptions session,
                                                     PoseOptions options) {
    spec.task = TaskKind::pose;
    return std::make_unique<StubPoseEstimator>(
        std::move(spec), std::move(session), std::move(options));
}

}  // namespace yolo
