#include "yolo/tasks/obb.hpp"

#include <memory>
#include <utility>

namespace yolo
{
namespace
{

class StubOrientedDetector final : public OrientedDetector
{
public:
    StubOrientedDetector(ModelSpec spec, SessionOptions session,
                         ObbOptions options)
        : spec_(std::move(spec)),
          session_(std::move(session)),
          options_(std::move(options)) {}

    const ModelSpec& model() const noexcept override { return spec_; }

    ObbResult run(const ImageView&) const override {
        return ObbResult{
            {},
            InferenceMetadata{.task = TaskKind::obb},
            make_error(ErrorCode::not_implemented,
                       "OBB pipeline is not implemented yet."),
        };
    }

private:
    ModelSpec spec_;
    SessionOptions session_;
    ObbOptions options_;
};

}  // namespace

std::unique_ptr<OrientedDetector> create_obb_detector(ModelSpec spec,
                                                      SessionOptions session,
                                                      ObbOptions options) {
    spec.task = TaskKind::obb;
    return std::make_unique<StubOrientedDetector>(
        std::move(spec), std::move(session), std::move(options));
}

}  // namespace yolo
