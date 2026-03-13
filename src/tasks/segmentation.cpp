#include "yolo/tasks/segmentation.hpp"

#include <memory>
#include <utility>

namespace yolo
{
namespace
{

class StubSegmenter final : public Segmenter
{
public:
    StubSegmenter(ModelSpec spec, SessionOptions session,
                  SegmentationOptions options)
        : spec_(std::move(spec)),
          session_(std::move(session)),
          options_(std::move(options)) {}

    const ModelSpec& model() const noexcept override { return spec_; }

    SegmentationResult run(const ImageView&) const override {
        return SegmentationResult{
            {},
            InferenceMetadata{.task = TaskKind::seg},
            make_error(ErrorCode::not_implemented,
                       "Segmentation pipeline is not implemented yet."),
        };
    }

private:
    ModelSpec spec_;
    SessionOptions session_;
    SegmentationOptions options_;
};

}  // namespace

std::unique_ptr<Segmenter> create_segmenter(ModelSpec spec,
                                            SessionOptions session,
                                            SegmentationOptions options) {
    spec.task = TaskKind::seg;
    return std::make_unique<StubSegmenter>(std::move(spec), std::move(session),
                                           std::move(options));
}

}  // namespace yolo
