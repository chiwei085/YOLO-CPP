#include "yolo/detail/detection_runtime.hpp"

#include <algorithm>

namespace yolo::detail
{
namespace
{

float intersection_over_union(const RectF& a, const RectF& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);

    const float w = std::max(0.0F, x2 - x1);
    const float h = std::max(0.0F, y2 - y1);
    const float inter = w * h;
    const float denom = a.area() + b.area() - inter;
    return denom > 0.0F ? inter / denom : 0.0F;
}

RectF unmap_rect(const RectF& rect, const PreprocessRecord& record) {
    RectF unmapped = rect;
    if (record.resize_mode == ResizeMode::letterbox) {
        unmapped.x = (unmapped.x - record.padding.left) / record.resize_scale.x;
        unmapped.y = (unmapped.y - record.padding.top) / record.resize_scale.y;
        unmapped.width /= record.resize_scale.x;
        unmapped.height /= record.resize_scale.y;
    }
    else if (record.resize_mode == ResizeMode::resize_crop) {
        unmapped.x = (unmapped.x + record.crop->x) / record.resize_scale.x;
        unmapped.y = (unmapped.y + record.crop->y) / record.resize_scale.y;
        unmapped.width /= record.resize_scale.x;
        unmapped.height /= record.resize_scale.y;
    }
    else {
        unmapped.x /= record.resize_scale.x;
        unmapped.y /= record.resize_scale.y;
        unmapped.width /= record.resize_scale.x;
        unmapped.height /= record.resize_scale.y;
    }

    unmapped.x = std::clamp(unmapped.x, 0.0F,
                            static_cast<float>(record.source_size.width));
    unmapped.y = std::clamp(unmapped.y, 0.0F,
                            static_cast<float>(record.source_size.height));
    unmapped.width =
        std::clamp(unmapped.width, 0.0F,
                   static_cast<float>(record.source_size.width) - unmapped.x);
    unmapped.height =
        std::clamp(unmapped.height, 0.0F,
                   static_cast<float>(record.source_size.height) - unmapped.y);
    return unmapped;
}

std::optional<std::string> label_for(const ModelSpec& spec, ClassId class_id) {
    const std::size_t index = static_cast<std::size_t>(class_id);
    if (index < spec.labels.size()) {
        return spec.labels[index];
    }

    return std::nullopt;
}

}  // namespace

std::vector<Detection> postprocess_detections(
    std::vector<DetectionCandidate> candidates, const PreprocessRecord& record,
    const DetectionOptions& options, const ModelSpec& spec) {
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [&](const DetectionCandidate& candidate) {
                           return candidate.score <
                                      options.confidence_threshold ||
                                  candidate.bbox.width <= 0.0F ||
                                  candidate.bbox.height <= 0.0F;
                       }),
        candidates.end());

    std::sort(candidates.begin(), candidates.end(),
              [](const DetectionCandidate& lhs, const DetectionCandidate& rhs) {
                  return lhs.score > rhs.score;
              });

    std::vector<Detection> detections{};
    detections.reserve(std::min(options.max_detections, candidates.size()));
    std::vector<bool> suppressed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        detections.push_back(Detection{
            .bbox = unmap_rect(candidates[i].bbox, record),
            .score = candidates[i].score,
            .class_id = candidates[i].class_id,
            .label = label_for(spec, candidates[i].class_id),
        });
        if (detections.size() >= options.max_detections) {
            break;
        }

        for (std::size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            if (!options.class_agnostic_nms &&
                candidates[i].class_id != candidates[j].class_id) {
                continue;
            }

            if (intersection_over_union(candidates[i].bbox,
                                        candidates[j].bbox) >=
                options.nms_iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return detections;
}

}  // namespace yolo::detail
