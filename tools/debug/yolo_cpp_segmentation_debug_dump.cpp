#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ppm_loader/image_ppm.hpp"
#include "support/cli.hpp"
#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/engine.hpp"
#include "yolo/detail/image_preprocess.hpp"
#include "yolo/detail/segmentation_runtime.hpp"
#include "yolo/detail/task_runtime_utils.hpp"

namespace
{

using yolo::ClassId;
using yolo::ModelSpec;
using yolo::Padding2i;
using yolo::Point2f;
using yolo::PreprocessRecord;
using yolo::RectF;
using yolo::ResizeMode;
using yolo::SessionOptions;
using yolo::Size2i;
using yolo::TaskKind;
using yolo::adapters::ultralytics::AdapterBindingSpec;
using yolo::detail::ProtoMaskTensor;
using yolo::detail::SegmentationCandidate;

std::string escape_json(std::string_view value) {
    std::string escaped{};
    escaped.reserve(value.size() + 8);
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            default:
                escaped.push_back(ch);
                break;
        }
    }

    return escaped;
}

void write_float_list(std::ostream& stream, const std::vector<float>& values) {
    stream << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        stream << values[i];
    }
    stream << ']';
}

void write_uint8_list(std::ostream& stream,
                      const std::vector<std::uint8_t>& values) {
    stream << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        stream << static_cast<int>(values[i]);
    }
    stream << ']';
}

void write_int_list(std::ostream& stream, const std::vector<int>& values) {
    stream << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        stream << values[i];
    }
    stream << ']';
}

void write_size(std::ostream& stream, const Size2i& size) {
    stream << '[' << size.width << ',' << size.height << ']';
}

void write_padding(std::ostream& stream, const Padding2i& padding) {
    stream << "{\"left\":" << padding.left << ",\"top\":" << padding.top
           << ",\"right\":" << padding.right << ",\"bottom\":" << padding.bottom
           << '}';
}

void write_bbox_xywh(std::ostream& stream, const RectF& bbox) {
    stream << '[' << bbox.x << ',' << bbox.y << ',' << bbox.width << ','
           << bbox.height << ']';
}

void write_bbox_xyxy(std::ostream& stream, const RectF& bbox) {
    stream << '[' << bbox.x << ',' << bbox.y << ',' << (bbox.x + bbox.width)
           << ',' << (bbox.y + bbox.height) << ']';
}

std::vector<std::size_t> encode_rle(const std::vector<std::uint8_t>& mask) {
    std::vector<std::size_t> runs{};
    if (mask.empty()) {
        return runs;
    }

    std::uint8_t current = 0;
    std::size_t count = 0;
    for (const std::uint8_t value : mask) {
        if (value == current) {
            ++count;
            continue;
        }

        runs.push_back(count);
        current = value;
        count = 1;
    }
    runs.push_back(count);
    return runs;
}

RectF xywh_to_rect(float cx, float cy, float w, float h) {
    return RectF{
        .x = cx - w * 0.5F,
        .y = cy - h * 0.5F,
        .width = w,
        .height = h,
    };
}

RectF rect_to_xyxy(const RectF& rect) {
    return RectF{
        .x = rect.x,
        .y = rect.y,
        .width = rect.x + rect.width,
        .height = rect.y + rect.height,
    };
}

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
        unmapped.x = (unmapped.x - static_cast<float>(record.padding.left)) /
                     record.resize_scale.x;
        unmapped.y = (unmapped.y - static_cast<float>(record.padding.top)) /
                     record.resize_scale.y;
        unmapped.width /= record.resize_scale.x;
        unmapped.height /= record.resize_scale.y;
    }
    else if (record.resize_mode == ResizeMode::resize_crop && record.crop) {
        unmapped.x = (unmapped.x + static_cast<float>(record.crop->x)) /
                     record.resize_scale.x;
        unmapped.y = (unmapped.y + static_cast<float>(record.crop->y)) /
                     record.resize_scale.y;
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

Point2f map_source_to_model(float source_x, float source_y,
                            const PreprocessRecord& record) {
    if (record.resize_mode == ResizeMode::letterbox) {
        return Point2f{
            .x = (source_x + 0.5F) * record.resize_scale.x - 0.5F +
                 static_cast<float>(record.padding.left),
            .y = (source_y + 0.5F) * record.resize_scale.y - 0.5F +
                 static_cast<float>(record.padding.top),
        };
    }

    if (record.resize_mode == ResizeMode::resize_crop && record.crop) {
        return Point2f{
            .x = (source_x + 0.5F) * record.resize_scale.x - 0.5F -
                 static_cast<float>(record.crop->x),
            .y = (source_y + 0.5F) * record.resize_scale.y - 0.5F -
                 static_cast<float>(record.crop->y),
        };
    }

    return Point2f{
        .x = (source_x + 0.5F) * record.resize_scale.x - 0.5F,
        .y = (source_y + 0.5F) * record.resize_scale.y - 0.5F,
    };
}

std::size_t proto_index(const ProtoMaskTensor& proto, std::size_t channel,
                        int y, int x) {
    return (channel * static_cast<std::size_t>(proto.size.height) +
            static_cast<std::size_t>(y)) *
               static_cast<std::size_t>(proto.size.width) +
           static_cast<std::size_t>(x);
}

std::vector<float> project_mask_logits(const SegmentationCandidate& candidate,
                                       const ProtoMaskTensor& proto) {
    std::vector<float> logits(
        static_cast<std::size_t>(proto.size.width * proto.size.height), 0.0F);
    if (candidate.mask_coefficients.size() != proto.channel_count ||
        proto.channel_count == 0 || proto.size.empty()) {
        return logits;
    }

    for (int y = 0; y < proto.size.height; ++y) {
        for (int x = 0; x < proto.size.width; ++x) {
            float value = 0.0F;
            for (std::size_t channel = 0; channel < proto.channel_count;
                 ++channel) {
                value += candidate.mask_coefficients[channel] *
                         proto.values[proto_index(proto, channel, y, x)];
            }
            logits[static_cast<std::size_t>(y * proto.size.width + x)] = value;
        }
    }

    return logits;
}

struct CandidateSnapshot
{
    RectF bbox{};
    RectF source_bbox{};
    float score{0.0F};
    ClassId class_id{0};
    std::vector<float> mask_coefficients{};
};

std::vector<CandidateSnapshot> snapshot_candidates(
    const std::vector<SegmentationCandidate>& candidates,
    const PreprocessRecord& record) {
    std::vector<CandidateSnapshot> snapshots{};
    snapshots.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        snapshots.push_back(CandidateSnapshot{
            .bbox = candidate.bbox,
            .source_bbox = unmap_rect(candidate.bbox, record),
            .score = candidate.score,
            .class_id = candidate.class_id,
            .mask_coefficients = candidate.mask_coefficients,
        });
    }
    return snapshots;
}

void write_candidate_list(std::ostream& stream,
                          const std::vector<CandidateSnapshot>& candidates) {
    stream << '[';
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        const auto& candidate = candidates[i];
        stream << "{\"class_id\":" << candidate.class_id
               << ",\"score\":" << candidate.score << ",\"bbox_xywh\":";
        write_bbox_xywh(stream, candidate.bbox);
        stream << ",\"bbox_xyxy\":";
        write_bbox_xyxy(stream, candidate.bbox);
        stream << ",\"source_bbox_xywh\":";
        write_bbox_xywh(stream, candidate.source_bbox);
        stream << ",\"mask_coefficients\":";
        write_float_list(stream, candidate.mask_coefficients);
        stream << '}';
    }
    stream << ']';
}

std::vector<SegmentationCandidate> apply_confidence_filter(
    const std::vector<SegmentationCandidate>& input, float threshold) {
    std::vector<SegmentationCandidate> filtered{};
    filtered.reserve(input.size());
    for (const auto& candidate : input) {
        if (candidate.score < threshold || candidate.bbox.width <= 0.0F ||
            candidate.bbox.height <= 0.0F) {
            continue;
        }
        filtered.push_back(candidate);
    }

    std::sort(
        filtered.begin(), filtered.end(),
        [](const SegmentationCandidate& lhs, const SegmentationCandidate& rhs) {
            return lhs.score > rhs.score;
        });
    return filtered;
}

std::vector<SegmentationCandidate> apply_class_aware_nms(
    const std::vector<SegmentationCandidate>& candidates, float iou_threshold,
    std::size_t max_detections) {
    std::vector<SegmentationCandidate> kept{};
    kept.reserve(std::min(max_detections, candidates.size()));
    std::vector<bool> suppressed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        kept.push_back(candidates[i]);
        if (kept.size() >= max_detections) {
            break;
        }

        for (std::size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j] ||
                candidates[i].class_id != candidates[j].class_id) {
                continue;
            }

            if (intersection_over_union(candidates[i].bbox,
                                        candidates[j].bbox) >= iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return kept;
}

void write_rle(std::ostream& stream, const std::vector<std::size_t>& runs) {
    stream << '[';
    for (std::size_t i = 0; i < runs.size(); ++i) {
        if (i > 0) {
            stream << ',';
        }
        stream << runs[i];
    }
    stream << ']';
}

void write_preprocess_record(std::ostream& stream,
                             const PreprocessRecord& record) {
    stream << "{\"source_size\":";
    write_size(stream, record.source_size);
    stream << ",\"target_size\":";
    write_size(stream, record.target_size);
    stream << ",\"resized_size\":";
    write_size(stream, record.resized_size);
    stream << ",\"resize_scale\":[";
    stream << record.resize_scale.x << ',' << record.resize_scale.y << "],";
    stream << "\"padding\":";
    write_padding(stream, record.padding);
    stream << '}';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: yolo_cpp_segmentation_debug_dump <model.onnx>"
                  << " <image.ppm>\n";
        return EXIT_FAILURE;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];

    auto binding_result = yolo::adapters::ultralytics::probe_segmentation_model(
        ModelSpec{.path = model_path, .task = TaskKind::seg}, SessionOptions{});
    if (!binding_result.ok()) {
        return examples::print_error(binding_result.error);
    }

    auto engine_result = yolo::detail::RuntimeEngine::create(
        binding_result.value->model, SessionOptions{});
    if (!engine_result.ok()) {
        return examples::print_error(engine_result.error);
    }

    auto image_result = examples::load_ppm_image(image_path);
    if (!image_result.ok()) {
        return examples::print_error(image_result.error);
    }

    auto input_info_result = yolo::detail::select_primary_input(
        **engine_result.value, "segmentation");
    if (!input_info_result.ok()) {
        return examples::print_error(input_info_result.error);
    }

    auto preprocess_result = yolo::detail::preprocess_image(
        image_result.value->view(), binding_result.value->preprocess,
        input_info_result.value->name);
    if (!preprocess_result.ok()) {
        return examples::print_error(preprocess_result.error);
    }

    yolo::detail::RawInputTensor input{
        .info = preprocess_result.value->tensor.info,
        .bytes = preprocess_result.value->tensor.bytes(),
    };
    auto outputs_result = (**engine_result.value).run(input);
    if (!outputs_result.ok()) {
        return examples::print_error(outputs_result.error);
    }

    auto decode_spec_result =
        yolo::detail::segmentation_decode_spec_from_binding(
            *binding_result.value);
    if (!decode_spec_result.ok()) {
        return examples::print_error(decode_spec_result.error);
    }

    auto decoded_result = yolo::detail::decode_segmentation(
        *outputs_result.value, *decode_spec_result.value);
    if (!decoded_result.ok()) {
        return examples::print_error(decoded_result.error);
    }

    yolo::SegmentationOptions options{};
    const auto confidence_filtered = apply_confidence_filter(
        decoded_result.value->candidates, options.confidence_threshold);
    const auto nms_kept = apply_class_aware_nms(
        confidence_filtered, options.nms_iou_threshold, options.max_detections);
    const auto final_instances = yolo::detail::postprocess_segmentation(
        decoded_result.value->candidates, decoded_result.value->proto,
        preprocess_result.value->record, options, binding_result.value->model);

    std::vector<float> top_projection{};
    if (!nms_kept.empty()) {
        top_projection =
            project_mask_logits(nms_kept.front(), decoded_result.value->proto);
    }

    std::ostringstream json{};
    std::vector<int> preprocess_shape{};
    preprocess_shape.reserve(
        preprocess_result.value->tensor.info.shape.dims.size());
    for (const auto& dim : preprocess_result.value->tensor.info.shape.dims) {
        preprocess_shape.push_back(static_cast<int>(dim.value.value_or(0)));
    }

    const auto image_name =
        std::filesystem::path(image_path).filename().string();
    json << "{\"task\":\"seg\",\"image\":\"" << escape_json(image_name)
         << "\",\"preprocess\":{\"tensor_shape\":";
    write_int_list(json, preprocess_shape);
    json << ",\"tensor_values\":";
    write_float_list(json, preprocess_result.value->tensor.values);
    json << ",\"record\":";
    write_preprocess_record(json, preprocess_result.value->record);
    json << "},\"raw_outputs\":[";
    for (std::size_t i = 0; i < outputs_result.value->size(); ++i) {
        if (i > 0) {
            json << ',';
        }
        const auto& output = (*outputs_result.value)[i];
        const auto values_result =
            yolo::detail::copy_float_tensor_data(output, "segmentation_debug");
        if (!values_result.ok()) {
            return examples::print_error(values_result.error);
        }

        std::vector<int> shape{};
        shape.reserve(output.info.shape.dims.size());
        for (const auto& dim : output.info.shape.dims) {
            shape.push_back(static_cast<int>(dim.value.value_or(0)));
        }

        json << "{\"name\":\"" << escape_json(output.info.name)
             << "\",\"shape\":";
        write_int_list(json, shape);
        json << ",\"values\":";
        write_float_list(json, *values_result.value);
        json << '}';
    }
    json << "],\"decoded\":{\"candidate_count\":"
         << decoded_result.value->candidates.size() << ",\"candidates\":";
    write_candidate_list(json,
                         snapshot_candidates(decoded_result.value->candidates,
                                             preprocess_result.value->record));
    json << "},\"confidence_filtered\":{\"candidate_count\":"
         << confidence_filtered.size() << ",\"candidates\":";
    write_candidate_list(json,
                         snapshot_candidates(confidence_filtered,
                                             preprocess_result.value->record));
    json << "},\"nms\":{\"candidate_count\":" << nms_kept.size()
         << ",\"candidates\":";
    write_candidate_list(
        json, snapshot_candidates(nms_kept, preprocess_result.value->record));
    json << "},\"mask_projection\":{";
    json << "\"proto_size\":";
    write_size(json, decoded_result.value->proto.size);
    json << ",\"top_candidate_logits\":";
    write_float_list(json, top_projection);
    json << "},\"final\":{\"instances\":[";
    for (std::size_t i = 0; i < final_instances.size(); ++i) {
        if (i > 0) {
            json << ',';
        }
        const auto& instance = final_instances[i];
        const auto rle = encode_rle(instance.mask.data);
        const std::size_t area = static_cast<std::size_t>(std::accumulate(
            instance.mask.data.begin(), instance.mask.data.end(), 0U));
        json << "{\"class_id\":" << instance.class_id
             << ",\"score\":" << instance.score << ",\"bbox_xywh\":";
        write_bbox_xywh(json, instance.bbox);
        json << ",\"mask\":{\"size\":";
        write_size(json, instance.mask.size);
        json << ",\"area\":" << area << ",\"rle\":";
        write_rle(json, rle);
        json << "}}";
    }
    json << "]}}\n";

    std::cout << json.str();
    return EXIT_SUCCESS;
}
