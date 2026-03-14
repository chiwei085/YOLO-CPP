#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "yolo/tasks/detection.hpp"
#include "yolo/tasks/pose.hpp"

namespace examples
{

struct OverlayStyle
{
    int line_width{2};
    int font_thickness{1};
    double font_scale{0.55};
    int badge_pad_x{3};
    int badge_pad_y{2};
    int badge_gap{1};
    int point_radius{2};
    int point_ring_thickness{1};
};

inline std::string fallback_class_name(yolo::ClassId class_id,
                                       const std::vector<std::string>& labels) {
    const std::size_t index = static_cast<std::size_t>(class_id);
    if (index < labels.size() && !labels[index].empty()) {
        return labels[index];
    }
    return "class " + std::to_string(class_id);
}

inline std::string detection_label(
    const yolo::Detection& detection,
    const std::vector<std::string>& labels = {}) {
    std::ostringstream label{};
    if (detection.label.has_value()) {
        label << *detection.label;
    }
    else {
        label << fallback_class_name(detection.class_id, labels);
    }
    label << ' ' << std::fixed << std::setprecision(2) << detection.score;
    return label.str();
}

inline std::string pose_label(const yolo::PoseDetection& pose) {
    std::ostringstream label{};
    if (pose.label.has_value()) {
        label << *pose.label;
    }
    else {
        label << "person";
    }
    label << ' ' << std::fixed << std::setprecision(2) << pose.score;
    return label.str();
}

inline cv::Rect clamp_rect(const yolo::RectF& bbox, const cv::Mat& image) {
    const int x = std::max(0, static_cast<int>(std::lround(bbox.x)));
    const int y = std::max(0, static_cast<int>(std::lround(bbox.y)));
    const int w = std::max(1, static_cast<int>(std::lround(bbox.width)));
    const int h = std::max(1, static_cast<int>(std::lround(bbox.height)));
    return cv::Rect{x, y, w, h} & cv::Rect{0, 0, image.cols, image.rows};
}

inline OverlayStyle make_overlay_style(const cv::Mat& image) {
    const double scale = std::clamp(
        static_cast<double>(std::max(image.cols, image.rows)) / 1000.0, 0.8,
        1.4);
    const int line_width =
        std::max(2, static_cast<int>(std::lround(scale * 2.0)));
    return OverlayStyle{
        .line_width = line_width,
        .font_thickness = std::max(1, line_width - 1),
        .font_scale = 0.55 * scale,
        .badge_pad_x = std::max(2, static_cast<int>(std::lround(
                                       static_cast<double>(line_width) * 1.5))),
        .badge_pad_y = std::max(1, static_cast<int>(std::lround(
                                       static_cast<double>(line_width) * 0.8))),
        .badge_gap = std::max(1, static_cast<int>(std::lround(
                                     static_cast<double>(line_width) * 0.5))),
        .point_radius = std::max(
            2, static_cast<int>(
                   std::lround(static_cast<double>(line_width) * 1.2))),
        .point_ring_thickness = std::max(1, line_width - 1),
    };
}

inline void draw_badge(cv::Mat& image, const cv::Rect& rect,
                       std::string_view text, const cv::Scalar& color,
                       const OverlayStyle& style) {
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(std::string{text}, cv::FONT_HERSHEY_SIMPLEX,
                        style.font_scale, style.font_thickness, &baseline);
    const int badge_width = text_size.width + style.badge_pad_x * 2;
    const int badge_height =
        text_size.height + baseline + style.badge_pad_y * 2;
    const int badge_x =
        std::clamp(rect.x, 0, std::max(0, image.cols - badge_width));
    const int badge_y = std::max(0, rect.y - badge_height - style.badge_gap);
    const cv::Rect badge_rect{
        badge_x,
        badge_y,
        badge_width,
        badge_height,
    };
    const int text_baseline_y =
        badge_rect.y + style.badge_pad_y + text_size.height;
    cv::rectangle(image, badge_rect, color, cv::FILLED);
    cv::putText(image, std::string{text},
                cv::Point{badge_rect.x + style.badge_pad_x, text_baseline_y},
                cv::FONT_HERSHEY_SIMPLEX, style.font_scale,
                cv::Scalar{255, 255, 255}, style.font_thickness, cv::LINE_AA);
}

inline void draw_detections(cv::Mat& image,
                            const std::vector<yolo::Detection>& detections,
                            const std::vector<std::string>& labels = {}) {
    const OverlayStyle style = make_overlay_style(image);
    const std::array<cv::Scalar, 5> palette{
        cv::Scalar{44, 95, 224},  cv::Scalar{15, 196, 241},
        cv::Scalar{80, 175, 76},  cv::Scalar{0, 159, 255},
        cv::Scalar{121, 82, 179},
    };
    for (std::size_t index = 0; index < detections.size(); ++index) {
        const auto& detection = detections[index];
        const cv::Scalar color = palette[index % palette.size()];
        const cv::Rect rect = clamp_rect(detection.bbox, image);
        cv::rectangle(image, rect, color, style.line_width, cv::LINE_AA);
        draw_badge(image, rect, detection_label(detection, labels), color,
                   style);
    }
}

inline void draw_pose_detections(
    cv::Mat& image, const std::vector<yolo::PoseDetection>& poses) {
    const OverlayStyle style = make_overlay_style(image);
    constexpr std::array<std::pair<int, int>, 17> skeleton_edges{{
        {0, 1},
        {0, 2},
        {1, 3},
        {2, 4},
        {5, 6},
        {5, 7},
        {7, 9},
        {6, 8},
        {8, 10},
        {5, 11},
        {6, 12},
        {11, 12},
        {11, 13},
        {13, 15},
        {12, 14},
        {14, 16},
        {0, 5},
    }};

    for (const auto& pose : poses) {
        const cv::Scalar color{32, 201, 151};
        const cv::Rect rect = clamp_rect(pose.bbox, image);
        cv::rectangle(image, rect, color, style.line_width, cv::LINE_AA);
        draw_badge(image, rect, pose_label(pose), color, style);

        for (const auto& [lhs, rhs] : skeleton_edges) {
            if (lhs >= static_cast<int>(pose.keypoints.size()) ||
                rhs >= static_cast<int>(pose.keypoints.size())) {
                continue;
            }
            const auto& left = pose.keypoints[static_cast<std::size_t>(lhs)];
            const auto& right = pose.keypoints[static_cast<std::size_t>(rhs)];
            if (left.score < 0.35F || right.score < 0.35F) {
                continue;
            }
            cv::line(image,
                     cv::Point{static_cast<int>(std::lround(left.point.x)),
                               static_cast<int>(std::lround(left.point.y))},
                     cv::Point{static_cast<int>(std::lround(right.point.x)),
                               static_cast<int>(std::lround(right.point.y))},
                     cv::Scalar{255, 214, 10}, style.line_width, cv::LINE_AA);
        }

        for (const auto& keypoint : pose.keypoints) {
            if (keypoint.score < 0.35F) {
                continue;
            }
            cv::circle(
                image,
                cv::Point{static_cast<int>(std::lround(keypoint.point.x)),
                          static_cast<int>(std::lround(keypoint.point.y))},
                style.point_radius, cv::Scalar{255, 255, 255}, cv::FILLED,
                cv::LINE_AA);
            cv::circle(
                image,
                cv::Point{static_cast<int>(std::lround(keypoint.point.x)),
                          static_cast<int>(std::lround(keypoint.point.y))},
                style.point_radius + 2, cv::Scalar{255, 94, 87},
                style.point_ring_thickness, cv::LINE_AA);
        }
    }
}

}  // namespace examples
