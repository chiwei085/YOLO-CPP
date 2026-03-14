#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include "yolo/core/error.hpp"
#include "yolo/core/image.hpp"

namespace examples
{

inline yolo::Result<cv::Mat> load_cv_image(std::string_view path) {
    cv::Mat image = cv::imread(std::string{path}, cv::IMREAD_COLOR);
    if (image.empty()) {
        return {.error = yolo::make_error(yolo::ErrorCode::io_error,
                                          "Failed to load image via OpenCV.")};
    }

    return {.value = image, .error = {}};
}

inline yolo::ImageView mat_view(const cv::Mat& image) {
    return yolo::ImageView{
        .bytes = std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(image.data),
            static_cast<std::size_t>(image.total() * image.elemSize())),
        .size = yolo::Size2i{image.cols, image.rows},
        .stride_bytes = static_cast<std::ptrdiff_t>(image.step),
        .format = image.channels() == 1 ? yolo::PixelFormat::gray8
                                        : yolo::PixelFormat::bgr8,
    };
}

inline std::optional<yolo::Error> save_cv_image(const cv::Mat& image,
                                                std::string_view path) {
    if (!cv::imwrite(std::string{path}, image)) {
        return yolo::make_error(yolo::ErrorCode::io_error,
                                "Failed to save OpenCV output image.");
    }
    return std::nullopt;
}

}  // namespace examples
