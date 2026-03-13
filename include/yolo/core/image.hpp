#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>

#include "yolo/core/types.hpp"

namespace yolo
{

enum class ResizeMode
{
    direct,
    letterbox,
    resize_crop,
};

enum class ColorConversion
{
    none,
    swap_rb,
};

enum class TensorLayout
{
    nchw,
    nhwc,
};

struct NormalizeSpec
{
    float input_scale{1.0F};
    std::array<float, 4> mean{0.0F, 0.0F, 0.0F, 0.0F};
    std::array<float, 4> std{1.0F, 1.0F, 1.0F, 1.0F};
};

struct PreprocessPolicy
{
    Size2i target_size{};
    ResizeMode resize_mode{ResizeMode::direct};
    PixelFormat output_format{PixelFormat::rgb8};
    ColorConversion color_conversion{ColorConversion::none};
    NormalizeSpec normalize{};
    TensorLayout tensor_layout{TensorLayout::nchw};
    std::array<float, 4> pad_value{114.0F, 114.0F, 114.0F, 0.0F};
};

struct PreprocessRecord
{
    Size2i source_size{};
    Size2i target_size{};
    Size2i resized_size{};
    Scale2f resize_scale{};
    Padding2i padding{};
    std::optional<RectI> crop{};
    PixelFormat source_format{PixelFormat::bgr8};
    PixelFormat output_format{PixelFormat::rgb8};
    ColorConversion color_conversion{ColorConversion::none};
    NormalizeSpec normalize{};
    ResizeMode resize_mode{ResizeMode::direct};
    TensorLayout tensor_layout{TensorLayout::nchw};
};

struct ImageView
{
    std::span<const std::byte> bytes{};
    Size2i size{};
    std::ptrdiff_t stride_bytes{0};
    PixelFormat format{PixelFormat::bgr8};

    [[nodiscard]] constexpr bool empty() const noexcept {
        return bytes.empty() || size.empty();
    }

    [[nodiscard]] constexpr int channels() const noexcept {
        switch (format) {
            case PixelFormat::gray8:
                return 1;
            case PixelFormat::bgr8:
            case PixelFormat::rgb8:
                return 3;
            case PixelFormat::bgra8:
            case PixelFormat::rgba8:
                return 4;
        }

        return 0;
    }
};

[[nodiscard]] PreprocessPolicy make_detection_preprocess_policy(
    Size2i target_size);
[[nodiscard]] PreprocessPolicy make_classification_preprocess_policy(
    Size2i target_size);

}  // namespace yolo
