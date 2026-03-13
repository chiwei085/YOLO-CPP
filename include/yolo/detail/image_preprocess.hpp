#pragma once

#include <span>
#include <vector>

#include <string_view>

#include "yolo/core/error.hpp"
#include "yolo/core/image.hpp"
#include "yolo/detail/tensor_utils.hpp"

namespace yolo::detail
{

struct FloatTensor
{
    TensorInfo info{};
    std::vector<float> values{};

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
        return std::as_bytes(std::span<const float>(values));
    }
};

struct PreprocessedImage
{
    FloatTensor tensor{};
    PreprocessRecord record{};
};

[[nodiscard]] Result<PreprocessedImage> preprocess_image(
    const ImageView& image, const PreprocessPolicy& policy,
    std::string_view input_name = "images");

}  // namespace yolo::detail
