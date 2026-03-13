#pragma once

#include <string_view>

#include "yolo/core/error.hpp"
#include "yolo/core/image.hpp"
#include "yolo/detail/tensor_utils.hpp"

namespace yolo::detail
{

struct PreprocessedImage
{
    RawTensor tensor{};
    PreprocessRecord record{};
};

[[nodiscard]] Result<PreprocessedImage> preprocess_image(
    const ImageView& image, const PreprocessPolicy& policy,
    std::string_view input_name = "images");

}  // namespace yolo::detail
