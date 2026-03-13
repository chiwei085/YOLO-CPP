#pragma once

#include "yolo/detail/image_preprocess.hpp"

namespace yolo::detail
{

[[nodiscard]] Result<ImageDebugBuffer> resize_classification_image(
    const ImageView& image, const PreprocessPolicy& policy,
    Size2i resized_size);

}  // namespace yolo::detail
