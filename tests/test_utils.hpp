#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "yolo/core/image.hpp"
#include "yolo/core/model_spec.hpp"
#include "yolo/core/tensor.hpp"
#include "yolo/core/types.hpp"
#include "yolo/detail/tensor_utils.hpp"

namespace yolo::test
{

struct OwnedImage
{
    std::vector<std::byte> storage{};
    Size2i size{};
    std::ptrdiff_t stride_bytes{0};
    PixelFormat format{PixelFormat::bgr8};

    [[nodiscard]] ImageView view() const noexcept {
        return ImageView{
            .bytes = storage,
            .size = size,
            .stride_bytes = stride_bytes,
            .format = format,
        };
    }
};

inline TensorInfo make_tensor_info(std::string name, TensorDataType data_type,
                                   std::initializer_list<std::int64_t> dims) {
    TensorShape shape{};
    for (const std::int64_t dim : dims) {
        shape.dims.push_back(TensorDimension::fixed(dim));
    }

    return TensorInfo{
        .name = std::move(name),
        .data_type = data_type,
        .shape = std::move(shape),
    };
}

inline yolo::detail::RawTensor make_float_tensor(
    std::string name, std::initializer_list<std::int64_t> dims,
    std::initializer_list<float> values) {
    yolo::detail::RawTensor tensor{};
    tensor.info = make_tensor_info(std::move(name), TensorDataType::float32, dims);
    tensor.storage.resize(values.size() * sizeof(float));
    if (!values.size()) {
        return tensor;
    }

    std::memcpy(tensor.storage.data(), values.begin(), tensor.storage.size());
    return tensor;
}

inline yolo::detail::RawTensor make_uint8_tensor(
    std::string name, std::initializer_list<std::int64_t> dims,
    std::initializer_list<std::uint8_t> values) {
    yolo::detail::RawTensor tensor{};
    tensor.info = make_tensor_info(std::move(name), TensorDataType::uint8, dims);
    tensor.storage.resize(values.size());
    std::memcpy(tensor.storage.data(), values.begin(), tensor.storage.size());
    return tensor;
}

inline OwnedImage make_bgr_image(Size2i size,
                                 const std::vector<std::uint8_t>& pixels) {
    OwnedImage image{};
    image.size = size;
    image.format = PixelFormat::bgr8;
    image.stride_bytes = size.width * 3;
    image.storage.resize(pixels.size());
    std::memcpy(image.storage.data(), pixels.data(), pixels.size());
    return image;
}

inline OwnedImage make_rgb_image(Size2i size,
                                 const std::vector<std::uint8_t>& pixels) {
    OwnedImage image{};
    image.size = size;
    image.format = PixelFormat::rgb8;
    image.stride_bytes = size.width * 3;
    image.storage.resize(pixels.size());
    std::memcpy(image.storage.data(), pixels.data(), pixels.size());
    return image;
}

inline ModelSpec make_model_spec(std::string path, TaskKind task) {
    return ModelSpec{
        .path = std::move(path),
        .task = task,
    };
}

}  // namespace yolo::test
