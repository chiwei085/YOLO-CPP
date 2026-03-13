#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "yolo/core/error.hpp"
#include "yolo/core/tensor.hpp"

namespace yolo::detail
{

struct RawTensor
{
    TensorInfo info{};
    std::vector<std::byte> storage{};

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
        return storage;
    }
};

struct RawInputTensor
{
    TensorInfo info{};
    std::span<const std::byte> bytes{};
};

using RawOutputTensors = std::vector<RawTensor>;

[[nodiscard]] std::size_t tensor_element_size(TensorDataType data_type);
[[nodiscard]] std::optional<std::size_t> dense_byte_count(
    const TensorInfo& info);
[[nodiscard]] std::string format_shape(const TensorShape& shape);
[[nodiscard]] std::string format_data_type(TensorDataType data_type);
[[nodiscard]] Error make_shape_error(std::string_view component,
                                     std::string_view name,
                                     const TensorShape& expected,
                                     const TensorShape& actual);
[[nodiscard]] Error make_type_error(std::string_view component,
                                    std::string_view name,
                                    TensorDataType expected,
                                    TensorDataType actual);

}  // namespace yolo::detail
