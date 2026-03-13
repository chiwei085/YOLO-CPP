#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "yolo/core/types.hpp"

namespace yolo
{

struct TensorDimension
{
    std::optional<std::int64_t> value{};

    [[nodiscard]] static constexpr TensorDimension dynamic() noexcept {
        return {};
    }

    [[nodiscard]] static constexpr TensorDimension fixed(
        std::int64_t dimension) noexcept {
        return TensorDimension{dimension};
    }

    [[nodiscard]] constexpr bool is_dynamic() const noexcept {
        return !value.has_value();
    }
};

struct TensorShape
{
    std::vector<TensorDimension> dims{};

    [[nodiscard]] bool empty() const noexcept { return dims.empty(); }

    [[nodiscard]] std::size_t rank() const noexcept { return dims.size(); }

    [[nodiscard]] bool is_dynamic() const noexcept {
        for (const TensorDimension& dim : dims) {
            if (dim.is_dynamic()) {
                return true;
            }
        }

        return false;
    }

    [[nodiscard]] std::optional<std::size_t> element_count() const noexcept {
        if (dims.empty() || is_dynamic()) {
            return std::nullopt;
        }

        std::size_t count = 1;
        for (const TensorDimension& dim : dims) {
            count *= static_cast<std::size_t>(*dim.value);
        }

        return count;
    }
};

struct TensorInfo
{
    std::string name{};
    TensorDataType data_type{TensorDataType::float32};
    TensorShape shape{};
};

template <class T>
struct TensorView
{
    std::span<const T> data{};
    TensorShape shape{};

    [[nodiscard]] constexpr bool empty() const noexcept { return data.empty(); }
};

}  // namespace yolo
