#pragma once

#include <optional>
#include <string>
#include <vector>

#include "yolo/core/types.hpp"

namespace yolo
{

struct ModelSpec
{
    std::string path{};
    TaskKind task{TaskKind::detect};
    std::optional<std::string> model_name{};
    std::optional<std::string> adapter{};
    std::optional<Size2i> input_size{};
    std::optional<std::size_t> class_count{};
    std::vector<std::string> labels{};

    [[nodiscard]] bool valid() const noexcept { return !path.empty(); }
};

}  // namespace yolo
