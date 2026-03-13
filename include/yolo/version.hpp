#pragma once

#include <string_view>

namespace yolo
{

inline constexpr int kVersionMajor = 0;
inline constexpr int kVersionMinor = 1;
inline constexpr int kVersionPatch = 0;
inline constexpr std::string_view kVersion = "0.1.0";

[[nodiscard]] inline constexpr std::string_view version_string() noexcept {
    return kVersion;
}

}  // namespace yolo
