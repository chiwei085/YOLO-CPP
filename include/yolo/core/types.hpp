#pragma once

#include <cstdint>
#include <vector>

namespace yolo
{

enum class TaskKind
{
    detect,
    classify,
    seg,
    pose,
    obb,
};

enum class PixelFormat
{
    bgr8,
    rgb8,
    gray8,
    bgra8,
    rgba8,
};

enum class TensorDataType
{
    boolean,
    uint8,
    int8,
    int16,
    int32,
    int64,
    float16,
    float32,
    float64,
};

struct Size2i
{
    int width{0};
    int height{0};

    [[nodiscard]] constexpr bool empty() const noexcept {
        return width <= 0 || height <= 0;
    }
};

struct Padding2i
{
    int left{0};
    int top{0};
    int right{0};
    int bottom{0};

    [[nodiscard]] constexpr int horizontal() const noexcept {
        return left + right;
    }

    [[nodiscard]] constexpr int vertical() const noexcept {
        return top + bottom;
    }
};

struct Scale2f
{
    float x{1.0F};
    float y{1.0F};
};

struct Size2f
{
    float width{0.0F};
    float height{0.0F};
};

struct Point2f
{
    float x{0.0F};
    float y{0.0F};
};

struct RectI
{
    int x{0};
    int y{0};
    int width{0};
    int height{0};

    [[nodiscard]] constexpr bool empty() const noexcept {
        return width <= 0 || height <= 0;
    }
};

struct RectF
{
    float x{0.0F};
    float y{0.0F};
    float width{0.0F};
    float height{0.0F};

    [[nodiscard]] constexpr float area() const noexcept {
        return width * height;
    }
};

struct Keypoint
{
    Point2f point{};
    float score{0.0F};
};

using ClassId = std::uint32_t;
using Shape = std::vector<std::int64_t>;

}  // namespace yolo
