#pragma once

#include <cctype>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "yolo/core/error.hpp"
#include "yolo/core/image.hpp"

namespace examples
{

struct LoadedImage
{
    std::vector<std::byte> pixels{};
    yolo::Size2i size{};
    yolo::PixelFormat format{yolo::PixelFormat::rgb8};

    [[nodiscard]] yolo::ImageView view() const noexcept {
        const int channels = format == yolo::PixelFormat::gray8 ? 1 : 3;
        return yolo::ImageView{
            .bytes = pixels,
            .size = size,
            .stride_bytes = static_cast<std::ptrdiff_t>(size.width * channels),
            .format = format,
        };
    }
};

inline std::optional<std::string> read_token(std::istream& stream) {
    std::string token{};
    char ch = '\0';
    while (stream.get(ch)) {
        if (ch == '#') {
            std::string ignored_line{};
            std::getline(stream, ignored_line);
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            token.push_back(ch);
            break;
        }
    }

    if (token.empty()) {
        return std::nullopt;
    }

    while (stream.get(ch) && !std::isspace(static_cast<unsigned char>(ch))) {
        token.push_back(ch);
    }

    return token;
}

inline yolo::Result<LoadedImage> load_ppm_image(std::string_view path) {
    std::ifstream stream(std::string{path}, std::ios::binary);
    if (!stream) {
        return {.error = yolo::make_error(yolo::ErrorCode::io_error,
                                          "Failed to open image file.")};
    }

    const auto magic = read_token(stream);
    const auto width_token = read_token(stream);
    const auto height_token = read_token(stream);
    const auto max_value_token = read_token(stream);
    if (!magic.has_value() || !width_token.has_value() ||
        !height_token.has_value() || !max_value_token.has_value()) {
        return {.error =
                    yolo::make_error(yolo::ErrorCode::io_error,
                                     "PPM header is incomplete or malformed.")};
    }

    if (*magic != "P6" && *magic != "P5") {
        return {.error = yolo::make_error(
                    yolo::ErrorCode::unsupported_model,
                    "Example loader only supports PPM P6/P5 images.")};
    }

    const int width = std::stoi(*width_token);
    const int height = std::stoi(*height_token);
    const int max_value = std::stoi(*max_value_token);
    if (width <= 0 || height <= 0 || max_value != 255) {
        return {.error = yolo::make_error(yolo::ErrorCode::invalid_argument,
                                          "Example loader requires positive "
                                          "dimensions and max value 255.")};
    }

    const int channels = *magic == "P5" ? 1 : 3;
    std::vector<std::byte> pixels(
        static_cast<std::size_t>(width * height * channels));
    stream.read(reinterpret_cast<char*>(pixels.data()),
                static_cast<std::streamsize>(pixels.size()));
    if (!stream) {
        return {.error = yolo::make_error(
                    yolo::ErrorCode::io_error,
                    "Failed to read the full PPM pixel payload.")};
    }

    return {.value =
                LoadedImage{
                    .pixels = std::move(pixels),
                    .size = yolo::Size2i{width, height},
                    .format = *magic == "P5" ? yolo::PixelFormat::gray8
                                             : yolo::PixelFormat::rgb8,
                },
            .error = {}};
}

inline int print_error(const yolo::Error& error) {
    std::cerr << "error: " << error.message << '\n';
    if (error.context.has_value()) {
        if (error.context->component.has_value()) {
            std::cerr << "component: " << *error.context->component << '\n';
        }
        if (error.context->input_name.has_value()) {
            std::cerr << "input: " << *error.context->input_name << '\n';
        }
        if (error.context->output_name.has_value()) {
            std::cerr << "output: " << *error.context->output_name << '\n';
        }
        if (error.context->expected.has_value()) {
            std::cerr << "expected: " << *error.context->expected << '\n';
        }
        if (error.context->actual.has_value()) {
            std::cerr << "actual: " << *error.context->actual << '\n';
        }
    }

    return 1;
}

}  // namespace examples
