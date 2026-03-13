#include "yolo/detail/classification_resize.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace yolo::detail
{
namespace
{

constexpr std::uint32_t precision_bits = 32U - 8U - 2U;

int source_channel_count(PixelFormat format) {
    switch (format) {
        case PixelFormat::gray8:
            return 1;
        case PixelFormat::bgr8:
        case PixelFormat::rgb8:
            return 3;
        case PixelFormat::bgra8:
        case PixelFormat::rgba8:
            return 4;
    }

    return 0;
}

int output_channel_count(PixelFormat format) {
    switch (format) {
        case PixelFormat::gray8:
            return 1;
        case PixelFormat::bgr8:
        case PixelFormat::rgb8:
            return 3;
        case PixelFormat::bgra8:
        case PixelFormat::rgba8:
            return 4;
    }

    return 0;
}

std::size_t image_index(int width, int channels, int x, int y, int c) {
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
            static_cast<std::size_t>(x)) *
               static_cast<std::size_t>(channels) +
           static_cast<std::size_t>(c);
}

std::array<float, 4> load_pixel(const ImageView& image, int x, int y) {
    const int channels = source_channel_count(image.format);
    const std::byte* row = image.bytes.data() + y * image.stride_bytes;
    const std::uint8_t* pixel =
        reinterpret_cast<const std::uint8_t*>(row + x * channels);

    switch (image.format) {
        case PixelFormat::gray8:
            return {static_cast<float>(pixel[0]), static_cast<float>(pixel[0]),
                    static_cast<float>(pixel[0]), 0.0F};
        case PixelFormat::bgr8:
            return {static_cast<float>(pixel[0]), static_cast<float>(pixel[1]),
                    static_cast<float>(pixel[2]), 0.0F};
        case PixelFormat::rgb8:
            return {static_cast<float>(pixel[2]), static_cast<float>(pixel[1]),
                    static_cast<float>(pixel[0]), 0.0F};
        case PixelFormat::bgra8:
            return {static_cast<float>(pixel[0]), static_cast<float>(pixel[1]),
                    static_cast<float>(pixel[2]), static_cast<float>(pixel[3])};
        case PixelFormat::rgba8:
            return {static_cast<float>(pixel[2]), static_cast<float>(pixel[1]),
                    static_cast<float>(pixel[0]), static_cast<float>(pixel[3])};
    }

    return {};
}

std::array<float, 4> convert_color(const std::array<float, 4>& bgr,
                                   const PreprocessPolicy& policy) {
    std::array<float, 4> converted = bgr;
    if (policy.output_format == PixelFormat::gray8) {
        const float gray = 0.114F * bgr[0] + 0.587F * bgr[1] + 0.299F * bgr[2];
        return {gray, gray, gray, bgr[3]};
    }

    if (policy.color_conversion == ColorConversion::swap_rb) {
        std::swap(converted[0], converted[2]);
    }

    return converted;
}

double bilinear_filter(double x) {
    if (x < 0.0) {
        x = -x;
    }
    return x < 1.0 ? 1.0 - x : 0.0;
}

std::uint8_t clip8(std::int64_t value) {
    const std::int64_t shifted = value >> precision_bits;
    if (shifted < 0) {
        return 0;
    }
    if (shifted > 255) {
        return 255;
    }
    return static_cast<std::uint8_t>(shifted);
}

struct CoeffTable
{
    int kernel_size{0};
    std::vector<int32_t> bounds{};
    std::vector<std::int32_t> weights{};
};

CoeffTable precompute_coeffs(int input_size, int output_size) {
    const double in0 = 0.0;
    const double in1 = static_cast<double>(input_size);
    const double scale = (in1 - in0) / static_cast<double>(output_size);
    double filter_scale = scale;
    if (filter_scale < 1.0) {
        filter_scale = 1.0;
    }

    const double support = 1.0 * filter_scale;
    const int kernel_size = static_cast<int>(std::ceil(support)) * 2 + 1;

    std::vector<int32_t> bounds(static_cast<std::size_t>(output_size) * 2U);
    std::vector<double> coeffs(static_cast<std::size_t>(output_size * kernel_size));

    constexpr double half_pixel = 0.5;
    for (int out_index = 0; out_index < output_size; ++out_index) {
        const double center = in0 + (static_cast<double>(out_index) + half_pixel) * scale;
        const double sample_scale = 1.0 / filter_scale;

        int xmin = static_cast<int>(center - support + half_pixel);
        xmin = std::max(xmin, 0);
        int xmax = static_cast<int>(center + support + half_pixel);
        xmax = std::min(xmax, input_size);
        const int count = xmax - xmin;

        double sum = 0.0;
        double* kernel = coeffs.data() + static_cast<std::size_t>(out_index * kernel_size);
        int k = 0;
        for (; k < count; ++k) {
            const double weight =
                bilinear_filter((static_cast<double>(k + xmin) - center + half_pixel) *
                                sample_scale);
            kernel[k] = weight;
            sum += weight;
        }
        for (int i = 0; i < count; ++i) {
            if (sum != 0.0) {
                kernel[i] /= sum;
            }
        }
        for (; k < kernel_size; ++k) {
            kernel[k] = 0.0;
        }
        bounds[static_cast<std::size_t>(out_index) * 2U] = xmin;
        bounds[static_cast<std::size_t>(out_index) * 2U + 1U] = count;
    }

    std::vector<std::int32_t> quantized(coeffs.size());
    const double scale_shift = static_cast<double>(1U << precision_bits);
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        const double coeff = coeffs[i];
        quantized[i] = static_cast<std::int32_t>(
            coeff < 0.0 ? std::trunc(-0.5 + coeff * scale_shift)
                        : std::trunc(0.5 + coeff * scale_shift));
    }

    return CoeffTable{
        .kernel_size = kernel_size,
        .bounds = std::move(bounds),
        .weights = std::move(quantized),
    };
}

std::vector<std::uint8_t> build_source_pixels(const ImageView& image,
                                              const PreprocessPolicy& policy,
                                              int output_channels) {
    std::vector<std::uint8_t> pixels(
        static_cast<std::size_t>(image.size.width) *
        static_cast<std::size_t>(image.size.height) *
        static_cast<std::size_t>(output_channels));

    for (int y = 0; y < image.size.height; ++y) {
        for (int x = 0; x < image.size.width; ++x) {
            const auto converted = convert_color(load_pixel(image, x, y), policy);
            for (int c = 0; c < output_channels; ++c) {
                pixels[image_index(image.size.width, output_channels, x, y, c)] =
                    static_cast<std::uint8_t>(std::clamp(
                        std::lround(converted[static_cast<std::size_t>(c)]), 0L, 255L));
            }
        }
    }

    return pixels;
}

std::vector<std::uint8_t> resample_horizontal(const std::vector<std::uint8_t>& input,
                                              int input_width, int input_height,
                                              int channels, int output_width,
                                              const CoeffTable& coeffs) {
    std::vector<std::uint8_t> output(
        static_cast<std::size_t>(output_width) *
        static_cast<std::size_t>(input_height) *
        static_cast<std::size_t>(channels));

    const std::int64_t rounding =
        static_cast<std::int64_t>(1U << (precision_bits - 1U));
    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            const int xmin = coeffs.bounds[static_cast<std::size_t>(x) * 2U];
            const int count = coeffs.bounds[static_cast<std::size_t>(x) * 2U + 1U];
            for (int c = 0; c < channels; ++c) {
                std::int64_t accum = rounding;
                for (int k = 0; k < count; ++k) {
                    const std::uint8_t source =
                        input[image_index(input_width, channels, xmin + k, y, c)];
                    accum += static_cast<std::int64_t>(source) *
                             coeffs.weights[static_cast<std::size_t>(x * coeffs.kernel_size + k)];
                }
                output[image_index(output_width, channels, x, y, c)] = clip8(accum);
            }
        }
    }

    return output;
}

std::vector<std::uint8_t> resample_vertical(const std::vector<std::uint8_t>& input,
                                            int input_width, int input_height,
                                            int channels, int output_height,
                                            const CoeffTable& coeffs) {
    std::vector<std::uint8_t> output(
        static_cast<std::size_t>(input_width) *
        static_cast<std::size_t>(output_height) *
        static_cast<std::size_t>(channels));

    const std::int64_t rounding =
        static_cast<std::int64_t>(1U << (precision_bits - 1U));
    for (int y = 0; y < output_height; ++y) {
        const int ymin = coeffs.bounds[static_cast<std::size_t>(y) * 2U];
        const int count = coeffs.bounds[static_cast<std::size_t>(y) * 2U + 1U];
        for (int x = 0; x < input_width; ++x) {
            for (int c = 0; c < channels; ++c) {
                std::int64_t accum = rounding;
                for (int k = 0; k < count; ++k) {
                    const std::uint8_t source =
                        input[image_index(input_width, channels, x, ymin + k, c)];
                    accum += static_cast<std::int64_t>(source) *
                             coeffs.weights[static_cast<std::size_t>(y * coeffs.kernel_size + k)];
                }
                output[image_index(input_width, channels, x, y, c)] = clip8(accum);
            }
        }
    }

    return output;
}

}  // namespace

Result<ImageDebugBuffer> resize_classification_image(const ImageView& image,
                                                     const PreprocessPolicy& policy,
                                                     Size2i resized_size) {
    if (image.empty()) {
        return {.error = make_error(
                    ErrorCode::invalid_argument,
                    "Classification resize source image must not be empty.",
                    ErrorContext{
                        .component = std::string{"classification_resize"}})};
    }
    if (resized_size.empty()) {
        return {.error = make_error(
                    ErrorCode::invalid_argument,
                    "Classification resize target size must not be empty.",
                    ErrorContext{
                        .component = std::string{"classification_resize"}})};
    }

    const int output_channels = output_channel_count(policy.output_format);
    if (output_channels <= 0) {
        return {.error = make_error(
                    ErrorCode::invalid_argument,
                    "Unsupported classification resize output format.",
                    ErrorContext{
                        .component = std::string{"classification_resize"}})};
    }

    const auto source_pixels =
        build_source_pixels(image, policy, output_channels);
    const CoeffTable x_coeffs =
        precompute_coeffs(image.size.width, resized_size.width);
    const CoeffTable y_coeffs =
        precompute_coeffs(image.size.height, resized_size.height);

    const auto horizontal = resample_horizontal(
        source_pixels, image.size.width, image.size.height, output_channels,
        resized_size.width, x_coeffs);
    const auto resized_pixels = resample_vertical(
        horizontal, resized_size.width, image.size.height, output_channels,
        resized_size.height, y_coeffs);

    ImageDebugBuffer resized{};
    resized.size = resized_size;
    resized.channels = output_channels;
    resized.values.resize(resized_pixels.size());
    for (std::size_t i = 0; i < resized_pixels.size(); ++i) {
        resized.values[i] = static_cast<float>(resized_pixels[i]);
    }

    return {.value = std::move(resized), .error = {}};
}

}  // namespace yolo::detail
