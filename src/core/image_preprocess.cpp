#include "yolo/detail/image_preprocess.hpp"
#include "yolo/detail/classification_resize.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace yolo
{

PreprocessPolicy make_detection_preprocess_policy(Size2i target_size) {
    return PreprocessPolicy{
        .target_size = target_size,
        .resize_mode = ResizeMode::letterbox,
        .output_format = PixelFormat::rgb8,
        .color_conversion = ColorConversion::swap_rb,
        .normalize =
            NormalizeSpec{
                .input_scale = 255.0F,
                .mean = {0.0F, 0.0F, 0.0F, 0.0F},
                .std = {1.0F, 1.0F, 1.0F, 1.0F},
            },
        .tensor_layout = TensorLayout::nchw,
        .pad_value = {114.0F, 114.0F, 114.0F, 0.0F},
    };
}

PreprocessPolicy make_classification_preprocess_policy(Size2i target_size) {
    return PreprocessPolicy{
        .target_size = target_size,
        .resize_mode = ResizeMode::resize_crop,
        .output_format = PixelFormat::rgb8,
        .color_conversion = ColorConversion::swap_rb,
        .normalize =
            NormalizeSpec{
                .input_scale = 255.0F,
                .mean = {0.0F, 0.0F, 0.0F, 0.0F},
                .std = {1.0F, 1.0F, 1.0F, 1.0F},
            },
        .tensor_layout = TensorLayout::nchw,
        .pad_value = {0.0F, 0.0F, 0.0F, 0.0F},
    };
}

}  // namespace yolo

namespace yolo::detail
{
namespace
{

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

std::array<float, 4> bilinear_sample(const ImageView& image, float x, float y) {
    const float clamped_x =
        std::clamp(x, 0.0F, static_cast<float>(image.size.width - 1));
    const float clamped_y =
        std::clamp(y, 0.0F, static_cast<float>(image.size.height - 1));

    const int x0 = static_cast<int>(std::floor(clamped_x));
    const int y0 = static_cast<int>(std::floor(clamped_y));
    const int x1 = std::min(x0 + 1, image.size.width - 1);
    const int y1 = std::min(y0 + 1, image.size.height - 1);

    const float dx = clamped_x - static_cast<float>(x0);
    const float dy = clamped_y - static_cast<float>(y0);

    const std::array<float, 4> p00 = load_pixel(image, x0, y0);
    const std::array<float, 4> p10 = load_pixel(image, x1, y0);
    const std::array<float, 4> p01 = load_pixel(image, x0, y1);
    const std::array<float, 4> p11 = load_pixel(image, x1, y1);

    std::array<float, 4> out{};
    for (std::size_t c = 0; c < out.size(); ++c) {
        const float top = p00[c] * (1.0F - dx) + p10[c] * dx;
        const float bottom = p01[c] * (1.0F - dx) + p11[c] * dx;
        out[c] = top * (1.0F - dy) + bottom * dy;
    }

    return out;
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

float normalize_channel(float value, const NormalizeSpec& normalize,
                        std::size_t channel) {
    const float scaled = value / normalize.input_scale;
    return (scaled - normalize.mean[channel]) / normalize.std[channel];
}

std::size_t image_index(int width, int channels, int x, int y, int c) {
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
            static_cast<std::size_t>(x)) *
               static_cast<std::size_t>(channels) +
           static_cast<std::size_t>(c);
}

PreprocessRecord build_record(const ImageView& image,
                              const PreprocessPolicy& policy) {
    PreprocessRecord record{};
    record.source_size = image.size;
    record.target_size = policy.target_size;
    record.source_format = image.format;
    record.output_format = policy.output_format;
    record.color_conversion = policy.color_conversion;
    record.normalize = policy.normalize;
    record.resize_mode = policy.resize_mode;
    record.tensor_layout = policy.tensor_layout;

    if (policy.resize_mode == ResizeMode::letterbox) {
        const float scale = std::min(
            static_cast<float>(policy.target_size.width) / image.size.width,
            static_cast<float>(policy.target_size.height) / image.size.height);
        record.resize_scale = {scale, scale};
        record.resized_size = {
            std::max(1,
                     static_cast<int>(std::lround(image.size.width * scale))),
            std::max(1,
                     static_cast<int>(std::lround(image.size.height * scale)))};
        const int pad_x = policy.target_size.width - record.resized_size.width;
        const int pad_y =
            policy.target_size.height - record.resized_size.height;
        record.padding = {pad_x / 2, pad_y / 2, pad_x - pad_x / 2,
                          pad_y - pad_y / 2};
    }
    else if (policy.resize_mode == ResizeMode::resize_crop) {
        float scale = 1.0F;
        if (image.size.width >= image.size.height) {
            scale = static_cast<float>(policy.target_size.height) /
                    static_cast<float>(image.size.height);
            record.resized_size = {
                std::max(1, static_cast<int>(image.size.width * scale)),
                policy.target_size.height};
        }
        else {
            scale = static_cast<float>(policy.target_size.width) /
                    static_cast<float>(image.size.width);
            record.resized_size = {
                policy.target_size.width,
                std::max(1, static_cast<int>(image.size.height * scale))};
        }
        record.resize_scale = {scale, scale};
        record.crop = RectI{
            .x = std::max(
                0, (record.resized_size.width - policy.target_size.width) / 2),
            .y = std::max(
                0,
                (record.resized_size.height - policy.target_size.height) / 2),
            .width = policy.target_size.width,
            .height = policy.target_size.height,
        };
    }
    else {
        record.resize_scale = {
            static_cast<float>(policy.target_size.width) / image.size.width,
            static_cast<float>(policy.target_size.height) / image.size.height};
        record.resized_size = policy.target_size;
    }

    return record;
}

}  // namespace

Result<PreprocessedImage> preprocess_image(const ImageView& image,
                                           const PreprocessPolicy& policy,
                                           std::string_view input_name) {
    if (image.empty()) {
        return {
            .error = make_error(
                ErrorCode::invalid_argument, "Input image must not be empty.",
                ErrorContext{.component = std::string{"image_preprocess"}})};
    }

    if (policy.target_size.empty()) {
        return {.error = make_error(ErrorCode::invalid_argument,
                                    "Preprocess target size must not be empty.",
                                    ErrorContext{.component = std::string{
                                                     "image_preprocess"}})};
    }

    const int channels = output_channel_count(policy.output_format);
    if (channels <= 0) {
        return {
            .error = make_error(
                ErrorCode::invalid_argument,
                "Unsupported preprocess output pixel format.",
                ErrorContext{.component = std::string{"image_preprocess"}})};
    }

    PreprocessedImage preprocessed{};
    preprocessed.record = build_record(image, policy);
    preprocessed.tensor.info.name = std::string{input_name};
    preprocessed.tensor.info.data_type = TensorDataType::float32;
    if (policy.tensor_layout == TensorLayout::nchw) {
        preprocessed.tensor.info.shape.dims = {
            TensorDimension::fixed(1), TensorDimension::fixed(channels),
            TensorDimension::fixed(policy.target_size.height),
            TensorDimension::fixed(policy.target_size.width)};
    }
    else {
        preprocessed.tensor.info.shape.dims = {
            TensorDimension::fixed(1),
            TensorDimension::fixed(policy.target_size.height),
            TensorDimension::fixed(policy.target_size.width),
            TensorDimension::fixed(channels)};
    }

    const std::size_t element_count =
        *preprocessed.tensor.info.shape.element_count();
    preprocessed.tensor.values.resize(element_count);
    float* tensor = preprocessed.tensor.values.data();
    std::optional<ImageDebugBuffer> resized_image{};
    if (policy.resize_mode == ResizeMode::resize_crop) {
        auto resized_result = resize_classification_image(
            image, policy, preprocessed.record.resized_size);
        if (!resized_result.ok()) {
            return {.error = resized_result.error};
        }
        resized_image = std::move(*resized_result.value);
    }

    const auto write_value = [&](int x, int y, int c, float value) {
        std::size_t index = 0;
        if (policy.tensor_layout == TensorLayout::nchw) {
            index = static_cast<std::size_t>(c) * policy.target_size.height *
                        policy.target_size.width +
                    static_cast<std::size_t>(y) * policy.target_size.width + x;
        }
        else {
            index =
                (static_cast<std::size_t>(y) * policy.target_size.width + x) *
                    channels +
                c;
        }
        tensor[index] = value;
    };

    for (int y = 0; y < policy.target_size.height; ++y) {
        for (int x = 0; x < policy.target_size.width; ++x) {
            bool use_pad = false;
            float source_x = 0.0F;
            float source_y = 0.0F;

            if (policy.resize_mode == ResizeMode::letterbox) {
                const int inner_x = x - preprocessed.record.padding.left;
                const int inner_y = y - preprocessed.record.padding.top;
                if (inner_x < 0 || inner_y < 0 ||
                    inner_x >= preprocessed.record.resized_size.width ||
                    inner_y >= preprocessed.record.resized_size.height) {
                    use_pad = true;
                }
                else {
                    source_x =
                        (inner_x + 0.5F) / preprocessed.record.resize_scale.x -
                        0.5F;
                    source_y =
                        (inner_y + 0.5F) / preprocessed.record.resize_scale.y -
                        0.5F;
                }
            }
            else if (policy.resize_mode == ResizeMode::resize_crop) {
                const int crop_x = preprocessed.record.crop->x + x;
                const int crop_y = preprocessed.record.crop->y + y;
                for (int c = 0; c < channels; ++c) {
                    const float value = resized_image->values[image_index(
                        preprocessed.record.resized_size.width, channels, crop_x,
                        crop_y, c)];
                    write_value(x, y, c,
                                normalize_channel(
                                    value, policy.normalize,
                                    static_cast<std::size_t>(c)));
                }
                continue;
            }
            else {
                source_x =
                    (x + 0.5F) / preprocessed.record.resize_scale.x - 0.5F;
                source_y =
                    (y + 0.5F) / preprocessed.record.resize_scale.y - 0.5F;
            }

            std::array<float, 4> pixel =
                use_pad
                    ? policy.pad_value
                    : convert_color(bilinear_sample(image, source_x, source_y),
                                    policy);

            if (policy.output_format == PixelFormat::gray8) {
                write_value(x, y, 0,
                            normalize_channel(pixel[0], policy.normalize, 0));
                continue;
            }

            for (int c = 0; c < channels; ++c) {
                write_value(x, y, c,
                            normalize_channel(
                                pixel[static_cast<std::size_t>(c)],
                                policy.normalize, static_cast<std::size_t>(c)));
            }
        }
    }

    return {.value = std::move(preprocessed), .error = {}};
}

Result<ClassificationPreprocessTrace> trace_classification_preprocess(
    const ImageView& image, const PreprocessPolicy& policy,
    std::string_view input_name) {
    if (policy.resize_mode != ResizeMode::resize_crop) {
        return {.error = make_error(
                    ErrorCode::invalid_argument,
                    "Classification preprocess trace expects resize-crop "
                    "policy.",
                    ErrorContext{
                        .component = std::string{"image_preprocess"}})};
    }

    auto preprocessed_result = preprocess_image(image, policy, input_name);
    if (!preprocessed_result.ok()) {
        return {.error = preprocessed_result.error};
    }

    const PreprocessRecord& record = preprocessed_result.value->record;
    const int channels = output_channel_count(policy.output_format);

    ClassificationPreprocessTrace trace{};
    trace.source_size = image.size;
    trace.target_size = policy.target_size;
    trace.resized_size = record.resized_size;
    trace.crop = record.crop;
    trace.record = record;
    trace.tensor = preprocessed_result.value->tensor;
    auto resized_result =
        resize_classification_image(image, policy, record.resized_size);
    if (!resized_result.ok()) {
        return {.error = resized_result.error};
    }
    trace.resized_image = std::move(*resized_result.value);

    trace.cropped_image.size = policy.target_size;
    trace.cropped_image.channels = channels;
    trace.cropped_image.values.resize(
        static_cast<std::size_t>(policy.target_size.width) *
        static_cast<std::size_t>(policy.target_size.height) *
        static_cast<std::size_t>(channels));

    for (int y = 0; y < policy.target_size.height; ++y) {
        for (int x = 0; x < policy.target_size.width; ++x) {
            const int crop_x = record.crop->x + x;
            const int crop_y = record.crop->y + y;
            for (int c = 0; c < channels; ++c) {
                trace.cropped_image.values[image_index(
                    policy.target_size.width, channels, x, y, c)] =
                    trace.resized_image.values[image_index(
                        record.resized_size.width, channels, crop_x, crop_y, c)];
            }
        }
    }

    return {.value = std::move(trace), .error = {}};
}

}  // namespace yolo::detail
