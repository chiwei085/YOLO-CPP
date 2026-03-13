#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include "yolo/detail/segmentation_runtime.hpp"

namespace
{

yolo::PreprocessRecord make_direct_record() {
    return yolo::PreprocessRecord{
        .source_size = {4, 4},
        .target_size = {4, 4},
        .resized_size = {4, 4},
        .resize_scale = {1.0F, 1.0F},
        .padding = {},
        .source_format = yolo::PixelFormat::bgr8,
        .output_format = yolo::PixelFormat::rgb8,
        .color_conversion = yolo::ColorConversion::swap_rb,
        .resize_mode = yolo::ResizeMode::direct,
        .tensor_layout = yolo::TensorLayout::nchw,
    };
}

std::size_t active_pixels(const yolo::SegmentationMask& mask) {
    return static_cast<std::size_t>(std::count(mask.data.begin(), mask.data.end(),
                                               static_cast<std::uint8_t>(1)));
}

TEST_CASE("segmentation mask projection thresholds and crops to bbox",
          "[component][segmentation]") {
    const yolo::detail::ProtoMaskTensor proto{
        .size = {2, 2},
        .channel_count = 1,
        .values = {1.0F, 1.0F, 1.0F, 1.0F},
    };
    const yolo::detail::SegmentationCandidate candidate{
        .bbox = yolo::RectF{.x = 1.0F, .y = 1.0F, .width = 2.0F, .height = 2.0F},
        .score = 0.9F,
        .class_id = 0,
        .mask_coefficients = {1.0F},
    };

    const auto mask = yolo::detail::project_segmentation_mask(
        candidate, proto, make_direct_record(), 0.5F);

    CHECK(mask.size.width == 4);
    CHECK(mask.size.height == 4);
    CHECK(active_pixels(mask) == 4);
    CHECK(mask.data[1 * 4 + 1] == 1);
    CHECK(mask.data[1 * 4 + 2] == 1);
    CHECK(mask.data[2 * 4 + 1] == 1);
    CHECK(mask.data[2 * 4 + 2] == 1);
    CHECK(mask.data[0] == 0);
}

TEST_CASE("segmentation mask projection restores through letterbox padding",
          "[component][segmentation]") {
    const yolo::detail::ProtoMaskTensor proto{
        .size = {2, 2},
        .channel_count = 1,
        .values = {2.0F, 2.0F, 2.0F, 2.0F},
    };
    const yolo::detail::SegmentationCandidate candidate{
        .bbox = yolo::RectF{.x = 0.0F, .y = 1.0F, .width = 4.0F, .height = 2.0F},
        .score = 0.9F,
        .class_id = 0,
        .mask_coefficients = {1.0F},
    };
    const yolo::PreprocessRecord record{
        .source_size = {4, 2},
        .target_size = {4, 4},
        .resized_size = {4, 2},
        .resize_scale = {1.0F, 1.0F},
        .padding = {.left = 0, .top = 1, .right = 0, .bottom = 1},
        .source_format = yolo::PixelFormat::bgr8,
        .output_format = yolo::PixelFormat::rgb8,
        .color_conversion = yolo::ColorConversion::swap_rb,
        .resize_mode = yolo::ResizeMode::letterbox,
        .tensor_layout = yolo::TensorLayout::nchw,
    };

    const auto mask =
        yolo::detail::project_segmentation_mask(candidate, proto, record, 0.5F);

    CHECK(mask.size.width == 4);
    CHECK(mask.size.height == 2);
    CHECK(active_pixels(mask) == 8);
}

TEST_CASE("segmentation mask projection handles empty coeffs and out-of-range bbox",
          "[component][segmentation]") {
    const yolo::detail::ProtoMaskTensor proto{
        .size = {2, 2},
        .channel_count = 1,
        .values = {1.0F, 1.0F, 1.0F, 1.0F},
    };

    SECTION("empty coefficients produce an empty mask") {
        const auto mask = yolo::detail::project_segmentation_mask(
            yolo::detail::SegmentationCandidate{
                .bbox = yolo::RectF{.x = 0.0F, .y = 0.0F, .width = 4.0F, .height = 4.0F},
                .score = 0.9F,
                .class_id = 0,
                .mask_coefficients = {},
            },
            proto, make_direct_record(), 0.5F);

        CHECK(active_pixels(mask) == 0);
    }

    SECTION("bbox outside image is clamped") {
        const auto mask = yolo::detail::project_segmentation_mask(
            yolo::detail::SegmentationCandidate{
                .bbox = yolo::RectF{.x = 3.0F, .y = 3.0F, .width = 4.0F, .height = 4.0F},
                .score = 0.9F,
                .class_id = 0,
                .mask_coefficients = {1.0F},
            },
            proto, make_direct_record(), 0.5F);

        CHECK(mask.size.width == 4);
        CHECK(mask.size.height == 4);
        CHECK(active_pixels(mask) == 1);
        CHECK(mask.data.back() == 1);
    }
}

}  // namespace
