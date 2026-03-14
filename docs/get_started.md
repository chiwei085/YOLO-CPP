# Get Started

YOLO-CPP has a small usage model:

1. create a pipeline from a model
2. provide an image as `yolo::ImageView`
3. call a task entry point such as `detect(...)`
4. inspect the structured result

If you already have the library built or installed, this is the fastest way to
start using it from your own C++ code.

## Minimal Detect Example

```cpp
#include <iostream>
#include <vector>

#include "yolo/core/image.hpp"
#include "yolo/facade.hpp"

int main() {
    auto pipeline_result = yolo::create_pipeline(yolo::ModelSpec{
        .path = "yolov8n.onnx",
    });
    if (!pipeline_result.ok()) {
        yolo::throw_if_error(pipeline_result.error);
    }

    const auto& pipeline = *pipeline_result.value;

    // Replace this with pixels from your own image loader.
    const int width = 640;
    const int height = 480;
    std::vector<std::byte> pixels(static_cast<std::size_t>(width * height * 3));

    yolo::ImageView image{
        .bytes = pixels,
        .size = yolo::Size2i{width, height},
        .stride_bytes = width * 3,
        .format = yolo::PixelFormat::rgb8,
    };

    const yolo::DetectionResult result = pipeline->detect(image);
    if (!result.ok()) {
        yolo::throw_if_error(result.error);
    }

    for (const auto& detection : result.detections) {
        std::cout << "class=" << detection.class_id
                  << " score=" << detection.score
                  << " bbox=(" << detection.bbox.x << ", " << detection.bbox.y
                  << ", " << detection.bbox.width << ", "
                  << detection.bbox.height << ")\n";
    }
}
```

## Other Task Entry Points

After pipeline creation, the main task calls are:

- `classify(...)`
- `segment(...)`
- `estimate_pose(...)`
- `detect_obb(...)`
- `run_raw(...)`

These all follow the same general pattern:

1. create or reuse a pipeline
2. provide input data
3. check the returned result object
4. read the task-specific structured output

## Image Input

The core library works with `yolo::ImageView`.

That means you can keep your own image loading stack and only bridge the final
pixel buffer into:

- pixel bytes
- image size
- row stride
- pixel format

If you already use OpenCV or another image library, the normal approach is to
load images there and then build a `yolo::ImageView` over that memory.

## Next Steps

- Read [API Overview](api_overview.md) for the main public types and entry
  points.
- Read [../examples/README.md](../examples/README.md) if you want small
  end-to-end sample programs.
- Read [../README.md](../README.md) for build, setup, and test details.
