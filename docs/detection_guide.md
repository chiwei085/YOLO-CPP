# Detection Guide

## What This Guide Covers

This guide assumes you already know how to create a pipeline and call into the
library at a basic level. Its job is narrower: it explains how to integrate
detection into your own C++ application, where YOLO-CPP should stop, and where
your application code should take over.

## Detection Data Flow In Practice

The practical model is simple: your application owns image acquisition and
result handling, and YOLO-CPP sits in the middle as the inference step.

In a typical application, detection starts before YOLO-CPP sees anything. Your
program receives image data from somewhere such as a file, camera, network
stream, GUI surface, or shared-memory transport, and your code decides how that
data is represented in memory. Once you have a usable pixel buffer, the
library-facing step is small: expose that buffer as a `yolo::ImageView`, call
`detect(...)`, and then consume the returned detections as plain structured
data inside your own application logic.

That boundary is important. YOLO-CPP is responsible for inference and for
producing structured detection results. It is not an application framework, a
windowing toolkit, a camera stack, or a rendering system. The output of the
library is intentionally shaped so that you can hand detections to whatever
comes next in your own program, whether that is a tracker, a rules engine, a
logger, a UI layer, or a message publisher.

Typical flow:

```text
image source
  -> application-owned image buffer
  -> yolo::ImageView
  -> pipeline->detect(...)
  -> std::vector<yolo::Detection>
  -> application-specific handling
```

## Where Image Loading Should Live

Keep image loading outside the library.

Image loading belongs in the application, not in the library. YOLO-CPP only
needs a `yolo::ImageView`, which means the library does not need to know how
you decoded a JPEG, pulled a frame from a camera SDK, received a DMA buffer, or
read from shared memory. You can keep the loader that already makes sense for
your codebase, and only bridge the final pixel buffer into the small amount of
metadata the library actually needs: bytes, size, stride, and pixel format.

That separation is useful because it prevents the inference layer from becoming
entangled with environment-specific concerns. A desktop tool, a robotics
pipeline, and a backend service may all prepare image memory in different ways,
but they can still pass the same `yolo::ImageView` shape into YOLO-CPP once the
buffer is ready. In practice, this usually means the application owns the
loading, lifetime, and transport of the image data, while the library only
borrows a view over that memory for the duration of the call.

## Reusing Pipelines

Create the pipeline once, then reuse it.

Pipelines should usually be **created once** and reused. Model loading, adapter
resolution, and runtime session setup are the expensive parts of the detection
path, so constructing a pipeline for every frame or every image is usually the
wrong integration shape. The normal pattern is to build the pipeline during
startup or component initialization, keep it alive as a long-lived object, and
then create a fresh `yolo::ImageView` for each input image as work arrives.

This distinction helps keep the system simple: the pipeline is the heavyweight
runtime object, while `ImageView` is only a lightweight description of one
piece of input memory. If you keep those roles separate in your design, the
code usually becomes easier to reason about and much cheaper to run.

## Consuming Detections In Your Application

Treat detections as application input, not final application behavior.

The most important application work happens after inference returns. Detection
results are already structured, but they are still raw application inputs, not
final business decisions. In one program you may discard low-confidence
detections, in another you may remap `class_id` values into your own label
table, and in another you may convert bounding boxes into internal domain
objects that carry timestamps, source IDs, or tracking state. The library gives
you detections with coordinates, class IDs, optional labels, and scores; your
application decides what those values mean operationally.

That also means drawing boxes is optional, not required. A visualization tool
may render detections onto an image, but a production integration might instead
send them to a tracker, emit them as telemetry, store them in a log, or feed
them into downstream decision logic. Treat the detection result as structured
data first. Rendering is just one possible consumer.

## Common Integration Mistakes

Most integration mistakes come from putting the boundary in the wrong place.

- Using the wrong pixel format
  If the memory is actually `bgr8` and the `ImageView` says `rgb8`, the call
  can still succeed while the output quality is wrong.
- Using the wrong row stride
  The view has to describe the real layout of the image buffer, not the layout
  you expected to have.
- Recreating the pipeline for every image
  This moves expensive setup into the hot path and usually becomes the first
  avoidable performance problem.
- Assuming the library owns image memory
  `yolo::ImageView` is a borrowed view; your application should keep the source
  buffer alive for the duration of the inference call.
- Mixing rendering concerns into inference too early
  It is usually cleaner to let detection produce structured results first, and
  let a later application layer decide whether to draw, filter, track, log, or
  publish them.

## Next Steps

- Read [API Overview](api_overview.md) for the full public detection-related
  types and functions.
- Read [Get Started](get_started.md) for the minimal pipeline usage model.
- Read [../examples/README.md](../examples/README.md) when you want an
  end-to-end working reference.
