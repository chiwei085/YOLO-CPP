# Examples

`examples/` is organized by demonstration route, not by internal layering.

## Routes

- `ppm_loader/`
  Minimal end-to-end examples that use the repo's tiny PPM helper.
- `opencv_viz/`
  Visualization examples that load `jpg` / `png` with OpenCV and save annotated
  output images.
- `assets/`
  Demo input images used by the example commands in the top-level README.
- `support/`
  Shared helpers used by the example entry points, including CLI utilities and
  OpenCV-specific support code.

## PPM Loader Route

Programs:

- `ppm_loader/detect_image.cpp`
- `ppm_loader/classify_image.cpp`

Helper:

- `ppm_loader/image_ppm.hpp`

Use this route when you want the smallest possible example and are happy to
feed `.ppm` images.

## OpenCV Visualization Route

Programs:

- `opencv_viz/detect_image.cpp`
- `opencv_viz/pose_image.cpp`

Helpers:

- `support/opencv_io.hpp`
- `support/opencv_overlay.hpp`
- `support/coco_labels.hpp`

Use this route when you want direct `jpg` / `png` input, COCO labels for
detection, and annotated output images for docs or demos.
