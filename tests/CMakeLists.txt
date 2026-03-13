include(FetchContent)

find_package(Catch2 3 CONFIG QUIET)
if(NOT Catch2_FOUND)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.1
  )
  FetchContent_MakeAvailable(Catch2)
endif()

include(Catch)

set(YOLO_CPP_TEST_COMMON_SOURCES
  test_utils.hpp
  unit/test_classification_semantics.cpp
  unit/test_classification_preprocess_contract.cpp
  unit/test_classification_resize_geometry.cpp
  unit/test_classification_center_crop_geometry.cpp
  unit/test_image_preprocess.cpp
  unit/test_task_common_input_selection.cpp
  unit/test_task_common_metadata.cpp
  unit/test_task_common_provider.cpp
  unit/test_task_common_shape_validation.cpp
  unit/test_tensor_utils.cpp
  unit/test_pipeline_info.cpp
  component/test_detection_decode.cpp
  component/test_detection_decode_layouts.cpp
  component/test_detection_decode_validation.cpp
  component/test_detection_postprocess.cpp
  component/test_detection_postprocess_filtering.cpp
  component/test_detection_postprocess_nms.cpp
  component/test_detection_postprocess_restore.cpp
  component/test_classification_decode.cpp
  component/test_classification_resize_reference.cpp
  component/test_classification_resize_numerics.cpp
  component/test_segmentation_decode_contract.cpp
  component/test_segmentation_mask_projection.cpp
  adapter/test_ultralytics_detect_probe.cpp
  adapter/test_ultralytics_classify_probe.cpp
  adapter/test_ultralytics_seg_binding_contract.cpp
  adapter/test_task_hard_constraint.cpp
)

add_executable(yolo_cpp_tests ${YOLO_CPP_TEST_COMMON_SOURCES})
target_link_libraries(yolo_cpp_tests PRIVATE yolo_cpp Catch2::Catch2WithMain)
target_include_directories(yolo_cpp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
catch_discover_tests(yolo_cpp_tests)

set(YOLO_CPP_TEST_MODEL_DIR "${CMAKE_SOURCE_DIR}/tests/assets/models")
set(YOLO_CPP_TEST_DETECT_MODEL "${YOLO_CPP_TEST_MODEL_DIR}/yolov8n.onnx")
set(YOLO_CPP_TEST_CLASSIFY_MODEL "${YOLO_CPP_TEST_MODEL_DIR}/yolov8n-cls.onnx")
set(YOLO_CPP_TEST_SEGMENT_MODEL "${YOLO_CPP_TEST_MODEL_DIR}/yolov8n-seg.onnx")

if(onnxruntime_FOUND
   AND EXISTS "${YOLO_CPP_TEST_DETECT_MODEL}"
   AND EXISTS "${YOLO_CPP_TEST_CLASSIFY_MODEL}")
  set(YOLO_CPP_INTEGRATION_SOURCES
    integration/test_detect_pipeline.cpp
    integration/test_classify_pipeline.cpp
    integration/test_run_raw_guards.cpp
    test_utils.hpp
  )
  add_executable(
    yolo_cpp_integration_tests
    ${YOLO_CPP_INTEGRATION_SOURCES}
  )
  target_link_libraries(
    yolo_cpp_integration_tests PRIVATE yolo_cpp Catch2::Catch2WithMain
  )
  target_include_directories(
    yolo_cpp_integration_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_definitions(
    yolo_cpp_integration_tests
    PRIVATE
      YOLO_CPP_TEST_DETECT_MODEL="${YOLO_CPP_TEST_DETECT_MODEL}"
      YOLO_CPP_TEST_CLASSIFY_MODEL="${YOLO_CPP_TEST_CLASSIFY_MODEL}"
  )

  if(EXISTS "${YOLO_CPP_TEST_SEGMENT_MODEL}")
    target_sources(
      yolo_cpp_integration_tests
      PRIVATE
        integration/test_segment_pipeline.cpp
    )
    target_compile_definitions(
      yolo_cpp_integration_tests
      PRIVATE
        YOLO_CPP_TEST_SEGMENT_MODEL="${YOLO_CPP_TEST_SEGMENT_MODEL}"
    )
  endif()

  catch_discover_tests(yolo_cpp_integration_tests)
endif()

add_custom_target(
  check
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  DEPENDS yolo_cpp_tests
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

if(TARGET yolo_cpp_integration_tests)
  add_dependencies(check yolo_cpp_integration_tests)
endif()
