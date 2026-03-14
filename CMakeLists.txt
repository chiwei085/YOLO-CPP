cmake_minimum_required(VERSION 3.20)

project(yolo_cpp VERSION 0.1.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(YOLO_CPP_BUILD_EXAMPLES "Build example executables" ON)
option(YOLO_CPP_BUILD_DIAGNOSTICS "Build diagnostic probes" ON)
include(CTest)

set(YOLO_CPP_ORT_PROVIDER "cpu" CACHE STRING
    "Requested ONNX Runtime provider package: cpu or cuda")
set_property(CACHE YOLO_CPP_ORT_PROVIDER PROPERTY STRINGS cpu cuda)

add_library(yolo_cpp
  src/core/engine.cpp
  src/core/facade.cpp
  src/core/onnx_session.cpp
  src/core/image_preprocess.cpp
  src/core/classification_resize.cpp
  src/core/pipeline_info_utils.cpp
  src/core/tensor_utils.cpp
  src/core/task_runtime_utils.cpp
  src/core/error.cpp
  src/tasks/detection.cpp
  src/tasks/detection_decode.cpp
  src/tasks/detection_postprocess.cpp
  src/tasks/classification.cpp
  src/tasks/segmentation.cpp
  src/tasks/pose.cpp
  src/tasks/obb.cpp
  src/adapters/ultralytics_detect.cpp
  src/adapters/ultralytics_classify.cpp
  src/adapters/ultralytics_seg.cpp
  src/adapters/ultralytics_pose.cpp
  src/adapters/ultralytics_obb.cpp
)

target_compile_features(yolo_cpp PUBLIC cxx_std_20)

target_include_directories(yolo_cpp
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

string(TOLOWER "${YOLO_CPP_ORT_PROVIDER}" YOLO_CPP_ORT_PROVIDER_NORMALIZED)
if(NOT YOLO_CPP_ORT_PROVIDER_NORMALIZED STREQUAL "cpu"
   AND NOT YOLO_CPP_ORT_PROVIDER_NORMALIZED STREQUAL "cuda")
  message(FATAL_ERROR
    "YOLO_CPP_ORT_PROVIDER must be either 'cpu' or 'cuda', got: "
    "${YOLO_CPP_ORT_PROVIDER}")
endif()

set(YOLO_CPP_ORT_EXPECTED_PACKAGE "onnxruntime")
if(YOLO_CPP_ORT_PROVIDER_NORMALIZED STREQUAL "cuda")
  set(YOLO_CPP_ORT_EXPECTED_PACKAGE "onnxruntime-gpu")
endif()

function(yolo_cpp_apply_ort_include_compat target_name)
  get_target_property(
    YOLO_CPP_ORT_INCLUDE_DIRS onnxruntime::onnxruntime INTERFACE_INCLUDE_DIRECTORIES
  )
  foreach(YOLO_CPP_ORT_INCLUDE_DIR IN LISTS YOLO_CPP_ORT_INCLUDE_DIRS)
    if(YOLO_CPP_ORT_INCLUDE_DIR MATCHES "/include/onnxruntime/?$")
      cmake_path(GET YOLO_CPP_ORT_INCLUDE_DIR PARENT_PATH YOLO_CPP_ORT_INCLUDE_ROOT)
      if(EXISTS "${YOLO_CPP_ORT_INCLUDE_ROOT}/onnxruntime/onnxruntime_cxx_api.h")
        target_include_directories(${target_name}
          PRIVATE
            "${YOLO_CPP_ORT_INCLUDE_ROOT}"
        )
      endif()
    endif()
  endforeach()
endfunction()

find_package(onnxruntime CONFIG QUIET
  PATHS
    ${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/onnxruntime
)

if(onnxruntime_FOUND)
  target_compile_definitions(yolo_cpp PRIVATE YOLO_CPP_HAS_ONNXRUNTIME=1)
  if(YOLO_CPP_ORT_PROVIDER_NORMALIZED STREQUAL "cuda")
    target_compile_definitions(yolo_cpp PRIVATE YOLO_CPP_ORT_WITH_CUDA=1)
  else()
    target_compile_definitions(yolo_cpp PRIVATE YOLO_CPP_ORT_WITH_CUDA=0)
  endif()

  target_link_libraries(yolo_cpp PRIVATE onnxruntime::onnxruntime)
  yolo_cpp_apply_ort_include_compat(yolo_cpp)

  message(STATUS
    "Using ONNX Runtime backend (${YOLO_CPP_ORT_PROVIDER_NORMALIZED}) "
    "via imported target onnxruntime::onnxruntime")
else()
  target_compile_definitions(
    yolo_cpp
    PRIVATE
      YOLO_CPP_HAS_ONNXRUNTIME=0
      YOLO_CPP_ORT_WITH_CUDA=0
  )

  message(STATUS
    "ONNX Runtime was not found. Expected vcpkg feature/package: "
    "${YOLO_CPP_ORT_EXPECTED_PACKAGE}. "
    "Configure with a vcpkg preset to install it automatically.")
endif()

if(CMAKE_BINARY_DIR MATCHES "/build/dev$" OR CMAKE_BINARY_DIR MATCHES "/build/cuda$")
  file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/build")
  file(CREATE_LINK
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_SOURCE_DIR}/build/compile_commands.json"
    SYMBOLIC
  )
endif()

if(YOLO_CPP_BUILD_EXAMPLES)
  add_executable(detect_image examples/detect_image.cpp)
  target_link_libraries(detect_image PRIVATE yolo_cpp)
  target_include_directories(
    detect_image PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)

  add_executable(classify_image examples/classify_image.cpp)
  target_link_libraries(classify_image PRIVATE yolo_cpp)
  target_include_directories(
    classify_image PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
endif()

if(YOLO_CPP_BUILD_DIAGNOSTICS AND onnxruntime_FOUND)
  add_subdirectory(tools)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
