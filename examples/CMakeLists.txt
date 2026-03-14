set(YOLO_CPP_BUILD_PPM_EXAMPLES ON CACHE BOOL
    "Build minimal PPM-based examples")

if(YOLO_CPP_BUILD_PPM_EXAMPLES)
  add_executable(detect_image ppm_loader/detect_image.cpp)
  target_link_libraries(detect_image PRIVATE yolo_cpp)
  target_include_directories(
    detect_image PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(classify_image ppm_loader/classify_image.cpp)
  target_link_libraries(classify_image PRIVATE yolo_cpp)
  target_include_directories(
    classify_image PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(YOLO_CPP_BUILD_OPENCV_EXAMPLES)
  find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc)
  if(OpenCV_FOUND)
    add_executable(detect_opencv_viz opencv_viz/detect_image.cpp)
    target_link_libraries(
      detect_opencv_viz PRIVATE yolo_cpp ${OpenCV_LIBS}
    )
    target_include_directories(
      detect_opencv_viz
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )

    add_executable(pose_opencv_viz opencv_viz/pose_image.cpp)
    target_link_libraries(
      pose_opencv_viz PRIVATE yolo_cpp ${OpenCV_LIBS}
    )
    target_include_directories(
      pose_opencv_viz
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )
  else()
    message(STATUS
      "YOLO_CPP_BUILD_OPENCV_EXAMPLES=ON but OpenCV was not found. "
      "These are optional local visualization examples; install OpenCV "
      "and reconfigure if you want to build them.")
  endif()
endif()
