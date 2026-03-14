function(yolo_cpp_configure_tool target_name)
  target_compile_features(${target_name} PRIVATE cxx_std_20)
  set_target_properties(${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
endfunction()

add_executable(ort_session_probe dev/ort_session_probe.cpp)
yolo_cpp_configure_tool(ort_session_probe)
target_link_libraries(ort_session_probe PRIVATE onnxruntime::onnxruntime)
yolo_cpp_apply_ort_include_compat(ort_session_probe)

add_executable(onnx_session_probe dev/onnx_session_probe.cpp)
yolo_cpp_configure_tool(onnx_session_probe)
target_link_libraries(onnx_session_probe PRIVATE yolo_cpp)

add_executable(yolo_cpp_parity_dump parity/yolo_cpp_parity_dump.cpp)
yolo_cpp_configure_tool(yolo_cpp_parity_dump)
target_link_libraries(yolo_cpp_parity_dump PRIVATE yolo_cpp)
target_include_directories(
  yolo_cpp_parity_dump PRIVATE ${CMAKE_SOURCE_DIR}/examples
)

add_executable(
  yolo_cpp_segmentation_debug_dump
  debug/yolo_cpp_segmentation_debug_dump.cpp
)
yolo_cpp_configure_tool(yolo_cpp_segmentation_debug_dump)
target_link_libraries(yolo_cpp_segmentation_debug_dump PRIVATE yolo_cpp)
target_include_directories(
  yolo_cpp_segmentation_debug_dump PRIVATE ${CMAKE_SOURCE_DIR}/examples
)

add_executable(yolo_cpp_pose_debug_dump debug/yolo_cpp_pose_debug_dump.cpp)
yolo_cpp_configure_tool(yolo_cpp_pose_debug_dump)
target_link_libraries(yolo_cpp_pose_debug_dump PRIVATE yolo_cpp)
target_include_directories(
  yolo_cpp_pose_debug_dump PRIVATE ${CMAKE_SOURCE_DIR}/examples
)

add_executable(yolo_cpp_obb_debug_dump debug/yolo_cpp_obb_debug_dump.cpp)
yolo_cpp_configure_tool(yolo_cpp_obb_debug_dump)
target_link_libraries(yolo_cpp_obb_debug_dump PRIVATE yolo_cpp)
target_include_directories(
  yolo_cpp_obb_debug_dump PRIVATE ${CMAKE_SOURCE_DIR}/examples
)
