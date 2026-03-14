#pragma once

#include <cstddef>
#include <iostream>
#include <string_view>

#include "yolo/core/error.hpp"

namespace examples
{

inline int print_error(const yolo::Error& error) {
    std::cerr << "error: " << error.message << '\n';
    if (error.context.has_value()) {
        if (error.context->component.has_value()) {
            std::cerr << "component: " << *error.context->component << '\n';
        }
        if (error.context->input_name.has_value()) {
            std::cerr << "input: " << *error.context->input_name << '\n';
        }
        if (error.context->output_name.has_value()) {
            std::cerr << "output: " << *error.context->output_name << '\n';
        }
        if (error.context->expected.has_value()) {
            std::cerr << "expected: " << *error.context->expected << '\n';
        }
        if (error.context->actual.has_value()) {
            std::cerr << "actual: " << *error.context->actual << '\n';
        }
    }

    return 1;
}

inline void print_pipeline_summary(std::string_view adapter_name,
                                   std::string_view task_name,
                                   std::size_t input_count,
                                   std::size_t output_count) {
    std::cout << "adapter: " << adapter_name << '\n';
    std::cout << "task: " << task_name << '\n';
    std::cout << "inputs: " << input_count << ", outputs: " << output_count
              << '\n';
}

}  // namespace examples
