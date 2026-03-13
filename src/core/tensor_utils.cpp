#include "yolo/detail/tensor_utils.hpp"

#include <sstream>

namespace yolo::detail
{

std::size_t tensor_element_size(TensorDataType data_type) {
    switch (data_type) {
        case TensorDataType::boolean:
        case TensorDataType::uint8:
        case TensorDataType::int8:
            return 1;
        case TensorDataType::int16:
        case TensorDataType::float16:
            return 2;
        case TensorDataType::int32:
        case TensorDataType::float32:
            return 4;
        case TensorDataType::int64:
        case TensorDataType::float64:
            return 8;
    }

    return 0;
}

std::optional<std::size_t> dense_byte_count(const TensorInfo& info) {
    const std::optional<std::size_t> element_count = info.shape.element_count();
    if (!element_count.has_value()) {
        return std::nullopt;
    }

    return *element_count * tensor_element_size(info.data_type);
}

std::string format_shape(const TensorShape& shape) {
    std::ostringstream stream;
    stream << '[';

    for (std::size_t i = 0; i < shape.dims.size(); ++i) {
        if (i != 0) {
            stream << ", ";
        }

        if (shape.dims[i].value.has_value()) {
            stream << *shape.dims[i].value;
        }
        else {
            stream << '?';
        }
    }

    stream << ']';
    return stream.str();
}

std::string format_data_type(TensorDataType data_type) {
    switch (data_type) {
        case TensorDataType::boolean:
            return "bool";
        case TensorDataType::uint8:
            return "uint8";
        case TensorDataType::int8:
            return "int8";
        case TensorDataType::int16:
            return "int16";
        case TensorDataType::int32:
            return "int32";
        case TensorDataType::int64:
            return "int64";
        case TensorDataType::float16:
            return "float16";
        case TensorDataType::float32:
            return "float32";
        case TensorDataType::float64:
            return "float64";
    }

    return "unknown";
}

Error make_shape_error(std::string_view component, std::string_view name,
                       const TensorShape& expected, const TensorShape& actual) {
    return make_error(ErrorCode::shape_mismatch, "Tensor shape mismatch.",
                      ErrorContext{
                          .component = std::string{component},
                          .output_name = std::string{name},
                          .expected = format_shape(expected),
                          .actual = format_shape(actual),
                      });
}

Error make_type_error(std::string_view component, std::string_view name,
                      TensorDataType expected, TensorDataType actual) {
    return make_error(ErrorCode::type_mismatch, "Tensor element type mismatch.",
                      ErrorContext{
                          .component = std::string{component},
                          .output_name = std::string{name},
                          .expected = format_data_type(expected),
                          .actual = format_data_type(actual),
                      });
}

}  // namespace yolo::detail
