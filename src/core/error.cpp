#include "yolo/core/error.hpp"

#include <optional>
#include <utility>

namespace yolo
{

YoloException::YoloException(Error error)
    : std::runtime_error(error.message), error_(std::move(error)) {}

Error make_error(ErrorCode code, std::string_view message,
                 std::optional<ErrorContext> context) {
    return Error{code, std::string{message}, std::move(context)};
}

void throw_if_error(const Error& error) {
    if (!error.ok()) {
        throw YoloException(error);
    }
}

}  // namespace yolo
