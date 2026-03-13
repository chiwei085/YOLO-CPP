#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace yolo
{

// Public API returns status-style results. Callers that prefer exceptions can
// opt in by using throw_if_error(), which converts Error into YoloException.
enum class ErrorCode
{
    ok,
    invalid_argument,
    invalid_state,
    unsupported_task,
    unsupported_model,
    shape_mismatch,
    type_mismatch,
    io_error,
    backend_error,
    decode_error,
    runtime_error,
    internal_error,
    not_implemented,
};

struct ErrorContext
{
    std::optional<std::string> component{};
    std::optional<std::string> input_name{};
    std::optional<std::string> output_name{};
    std::optional<std::string> expected{};
    std::optional<std::string> actual{};
};

struct Error
{
    ErrorCode code{ErrorCode::ok};
    std::string message{};
    std::optional<ErrorContext> context{};

    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == ErrorCode::ok;
    }

    [[nodiscard]] explicit constexpr operator bool() const noexcept {
        return !ok();
    }
};

template <class T>
struct Result
{
    std::optional<T> value{};
    Error error{};

    [[nodiscard]] constexpr bool ok() const noexcept { return error.ok(); }
};

class YoloException : public std::runtime_error
{
public:
    explicit YoloException(Error error);

    [[nodiscard]] const Error& error() const noexcept { return error_; }

private:
    Error error_;
};

[[nodiscard]] Error make_error(
    ErrorCode code, std::string_view message,
    std::optional<ErrorContext> context = std::nullopt);

void throw_if_error(const Error& error);

}  // namespace yolo
