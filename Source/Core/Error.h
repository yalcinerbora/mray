#pragma once

#include <string>
#include <array>
#include <atomic>
#include "Log.h"

// Very generic error
// Throw it return it, stick in a stew
struct MRayError
{
    public:
    enum Type
    {
        OK,
        HAS_ERROR
    };

    private:
    Type        type = OK;
    std::string customInfo;

    public:
    // Constructors & Destructor
                MRayError(Type = Type::OK);
                MRayError(std::string_view);
    // This pattern emerges a lot, so added as a template
    template<class... Args>
                MRayError(fmt::format_string<Args...> fstr, Args&&... args);
                MRayError(const MRayError&)     = default;
                MRayError(MRayError&&) noexcept = default;
    MRayError&  operator=(const MRayError&)     = default;
    MRayError&  operator=(MRayError&&) noexcept = default;
                ~MRayError()                    = default;

    explicit    operator bool() const;
    explicit    operator bool();

    std::string GetError() const;
    void        AppendInfo(const std::string&);
};

struct ErrorList
{
    private:
    static constexpr size_t MaxExceptionSize = 128;
    using ExceptionArray = std::array<MRayError, MaxExceptionSize>;

    public:
    std::atomic_size_t  size = 0;
    ExceptionArray      exceptions;
    void                AddException(MRayError&&);
};

template<class... Args>
MRayError::MRayError(fmt::format_string<Args...> fstr, Args&&... args)
    : type(MRayError::HAS_ERROR)
    , customInfo(MRAY_FORMAT(fstr, std::forward<Args>(args)...))
{}
