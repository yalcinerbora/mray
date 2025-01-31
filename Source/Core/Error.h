#pragma once

#include <string>

namespace fmt { inline namespace v10
{
    template <typename T> struct type_identity;
    template <typename T> using type_identity_t = typename type_identity<T>::type;

    template <typename Char, typename... Args> class basic_format_string;

    template <typename... Args>
    using format_string = basic_format_string<char, type_identity_t<Args>...>;

}}

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

    explicit operator bool() const;
    explicit operator bool();

    std::string GetError() const;
    void        AppendInfo(const std::string&);
};

inline MRayError::MRayError(Type t)
    : type(t)
{}

inline MRayError::MRayError(std::string_view sv)
    : type(MRayError::HAS_ERROR)
    , customInfo(sv)
{}

inline MRayError::operator bool() const
{
    return type != MRayError::OK;
}

inline MRayError::operator bool()
{
    return type != MRayError::OK;
}

inline std::string MRayError::GetError() const
{
    return customInfo;
}

#include "Error.hpp"
