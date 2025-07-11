#pragma once

#include "Error.h"
#include "Types.h"

// std::expected is not in standard as of c++20 so rolling a simple
// version of it, all errors in this codebase is MRayError
// so no template for it.
//
// Piggybacking variant here for most of the construction stuff
template<class T>
struct Expected : protected Variant<T, MRayError>
{
    private:
    using Base = Variant<T, MRayError>;

    public:
    using Base::Base;

    // Provide semantically the same API,
    // we may switch this to actual std::expected later
    // (for better move semantics etc)
    // Utilize bare minimum subset of the API so refactoring will be easier
    explicit
    constexpr       operator bool() const noexcept;
    constexpr bool  has_value() const noexcept;
    constexpr bool  has_error() const noexcept;

    constexpr const T&  value() const;
    constexpr T&        value();

    constexpr const MRayError&  error() const noexcept;
    constexpr MRayError&        error() noexcept;

    // This is not technically 1 to 1
    // but it is more restrictive so ok
    constexpr T value_or(const T&) const;
};

template <class T>
constexpr Expected<T>::operator bool() const noexcept
{
    return has_error();
}

template <class T>
constexpr bool Expected<T>::has_value() const noexcept
{
    return !std::holds_alternative<MRayError>(*this);
}

template <class T>
constexpr bool Expected<T>::has_error() const noexcept
{
    return !has_value();
}

template <class T>
constexpr const T& Expected<T>::value() const
{
    return std::get<T>(*this);
}

template <class T>
constexpr T& Expected<T>::value()
{
    return std::get<T>(*this);
}

template <class T>
constexpr const MRayError& Expected<T>::error() const noexcept
{
    return std::get<MRayError>(*this);
}

template <class T>
constexpr MRayError& Expected<T>::error() noexcept
{
    return std::get<MRayError>(*this);
}

template <class T>
constexpr T Expected<T>::value_or(const T& t) const
{
    if(std::holds_alternative<MRayError>(*this))
        return t;
    else
        return std::get<T>(*this);
}