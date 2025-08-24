#pragma once

#include "Error.h"
#include "Types.h"

#include <memory>

// std::expected is not in standard as of c++20 so rolling a simple
// version of it, all errors in this codebase is MRayError
// so no template for it.
template<class Val>
struct Expected
{
    template<class T>
    static constexpr bool
    IsValConstructor = (!std::is_same_v<std::remove_cvref_t<T>, Expected> &&
                        !std::is_same_v<std::remove_cvref_t<T>, MRayError>);
    template<class T>
    static constexpr bool
    IsErrConstructor = (!std::is_same_v<std::remove_cvref_t<T>, Expected> &&
                        std::is_same_v<std::remove_cvref_t<T>, MRayError>);

    enum class State : uint8_t
    {
        VALUELESS,
        HAS_ERROR,
        HAS_VALUE
    };

    // TODO: Defensively checking for this without any reason.
    // Having an expected error is not logical. But if
    // it is required later we can remove this check.
    static_assert(!std::is_same_v<Val, MRayError>,
                  "Expected can not hold MRayError as its "
                  "template parameter");
    private:
    union
    {
        MRayError   err;
        Val         val;
    };
    State state = State::VALUELESS;

    public:
    // Constructors
    // Enum Constructor
                Expected(MRayError::Type E);
    // Data Constructor
    template<class T>
                Expected(T&& v) requires(IsValConstructor<T>);
    // We also define Error as template so it gets forwarded properly
    template<class E>
                Expected(E&& e) requires(IsErrConstructor<E>);
    // Logistics
                Expected(const Expected&);
                Expected(Expected&&);
    Expected&   operator=(const Expected&);
    Expected&   operator=(Expected&&);
                ~Expected();

    // Provide semantically the same API,
    // we may switch this to actual std::expected later
    // (for better move semantics etc)
    // Utilize bare minimum subset of the API so refactoring will be easier
    explicit
    constexpr       operator bool() const noexcept;
    constexpr bool  has_value() const noexcept;
    constexpr bool  has_error() const noexcept;

    constexpr const Val&  value() const;
    constexpr Val&        value();

    constexpr const MRayError&  error() const noexcept;
    constexpr MRayError&        error() noexcept;

    // This is not technically 1 to 1
    // but it is more restrictive so ok
    constexpr Val value_or(const Val&) const;
};

template <class V>
Expected<V>::Expected(MRayError::Type E)
    : err(E)
    , state(State::HAS_ERROR)
{}

// Error Constructor
template<class V>
template<class T>
Expected<V>::Expected(T&& v) requires(IsValConstructor<T>)
{
    std::construct_at(&val, std::forward<T>(v));
    state = State::HAS_VALUE;
}

// We also define Error as template so it gets forwarded properly
template<class V>
template<class E>
Expected<V>::Expected(E&& e) requires(IsErrConstructor<E>)
{
    std::construct_at(&err, std::forward<E>(e));
    state = State::HAS_ERROR;
}

template<class V>
Expected<V>::Expected(const Expected& other)
{
         if(state == State::HAS_ERROR) err = other.err;
    else if(state == State::HAS_VALUE) val = other.val;
    //
    state = other.state;
}

template<class V>
Expected<V>::Expected(Expected&& other)
{
         if(state == State::HAS_ERROR) err = std::move(other.err);
    else if(state == State::HAS_VALUE) val = std::move(other.val);
    //
    state = other.state;
    other.state = State::VALUELESS;
}

template<class V>
Expected<V>& Expected<V>::operator=(const Expected& other)
{
    assert(this != &other);
         if(state == State::HAS_ERROR) std::destroy_at(&err);
    else if(state == State::HAS_VALUE) std::destroy_at(&val);
    //
         if(state == State::HAS_ERROR) err = other.err;
    else if(state == State::HAS_VALUE) val = other.val;
    //
    state = other.state;
    return *this;
}

template<class V>
Expected<V>& Expected<V>::operator=(Expected&& other)
{
    assert(this != &other);
         if(state == State::HAS_ERROR) std::destroy_at(&err);
    else if(state == State::HAS_VALUE) std::destroy_at(&val);
    //
         if(state == State::HAS_ERROR) err = std::move(other.err);
    else if(state == State::HAS_VALUE) val = std::move(other.val);
    //
    state = other.state;
    other.state = State::VALUELESS;
    return *this;
}

template<class V>
Expected<V>::~Expected()
{
         if(state == State::HAS_ERROR) std::destroy_at(&err);
    else if(state == State::HAS_VALUE) std::destroy_at(&val);
}

template <class V>
constexpr Expected<V>::operator bool() const noexcept
{
    return has_error();
}

template <class V>
constexpr bool Expected<V>::has_value() const noexcept
{
    return state == State::HAS_VALUE;
}

template <class V>
constexpr bool Expected<V>::has_error() const noexcept
{
    return state == State::HAS_ERROR;
}

template <class V>
constexpr const V& Expected<V>::value() const
{
    assert(has_value());
    return val;
}

template <class V>
constexpr V& Expected<V>::value()
{
    assert(has_value());
    return val;
}

template <class V>
constexpr const MRayError& Expected<V>::error() const noexcept
{
    assert(has_error());
    return err;
}

template <class V>
constexpr MRayError& Expected<V>::error() noexcept
{
    assert(has_error());
    return err;
}

template <class V>
constexpr V Expected<V>::value_or(const V& t) const
{
    if(state == State::HAS_VALUE)   return val;
    else                            return t;
}