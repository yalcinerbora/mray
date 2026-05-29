#pragma once

// And the time has come. Quote from the old code:
//
// "
// Rename the std::optional, gpu may not like it
// most(all after c++20) of optional is constexpr
// so the "relaxed-constexpr" flag of nvcc will be able to compile it
// Just to be sure, aliasing here to ease refactoring
//
// template <class T>
// using Optional = std::optional<T>;
// "
//
// HIP (clang of AMD) does not like throw obviously but also does not
// like the "_STL_VERIFY" on Windows. So we implement our own Optional.
//
// Simplified Optional type, for GPU usage. All constexpr
// ===========================================
// Plase check "Variant.h" for a warning.
// Probably not optimal or may have UB.
// ===========================================
//
// TODO: After checking the codebase
// we can use a form of optional
// like this one.
//
// template<class T, T IvalidMarker = T()>
// class OptionalStateless
// { T val; ...};
//
// This can be usefull implementation for optional reference
// and any type T that has user defined "invalid" state.
// Other Implementation Tactic:
//  - Instead of "InvalidMarker" which is the whole object,
//    we can utilize a function that compares subset of the type.
//
// (Or second template parameter can be a function that gets
// the object and returns bool, which is fancier)
//
// This can be useful when we use an Optional on device code
// (which we do quite a bit, and these types are trivial anyway)
// which maybe reduces register pressure (since we do not have an extra
// bool variable which occupies space if compiler could not optimize it).
//
// Dunno maybe another time. It will just improve the semantic meaning of the
// code.
//
#include "Types.h"
#include <memory>
#include <optional>
#include <type_traits>

namespace OptionalDetail
{
    template<class T>
    class OptionalRef
    {
        using InnerType = std::remove_reference_t<T>;
        static constexpr bool InnerConst = std::is_const_v<InnerType>;
        using BareType = std::remove_cv_t<InnerType>;

        private:
        InnerType* ptr;

        public:
        // Constructors & Destructor
        constexpr              OptionalRef();
        constexpr              OptionalRef(std::nullopt_t);
        constexpr              OptionalRef(InnerType&) noexcept;
        constexpr              OptionalRef(const InnerType&) noexcept requires(!InnerConst);
        constexpr OptionalRef& operator=(InnerType&) noexcept;
        constexpr OptionalRef& operator=(const InnerType&) noexcept requires(!InnerConst);
        constexpr OptionalRef& operator=(std::nullopt_t) noexcept;

        constexpr explicit     operator bool() const noexcept;
        constexpr bool         HasValue() const noexcept;

        // TODO: Overload these for move contexts
        constexpr InnerType&       Value();
        constexpr InnerType&       Value() const requires(InnerConst);
        constexpr const InnerType& Value() const requires(!InnerConst);

        constexpr InnerType&       ValueOr(InnerType&);
        constexpr InnerType&       ValueOr(InnerType&) const requires(InnerConst);
        constexpr const InnerType& ValueOr(const InnerType&) const requires(!InnerConst);
    };

    // This is due to my bad implementation
    // or a bug on NVCC (CUDA 13.0) that compiles incorrectly
    // For most GPU types this type overload should activate
    // and we will not bother with union stuff.
    template<class T>
    class OptionalPOD
    {
        private:
        using This = OptionalPOD;
        static_assert(!std::is_const_v<T>,
                      "Optional can't be instantiated with a \"const\" type");
        static_assert(!std::is_volatile_v<T>,
                      "Optional can't be instantiated with a \"volatile\" type");

        template<class In>
        static constexpr bool SameType = std::is_same_v<std::remove_cvref_t<In>, T>;

        private:
        T       data;
        bool    isActive;

        public:
        // Constructors & Destructor
        constexpr               OptionalPOD() noexcept;
        constexpr               OptionalPOD(std::nullopt_t) noexcept;
        template<class Input>
        constexpr               OptionalPOD(Input&&) noexcept  requires(SameType<Input>);
        template<class Input>
        constexpr OptionalPOD& operator=(Input&&) noexcept requires(SameType<Input>);
        template<class Input>
        explicit constexpr      OptionalPOD(const std::optional<Input>&) requires(SameType<Input>);
        template<class Input>
        explicit constexpr      OptionalPOD(std::optional<Input>&&) requires(SameType<Input>);

        // These all can be default
        constexpr       OptionalPOD(const This& other) noexcept = default;
        constexpr       OptionalPOD(This&& other) noexcept      = default;
        constexpr This& operator=(const This& other) noexcept   = default;
        constexpr This& operator=(This&& other) noexcept        = default;
        //
        constexpr explicit     operator bool() const noexcept;
        constexpr bool         HasValue() const noexcept;

        // TODO: Add fancy other contexts later
        constexpr T&       Value();
        constexpr const T& Value() const;
        constexpr T&       ValueOr(T&);
        constexpr const T& ValueOr(const T&) const;
    };

    // Now the big boy
    template<class T>
    class OptionalBase
    {
        private:
        static_assert(!std::is_const_v<T>,
                      "Optional can't be instantiated with a \"const\" type");
        static_assert(!std::is_volatile_v<T>,
                      "Optional can't be instantiated with a \"volatile\" type");

        using InnerType = std::remove_cvref_t<T>;

        static constexpr bool TD = std::is_trivially_destructible_v<InnerType>;
        // Logistics, if non-trivial, implement else do "=default"
        // and let the compiler take the wheel.
        static constexpr bool CC = (std::is_copy_constructible_v<InnerType> &&
                                    !std::is_trivially_copy_constructible_v<InnerType>);
        static constexpr bool MC = (std::is_move_constructible_v<InnerType> &&
                                    !std::is_trivially_move_constructible_v<InnerType>);
        static constexpr bool CA = (std::is_copy_assignable_v<InnerType> &&
                                    !std::is_trivially_copy_assignable_v<InnerType>);
        static constexpr bool MA = (std::is_move_assignable_v<InnerType> &&
                                    !std::is_trivially_move_assignable_v<InnerType>);

        template<class In>
        static constexpr bool SameType = std::is_same_v<std::remove_cvref_t<In>, T>;

        private:
        union
        {
            char empty;
            T    data;
        };
        bool isActive;

        public:
        // Constructors & Destructor
        // TODO: MSVC Bug? Cant define outside class. It says it is ambiguous
        constexpr ~OptionalBase() noexcept requires(!TD)
        {
            if(isActive) std::destroy_at(&data);
        }
        constexpr               OptionalBase() noexcept;
        constexpr               OptionalBase(std::nullopt_t) noexcept;
        template<class Input>
        constexpr               OptionalBase(Input&&) noexcept  requires(SameType<Input>);
        template<class Input>
        constexpr OptionalBase& operator=(Input&&) noexcept requires(SameType<Input>);
        template<class Input>
        explicit constexpr      OptionalBase(const std::optional<Input>&) requires(SameType<Input>);
        template<class Input>
        explicit constexpr      OptionalBase(std::optional<Input>&&) requires(SameType<Input>);
        // I did not bother for converting constructors/assigners
        // (i.e OptionalBase(Optional<Input>&&) etc.) Since rules are too heavy.
        // After porting the entire code base to this Optional, that constructors
        // are only used in one place (which was std::optional<std::string> -> std::optional<std::string_view>, and
        // it is scary semantics-wise).
        // Logistics
        constexpr               OptionalBase(const OptionalBase& other) noexcept requires(CC);
        constexpr               OptionalBase(OptionalBase&& other) noexcept      requires(MC);
        constexpr OptionalBase& operator=(const OptionalBase& other) noexcept    requires(CA);
        constexpr OptionalBase& operator=(OptionalBase&& other) noexcept         requires(MA);
        //
        constexpr               OptionalBase(const OptionalBase& other) noexcept requires(!CC) = default;
        constexpr               OptionalBase(OptionalBase&& other) noexcept      requires(!MC) = default;
        constexpr OptionalBase& operator=(const OptionalBase& other) noexcept    requires(!CA) = default;
        constexpr OptionalBase& operator=(OptionalBase&& other) noexcept         requires(!MA) = default;
        constexpr               ~OptionalBase() noexcept                         requires(TD)  = default;

        constexpr explicit     operator bool() const noexcept;
        constexpr bool         HasValue() const noexcept;

        // TODO: Add fancy other contexts later
        constexpr InnerType&       Value();
        constexpr const InnerType& Value() const;
        constexpr InnerType&       ValueOr(InnerType&);
        constexpr const InnerType& ValueOr(const InnerType&) const;
    };

    // Combination of the Type
    // Oh boy going full Allman here
    template<class T>
    using OptionalImpl = std::conditional_t
    <
        std::is_reference_v<T>,
        OptionalDetail::OptionalRef<T>,
        std::conditional_t
        <
            std::is_default_constructible_v<T>,
            OptionalDetail::OptionalPOD<T>,
            OptionalDetail::OptionalBase<T>
        >
    >;
}

template<class T>
struct Optional : public OptionalDetail::OptionalImpl<T>
{
    using Base = OptionalDetail::OptionalImpl<T>;

    public:
    using Base::Base;
};

namespace OptionalDetail
{

template<class T>
constexpr OptionalRef<T>::OptionalRef()
    : ptr{nullptr}
{}

template<class T>
constexpr OptionalRef<T>::OptionalRef(std::nullopt_t)
    : ptr{nullptr}
{}

template<class T>
constexpr OptionalRef<T>::OptionalRef(InnerType& t) noexcept
    : ptr{&t}
{}

template<class T>
constexpr OptionalRef<T>::OptionalRef(const InnerType& t) noexcept requires(!InnerConst)
    : ptr{&t}
{}

template<class T>
constexpr OptionalRef<T>&
OptionalRef<T>::operator=(InnerType& t) noexcept
{
    ptr = &t;
    return *this;
}

template<class T>
constexpr OptionalRef<T>&
OptionalRef<T>::operator=(const InnerType& t) noexcept requires(!InnerConst)
{
    ptr = &t;
    return *this;
}

template<class T>
constexpr OptionalRef<T>&
OptionalRef<T>::operator=(std::nullopt_t) noexcept
{
    ptr = nullptr;
    return *this;
}

template<class T>
constexpr OptionalRef<T>::operator bool() const noexcept
{
    return (ptr != nullptr);
}

template<class T>
constexpr
bool OptionalRef<T>::HasValue() const noexcept
{
    return (ptr != nullptr);
}

template<class T>
constexpr typename OptionalRef<T>::InnerType&
OptionalRef<T>::Value()
{
    assert(ptr != nullptr && "Null reference on optional!");
    return *ptr;
}

template<class T>
constexpr typename OptionalRef<T>::InnerType&
OptionalRef<T>::Value() const requires(InnerConst)
{
    assert(ptr != nullptr && "Null reference on optional!");
    return *ptr;
}

template<class T>
constexpr const typename OptionalRef<T>::InnerType&
OptionalRef<T>::Value() const requires(!InnerConst)
{
    assert(ptr != nullptr && "Null reference on optional!");
    return *ptr;
}

template<class T>
constexpr typename OptionalRef<T>::InnerType&
OptionalRef<T>::ValueOr(InnerType& t)
{
    return (ptr != nullptr) ? *ptr : t;
}

template<class T>
constexpr typename OptionalRef<T>::InnerType&
OptionalRef<T>::ValueOr(InnerType& t) const requires(InnerConst)
{
    return (ptr != nullptr) ? *ptr : t;
}

template<class T>
constexpr const typename OptionalRef<T>::InnerType&
OptionalRef<T>::ValueOr(const InnerType& t) const requires(!InnerConst)
{
    return (ptr != nullptr) ? *ptr : t;
}

}

namespace OptionalDetail
{

template<class T>
constexpr
OptionalPOD<T>::OptionalPOD() noexcept
    : isActive{false}
{}

template<class T>
constexpr
OptionalPOD<T>::OptionalPOD(std::nullopt_t) noexcept
    : isActive{false}
{}

template<class T>
template<class Input>
constexpr
OptionalPOD<T>::OptionalPOD(Input&& other) noexcept  requires(SameType<Input>)
    : data(std::forward<Input>(other))
    , isActive{true}
{}

template<class T>
template<class Input>
constexpr
OptionalPOD<T>& OptionalPOD<T>::operator=(Input&& other) noexcept requires(SameType<Input>)
{
    data = std::forward<Input>(other);
    isActive = true;
}

template<class T>
template<class Input>
constexpr
OptionalPOD<T>::OptionalPOD(const std::optional<Input>& in) requires(SameType<Input>)
{
    if(in.has_value()) data = in.value();
    isActive = in.has_value();
}

template<class T>
template<class Input>
constexpr
OptionalPOD<T>::OptionalPOD(std::optional<Input>&& in) requires(SameType<Input>)
{
    if(in.has_value()) data = std::move(in.value());
    isActive = in.has_value();
}

template<class T>
constexpr
OptionalPOD<T>::operator bool() const noexcept
{
    return isActive;
}

template<class T>
constexpr
bool OptionalPOD<T>::HasValue() const noexcept
{
    return isActive;
}

template<class T>
constexpr
T& OptionalPOD<T>::Value()
{
    assert(isActive && "Inactive data access on Optional!");
    return data;
}

template<class T>
constexpr
const T& OptionalPOD<T>::Value() const
{
    assert(isActive && "Inactive data access on Optional!");
    return data;
}

template<class T>
constexpr
T& OptionalPOD<T>::ValueOr(T& other)
{
    return (isActive) ? data : other;
}

template<class T>
constexpr
const T& OptionalPOD<T>::ValueOr(const T& other) const
{
    return (isActive) ? data : other;
}

}

namespace OptionalDetail
{

template<class T>
constexpr
OptionalBase<T>::OptionalBase() noexcept
    : empty{}
    , isActive{false}
{}

template<class T>
constexpr
OptionalBase<T>::OptionalBase(std::nullopt_t) noexcept
    : empty{}
    , isActive{false}
{}

template<class T>
template<class Input>
constexpr
OptionalBase<T>::OptionalBase(Input&& in) noexcept requires(SameType<Input>)
    : data(std::forward<Input>(in))
    , isActive{true}
{}

template<class T>
template<class Input>
constexpr
OptionalBase<T>&
OptionalBase<T>::operator=(Input&& in) noexcept requires(SameType<Input>)
{
    if(isActive)
        data = std::forward<Input>(in);
    else
        std::construct_at(&data, std::forward<Input>(in));
    isActive = true;
    return *this;
}

template<class T>
template<class Input>
constexpr
OptionalBase<T>::OptionalBase(const std::optional<Input>& in) requires(SameType<Input>)
    : isActive{in.has_value()}
{
    if(in) std::construct_at(&data, in.value());
}

template<class T>
template<class Input>
constexpr
OptionalBase<T>::OptionalBase(std::optional<Input>&& in) requires(SameType<Input>)
    : isActive{in.has_value()}
{
    if(in) std::construct_at(&data, std::move(in.value()));
}

template<class T>
constexpr
OptionalBase<T>::OptionalBase(const OptionalBase& other) noexcept requires(CC)
    : isActive{other.isActive}
{
    if(other.isActive) std::construct_at(&data, other.data);
}

template<class T>
constexpr
OptionalBase<T>::OptionalBase(OptionalBase&& other) noexcept requires(MC)
    : isActive(other.isActive)
{
    if(other.isActive)
    {
        std::construct_at(&data, std::move(other.data));
        std::destroy_at(&other.data);   // TODO: Is this correct?
        other.isActive = false;
    }
}

template<class T>
constexpr OptionalBase<T>&
OptionalBase<T>::operator=(const OptionalBase& other) noexcept requires(CA)
{
    if(isActive && other.isActive)
    {
        data = other.data;
    }
    else if(!isActive && other.isActive)
    {
        std::construct_at(&data, other.data);
        isActive = true;
    }
    else if(isActive && !other.isActive)
    {
        std::destroy_at(&data);
        isActive = false;
    }
    return *this;
}

template<class T>
constexpr OptionalBase<T>&
OptionalBase<T>::operator=(OptionalBase&& other) noexcept requires(MA)
{
    if(isActive && other.isActive)
    {
        data = std::move(other.data);
        std::destroy_at(&other.data);
        other.isActive = false;
    }
    else if(!isActive && other.isActive)
    {
        std::construct_at(&data, std::move(other.data));
        std::destroy_at(&other.data);
        isActive = true;
        other.isActive = false;
    }
    else if(isActive && !other.isActive)
    {
        std::destroy_at(&data);
        isActive = false;
    }
    return *this;
}

template<class T>
constexpr
OptionalBase<T>::operator bool() const noexcept
{
    return isActive;
}

template<class T>
constexpr
bool OptionalBase<T>::HasValue() const noexcept
{
    return isActive;
}

template<class T>
constexpr typename OptionalBase<T>::InnerType&
OptionalBase<T>::Value()
{
    assert(isActive && "Inactive data access on Optional!");
    return data;
}

template<class T>
constexpr
const typename OptionalBase<T>::InnerType&
OptionalBase<T>::Value() const
{
    assert(isActive && "Inactive data access on Optional!");
    return data;
}

template<class T>
constexpr
typename OptionalBase<T>::InnerType&
OptionalBase<T>::ValueOr(InnerType& other)
{
    return (isActive) ? data : other;
}

template<class T>
constexpr
const typename OptionalBase<T>::InnerType&
OptionalBase<T>::ValueOr(const InnerType& other) const
{
    return (isActive) ? data : other;
}

}