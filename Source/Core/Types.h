#pragma once

#include <span>
#include <optional>
#include <variant>
#include <vector>
#include <functional>

#include "MathFunctions.h"
#include "Error.h"

// Rename the std::optional, gpu may not like it
// most(all after c++20) of optional is constexpr
// so the "relaxed-constexpr" flag of nvcc will be able to compile it
// Just to be sure, aliasing here to ease refactoring
template <class T>
using Optional = std::optional<T>;

template <class T, std::size_t Extent = std::dynamic_extent>
using Span = std::span<T, Extent>;

template <class... Types>
using Variant = std::variant<Types...>;

template <class T0, class T1>
using Pair = std::pair<T0, T1>;

template <class... Args>
using Tuple = std::tuple<Args...>;

// TODO: reference_wrapper<T> vs. span<T,1> which is better?
template <class T>
using Ref = std::reference_wrapper<T>;

// First failure related to the concerns described above
// nvcc did not like it (see: https://godbolt.org/z/fM811b4cx)
// Thankfully it was not the actual variant implementation, I dunno how to
// implement it :(
// Some basic implementation of std::visit (recursive invocation)
// (Above link shows the compiled down result, it decays to nothing when all variables
// are known at compile time.)
namespace detail
{
    template<uint32_t I, class VariantT, class Func>
    requires(I < std::variant_size_v<std::remove_reference_t<VariantT>>)
    MRAY_HYBRID
    constexpr auto LoopAndInvoke(VariantT&& v, Func&& f) -> decltype(auto);

    template<class... Args, std::size_t... I>
    MRAY_HYBRID
    constexpr Tuple<Args&...> ToTupleRef(Tuple<Args...>&t,
                                         std::index_sequence<I...>);
}

template<class VariantT, class Func>
MRAY_HYBRID
constexpr auto DeviceVisit(VariantT&& v, Func&& f) -> decltype(auto);

template<class... Args, typename Indices = std::index_sequence_for<Args...>>
MRAY_HYBRID
constexpr Tuple<Args&...> ToTupleRef(Tuple<Args...>& t);

// Some span wrappers for convenience
template<class T, std::size_t Extent = std::dynamic_extent>
constexpr Span<const T, Extent> ToConstSpan(Span<T, Extent> s);

//
template<class T0, std::size_t E0,
         class T1, std::size_t E1>
requires std::is_same_v<std::decay_t<T0>, std::decay_t<T1>>
constexpr bool IsSubspan(Span<T0, E0> checkedSpan, Span<T1, E1> bigSpan);

// This is definately a thing that i cannot even comprahend
// Thanks to:
// https://stackoverflow.com/questions/55941964/how-to-filter-duplicate-types-from-tuple-c
namespace UniqueVariantDetail
{
template <typename T, typename... Ts>
struct Unique : std::type_identity<T> {};

template <typename... Ts, typename U, typename... Us>
struct Unique<Variant<Ts...>, U, Us...>
    : std::conditional_t<(std::is_same_v<U, Ts> || ...)
    , Unique<Variant<Ts...>, Us...>
    , Unique<Variant<Ts..., U>, Us...>> {};

}

template <typename... Ts>
using UniqueVariant = typename UniqueVariantDetail::Unique<Variant<>, Ts...>::type;

template <class T>
struct SampleT
{
    T           sampledResult;
    Float       pdf;
};

// SoA style span
// This may be usefull when DataSoA structures become too large
// due to holding size_t for all spans. Currently not used.
template<class... Args>
struct SoASpan
{
    constexpr Tuple<Args*...> NullifyPtrs();

    private:
    // TODO: Find a way to default initialize this.
    // My template metaprogramming capabilities was not enough.
    // We are setting size to zero at least.
    Tuple<Args*...> ptrs = NullifyPtrs();
    size_t          size = 0;

    public:
            SoASpan() = default;
    template<class... Spans>
            SoASpan(const Spans&... args);

    template<size_t I>
    auto    Get() -> Span<std::tuple_element_t<I, Tuple<Args...>>>;

    template<size_t I>
    auto    Get() const -> Span<std::tuple_element_t<I, Tuple<const Args...>>>;
};

// std::expected is not in standart as of c++20 so rolling a simple
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
    // we may switch this to actual std::excpected later
    // (for better move semantics etc)
    // Utilize bare minimum subset of the API so refactoring will be easier
    explicit
    constexpr operator  bool() const noexcept;
    constexpr bool      has_value() const noexcept;
    constexpr bool      has_error() const noexcept;

    constexpr const T&  value() const;
    constexpr T&        value();

    constexpr const MRayError&  error() const noexcept;
    constexpr MRayError&        error() noexcept;

    // This is not technically 1 to 1
    // but it is more restictive so ok
    constexpr T         value_or(const T&) const;
};

template<uint32_t I, class VariantT, class Func>
requires(I < std::variant_size_v<std::remove_reference_t<VariantT>>)
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr auto detail::LoopAndInvoke(VariantT&& v, Func&& f) -> decltype(auto)
{
    using CurrentType = decltype(std::get<I>(v));

    if(I == v.index())
        return std::invoke(f, std::forward<CurrentType>(std::get<I>(v)));
    else if constexpr(I < std::variant_size_v<std::remove_reference_t<VariantT>> -1)
        return LoopAndInvoke<I + 1>(std::forward<VariantT>(v), std::forward<Func>(f));
    else
    {
        #ifdef __CUDA_ARCH__
            if constexpr (MRAY_IS_DEBUG)
                printf("Invalid variant access on device!\n");
        __trap();
        #else
            throw MRayError("Invalid variant access on device!");
        #endif
    }
}

template<class... Args, std::size_t... I>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Tuple<Args&...> detail::ToTupleRef(Tuple<Args...>&t,
                                             std::index_sequence<I...>)
{
    return std::tie(std::get<I>(t)...);
};

template<class VariantT, class Func>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr auto DeviceVisit(VariantT&& v, Func&& f) -> decltype(auto)
{
    return detail::LoopAndInvoke<0>(std::forward<VariantT>(v), std::forward<Func>(f));
}

template<class... Args, typename Indices>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Tuple<Args&...> ToTupleRef(Tuple<Args...>& t)
{
    return detail::ToTupleRef(t, Indices{});
}

template<class T, std::size_t Extent>
constexpr Span<const T, Extent> ToConstSpan(Span<T, Extent> s)
{
    return Span<const T, Extent>(s);
}

// TODO add arrays maybe? (decay changes c arrays to ptrs)
template<class T0, std::size_t E0,
         class T1, std::size_t E1>
requires std::is_same_v<std::decay_t<T0>, std::decay_t<T1>>
constexpr bool IsSubspan(Span<T0, E0> checkedSpan, Span<T1, E1> bigSpan)
{
    ptrdiff_t diff = checkedSpan.data() - bigSpan.data();
    if(diff >= 0)
    {
        size_t diffS = static_cast<size_t>(diff);
        bool ptrInRange = diffS < bigSpan.size();
        bool backInRange = (diffS + checkedSpan.size()) <= bigSpan.size();
        return (ptrInRange && backInRange);
    }
    else return false;
}

template<class... Args>
constexpr Tuple<Args*...> SoASpan<Args...>::NullifyPtrs()
{
    Tuple<Args*...> result;
    std::apply([](auto&&... args)
    {
        ((args = nullptr), ...);
    }, result);
    return result;
}

template<class... Args>
template<class... Spans>
SoASpan<Args...>::SoASpan(const Spans&... args)
    : ptrs(args.data()...)
    , size(std::get<0>(Tuple<Spans...>(args...)).size())
{
    assert((args.size() == size) &&...);
}

template<class... Args>
template<size_t I>
auto SoASpan<Args...>::Get() -> Span<std::tuple_element_t<I, Tuple<Args...>>>
{
    using ResulT = Span<std::tuple_element_t<I, Tuple<Args...>>>;
    return ResulT(std::get<I>(ptrs), size);
}

template<class... Args>
template<size_t I>
auto SoASpan<Args...>::Get() const -> Span<std::tuple_element_t<I, Tuple<const Args...>>>
{
    using ResulT = Span<std::tuple_element_t<I, Tuple<const Args...>>>;
    return ResulT(std::get<I>(ptrs), size);
}

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