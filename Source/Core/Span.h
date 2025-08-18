#pragma once

#include "Tuple.h"
#include <span>

// Some span wrappers for convenience
template <class T, std::size_t Extent = std::dynamic_extent>
using Span = std::span<T, Extent>;

template<class T, std::size_t Extent = std::dynamic_extent>
constexpr Span<const T, Extent> ToConstSpan(Span<T, Extent> s);

//
template<class T0, std::size_t E0,
         class T1, std::size_t E1>
requires std::is_same_v<std::decay_t<T0>, std::decay_t<T1>>
constexpr bool IsSubspan(Span<T0, E0> checkedSpan, Span<T1, E1> bigSpan);

template<class T, std::size_t Extent>
constexpr Span<const T, Extent> ToConstSpan(Span<T, Extent>);

// SoA style span
// This may be usefull when DataSoA structures become too large
// due to holding size_t for all spans.
// Removed compiletime size portion since we rarely use it in this code base
// and probably increase compilation times
template<class... Args>
struct SoASpan
{
    using PtrTuple = Tuple<Args*...>;
    // Instead of doing tuple element over Tuple<Args...>
    // we do it over Tuple<Args*...> (notice the pointer)
    // and remove the pointer afterwards, it may be faster?
    // since we do not need to instantiate non-pointer tuple list
    // just to get the Ith element
    template<size_t I>
    using IthElement = std::remove_pointer_t<TupleElement<I, PtrTuple>>;

    private:
    PtrTuple ptrs = Tuple(static_cast<Args*>(nullptr)...);
    size_t   size = 0;

    public:
                    SoASpan() = default;
    template<class... Spans>
    constexpr       SoASpan(const Spans&... args);

    template<size_t I> constexpr auto Get() -> Span<IthElement<I>>;
    template<size_t I> constexpr auto Get() const -> Span<IthElement<I>>;
    //
    constexpr size_t Size() const;
};

// Deduction guide for constructor
template<class... Spans>
SoASpan(const Spans&... spans) -> SoASpan<typename Spans::element_type...>;

// TODO: add arrays maybe? (decay changes c arrays to ptrs)
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

template<class T, std::size_t Extent>
constexpr Span<const T, Extent> ToConstSpan(Span<T, Extent> s)
{
    return Span<const T, Extent>(s);
}

template<class... Args>
template<class... Spans>
constexpr SoASpan<Args...>::SoASpan(const Spans&... args)
    : ptrs(args.data()...)
    , size(get<0>(Tuple<Spans...>(args...)).size())
{
    assert(((args.size() == size) &&...));
}

template<class... Args>
template<size_t I>
constexpr auto SoASpan<Args...>::Get() -> Span<SoASpan<Args...>::IthElement<I>>
{
    using ResulT = Span<SoASpan<Args...>::template IthElement<I>>;
    return ResulT(get<I>(ptrs), size);
}

template<class... Args>
template<size_t I>
constexpr auto SoASpan<Args...>::Get() const -> Span<SoASpan<Args...>::IthElement<I>>
{
    using ResulT = const Span<SoASpan<Args...>::template IthElement<I>>;
    return ResulT(get<I>(ptrs), size);
}

template<class... Args>
constexpr size_t SoASpan<Args...>::Size() const
{
    return size;
}