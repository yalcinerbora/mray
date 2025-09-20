#pragma once

#include "Tuple.h"
#include "Definitions.h"
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <type_traits>

// We also implement our span. When checking -ftime-trace of
// clang, a single span instantiation takes 33ms!
// Big culprit is the reverse iterator, and it is an adapter
// so I dunno what is going on.
//
// Anyway, since we implement our own, we also do some changes
// to ease the register usage on GPUs. Size parameter is
// 32-bit so it does not allocate 2 registers on GPU (potentially,,
// probably NVCC implementors do special stuff maybe?)
//
// Second big change is span automatically is a restricted
// pointer. This fits our design since we do not alias memory
// with a multiple pointers.
//
// Third change is the removal the dynamic extent mark is zero
// and zero-sized static extent spans are not allowed.
// This is purely cosmetic, when compilation error / demangled
// name is encountered std::numeric_limits<>::max() pollutes
// the errors etc.
static constexpr uint32_t DynamicExtent = uint32_t(0);
//
namespace SpanDetail
{
    template <class T>
    struct alignas(16) DSpan
    {
        T* MRAY_RESTRICT ptr;
        uint32_t         count;
    };

    template <class T, uint32_t Extent>
    struct SSpan
    {
        T* MRAY_RESTRICT          ptr;
        static constexpr uint32_t count = Extent;
    };

    // This is not available until C++23
    template<class It>
    struct IsConstIteratorV
    {
        using P = typename std::iterator_traits<It>::pointer;
        static const bool value = std::is_const<std::remove_pointer_t<P>>::value;
    };

    template<class It>
    inline constexpr bool IsConstIterator = IsConstIteratorV<It>::value;
}

template<class It>
concept SpanIt = std::contiguous_iterator<It> && !SpanDetail::IsConstIterator<It>;

template<class It>
concept ConstSpanIt = std::contiguous_iterator<It> && SpanDetail::IsConstIterator<It>;

//
template <class T, uint32_t Extent = DynamicExtent>
class Span : public std::conditional_t<Extent == DynamicExtent,
                                       SpanDetail::DSpan<T>,
                                       SpanDetail::SSpan<T, Extent>>
{
    public:
    using Type = T;
    using NonConstType = std::remove_const_t<T>;
    //
    static constexpr bool IsConstPtr = std::is_const_v<Type>;
    static constexpr bool IsDynamic  = (Extent == DynamicExtent);

    public:
    // Constructors & Destructor
    constexpr           Span() requires(IsDynamic);
    // Ptr and Size
    explicit constexpr  Span(T* data, size_t size) noexcept;
    explicit constexpr  Span(NonConstType* data, size_t size) noexcept requires(IsConstPtr);
    // Iterator Pair
    template<SpanIt It>
    explicit constexpr  Span(It begin, It end) noexcept;
    template<ConstSpanIt It>
    explicit constexpr  Span(It begin, It end) noexcept;
    // Array and Vector Access
    template<std::size_t N>
    constexpr           Span(std::array<T, N>&) noexcept;
    template<std::size_t N>
    constexpr           Span(const std::array<NonConstType, N>&) noexcept requires(IsConstPtr);
    template<std::size_t N>
    constexpr           Span(T(&arr)[N]) noexcept;
    template<std::size_t N>
    constexpr           Span(NonConstType(&arr)[N]) noexcept requires(IsConstPtr);
    //
    constexpr           Span(std::vector<T>&) noexcept;
    constexpr           Span(const std::vector<NonConstType>&) noexcept requires(IsConstPtr);
    // Special copy constructor like syntax for converting to other spans
    constexpr           Span(const Span<NonConstType, Extent>&) noexcept requires(IsConstPtr);
    //
    template<uint32_t N>
    explicit constexpr  Span(const Span<T, N>&) noexcept requires(Extent != N);
    template<uint32_t N>
    explicit constexpr  Span(const Span<NonConstType, N>&) noexcept requires((Extent != N) && IsConstPtr);
    //
    template<uint32_t N>
    explicit constexpr  Span(const Span<T, N>&) noexcept
    requires(IsDynamic && !Span<T, N>::IsDynamic);
    // Logistics
                        Span(const Span&) = default;
                        Span(Span&&) = default;
    Span&               operator=(const Span&) = default;
    Span&               operator=(Span&&) = default;
                        ~Span() = default;
    // Access
    constexpr T&        operator[](uint32_t index) const;
    constexpr T*        data() const;
    constexpr T&        front() const;
    constexpr T&        back() const;
    // Iterator
    constexpr T*        begin() const;
    constexpr const T*  cbegin() const;
    //
    constexpr T*        end() const;
    constexpr const T*  cend() const;
    // Query
    constexpr uint32_t  size() const;
    constexpr uint32_t  size_bytes() const;
    constexpr bool      empty() const;
    // Subspan
    constexpr Span<T>   subspan(size_t offset, size_t count = UINT32_MAX) const;
    //
    template<uint32_t Offset, uint32_t Count = UINT32_MAX>
    constexpr Span<T, (Count == SIZE_MAX) ? Extent - uint32_t(Offset) : uint32_t(Count)>
    subspan() const
    requires(!IsDynamic && (Count) > 0)
    {
        static_assert(Offset + Count <= Extent);
        return Span<T, Count - Offset>(this->ptr + Offset, Extent - Offset);
    }
};

template<ConstSpanIt It>
Span(It begin, It end) -> Span<const typename std::iterator_traits<It>::value_type>;

template<class T, uint32_t Extent = DynamicExtent>
constexpr Span<const T, Extent> ToConstSpan(Span<T, Extent> s);

template<class T0, uint32_t E0,
         class T1, uint32_t E1>
requires std::is_same_v<std::remove_cvref_t<T0>, std::remove_cvref_t<T1>>
constexpr bool IsSubspan(Span<T0, E0> checkedSpan, Span<T1, E1> bigSpan);

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

template<class T, uint32_t E>
constexpr Span<T, E>::Span() requires(IsDynamic)
{
    this->ptr = nullptr;
    this->count = 0;
}

template<class T, uint32_t E>
constexpr Span<T, E>::Span(T* data, size_t size) noexcept
{
    assert(size <= UINT32_MAX);
    this->ptr = data;
    if constexpr(IsDynamic) this->count = uint32_t(size);
    else                    assert(size >= E);
}

template<class T, uint32_t E>
constexpr Span<T, E>::Span(NonConstType* data, size_t size) noexcept requires(IsConstPtr)
{
    assert(size <= UINT32_MAX);
    this->ptr = data;
    if constexpr(IsDynamic) this->count = uint32_t(size);
    else                    assert(size == E);
}

template<class T, uint32_t E>
template<SpanIt It>
constexpr Span<T, E>::Span(It begin, It end) noexcept
{
    this->ptr = std::to_address(begin);
    if constexpr(IsDynamic) this->count = uint32_t(end - begin);
    else                    assert(end > begin && uint32_t(end - begin) == E);
}

template<class T, uint32_t E>
template<ConstSpanIt It>
constexpr Span<T, E>::Span(It begin, It end) noexcept
{
    assert(end >= begin && (end - begin) <= std::ptrdiff_t(UINT32_MAX));

    this->ptr = std::to_address(begin);
    if constexpr(IsDynamic) this->count = uint32_t(end - begin);
    else                    assert(end > begin && uint32_t(end - begin) == E);
}

template<class T, uint32_t E>
template<std::size_t N>
constexpr Span<T, E>::Span(std::array<T, N>& arr) noexcept
{
    this->ptr = arr.data();
    if constexpr(IsDynamic) this->count = uint32_t(N);
    else                    static_assert(E <= N, "Cannot fit array to the static span");
}

template<class T, uint32_t E>
template<std::size_t N>
constexpr Span<T, E>::Span(const std::array<NonConstType, N>& arr) noexcept requires(IsConstPtr)
{
    this->ptr = arr.data();
    if constexpr(IsDynamic) this->count = uint32_t(N);
    else                    static_assert(E <= N, "Cannot fit array to the static span");
}

template<class T, uint32_t E>
template<std::size_t N>
constexpr Span<T, E>::Span(T(&arr)[N]) noexcept
{
    this->ptr = arr;
    if constexpr(IsDynamic) this->count = uint32_t(N);
    else                    static_assert(E <= N, "Cannot fit array to the static span");
}

template<class T, uint32_t E>
template<std::size_t N>
constexpr Span<T, E>::Span(NonConstType(&arr)[N]) noexcept requires(IsConstPtr)
{
    this->ptr = arr;
    if constexpr(IsDynamic) this->count = uint32_t(N);
    else                    static_assert(E <= N, "Cannot fit array to the static span");
}

template<class T, uint32_t E>
constexpr Span<T, E>::Span(std::vector<T>& v) noexcept
    : Span(v.data(), v.size())
{}

template<class T, uint32_t E>
constexpr Span<T, E>::Span(const std::vector<NonConstType>& v) noexcept requires(IsConstPtr)
    : Span(v.data(), v.size())
{}

template<class T, uint32_t E>
constexpr Span<T, E>::Span(const Span<NonConstType, E>& other) noexcept requires(IsConstPtr)
{
    this->ptr = other.ptr;
    if constexpr(IsDynamic)
        this->count = other.count;
}

template<class T, uint32_t E>
template<uint32_t N>
constexpr Span<T, E>::Span(const Span<T, N>& other) noexcept requires(E != N)
{
    constexpr bool OtherIsDynamic = Span<T, N>::IsDynamic;
    if constexpr(IsDynamic && !OtherIsDynamic)
    {
        this->ptr = other.ptr;
        this->count = N;
    }
    else if constexpr(!IsDynamic && OtherIsDynamic)
    {
        assert(E <= other.count);
    }
    else static_assert(E <= N, "Could not fit Span to other span!");
    //
    this->ptr = other.ptr;
}

template<class T, uint32_t E>
template<uint32_t N>
constexpr Span<T, E>::Span(const Span<NonConstType, N>& other) noexcept requires((E != N) && IsConstPtr)
{
    constexpr bool OtherIsDynamic = Span<T, N>::IsDynamic;
    if constexpr(IsDynamic && !OtherIsDynamic)
    {
        this->ptr = other.ptr;
        this->count = N;
    }
    else if constexpr(!IsDynamic && OtherIsDynamic)
    {
        assert(E <= other.count);
    }
    else static_assert(E <= N, "Could not fit Span to other span!");
    //
    this->ptr = other.ptr;
}

template<class T, uint32_t E>
constexpr T&
Span<T, E>::operator[](uint32_t index) const
{
    [[maybe_unused]]
    uint32_t C;
    if constexpr(IsDynamic) C = this->count;
    else                    C = E;
    assert(index < C);
    //
    //
    return this->ptr[index];
}

template<class T, uint32_t E>
constexpr T* Span<T, E>::data() const
{
    return this->ptr;
}

template<class T, uint32_t E>
constexpr T& Span<T, E>::front() const
{
    if constexpr(IsDynamic)
    {
        assert(this->count != 0);
        return *(this->ptr);
    }
    else return *(this->ptr);
}
template<class T, uint32_t E>
constexpr T& Span<T, E>::back() const
{
    if constexpr(IsDynamic)
    {
        assert(this->count != 0);
        return *(this->ptr + this->count - 1);
    }
    else return *(this->ptr + (E - 1));
}

template<class T, uint32_t E>
constexpr T*
Span<T, E>::begin() const
{
    return this->ptr;
}

template<class T, uint32_t E>
constexpr const T*
Span<T, E>::cbegin() const
{
    return this->ptr;
}

template<class T, uint32_t E>
constexpr T*
Span<T, E>::end() const
{
    if constexpr(IsDynamic) return this->ptr + this->count;
    else                    return this->ptr + E;
}

template<class T, uint32_t E>
constexpr const T*
Span<T, E>::cend() const
{
    if constexpr(IsDynamic) return this->ptr + this->count;
    else                    return this->ptr + E;
}

template<class T, uint32_t E>
constexpr uint32_t
Span<T, E>::size() const
{
    if constexpr(IsDynamic) return this->count;
    else                    return E;
}

template<class T, uint32_t E>
constexpr uint32_t
Span<T, E>::size_bytes() const
{
    if constexpr(IsDynamic) return this->count * sizeof(T);
    else                    return E * sizeof(T);
}

template<class T, uint32_t E>
constexpr bool
Span<T, E>::empty() const
{
    if constexpr(IsDynamic) return this->count == 0;
    else                    return false;
}

template<class T, uint32_t E>
constexpr Span<T>
Span<T, E>::subspan(size_t offset, size_t count) const
{
    assert(count <= UINT32_MAX);
    if(count == UINT32_MAX)
    {
        if constexpr(E == DynamicExtent) count = this->count - offset;
        else                             count = E - offset;
    }
    //
    [[maybe_unused]]
    uint32_t C;
    if constexpr(MRAY_IS_DEBUG)
    {
        if constexpr(E == DynamicExtent) C = this->count;
        else                             C = E;
    }
    assert(offset + count <= C);
    //
    return Span<T>(this->ptr + offset, count);
}

template<class T0, uint32_t E0,
         class T1, uint32_t E1>
requires std::is_same_v<std::remove_cvref_t<T0>, std::remove_cvref_t<T1>>
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

template<class T, uint32_t Extent>
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