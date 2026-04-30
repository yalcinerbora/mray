#pragma once

// To aid the compilation of ROCm we implement a basic
// std::array-like data strucuture. Which will be slightly different
// since we can't do default construction when internal data type is
// not default constructible unlike std::array (it requires special treatment from
// the compiler).
//
// Also it is mostly 32-bit based, due to register pressure on GPUs.
#include "Definitions.h"

#include <array>

template<class T, uint32_t N>
class Array
{
    static_assert(N != 0, "Zero-sized array is disallowed!");

    // We want Array to be passed as NTTP, so the value is not private
    // https://en.cppreference.com/cpp/language/template_parameters
    public:
    T v[N];

    public:
    constexpr    Array()  = default;
    template<class... Args>
    MR_PF_DECL_V Array(Args...);
    MR_PF_DECL_V Array(const std::array<T, N>& right);

    MR_PF_DECL T&       operator[](uint32_t);
    MR_PF_DECL const T& operator[](uint32_t) const;

    MR_PF_DECL T*       data();
    MR_PF_DECL const T* data() const;

    MR_PF_DECL T&       front();
    MR_PF_DECL const T& front() const;

    MR_PF_DECL T&       back();
    MR_PF_DECL const T& back() const;

    MR_PF_DECL T*       begin();
    MR_PF_DECL const T* begin() const;

    MR_PF_DECL T*       end();
    MR_PF_DECL const T* end() const;

    MR_PF_DECL const T* cbegin() const;
    MR_PF_DECL const T* cend() const;

    MR_PF_DECL bool     empty() const;
    MR_PF_DECL uint32_t size() const;
    MR_PF_DECL uint32_t max_size() const;

    MR_PF_DECL_V void   fill(const T&);
};

// https://cppreference.com/cpp/container/array/deduction_guides
template<class T, class... U>
Array(T, U...) -> Array<T, 1 + sizeof...(U)>;

// For structured bindings
template <size_t I, class T, uint32_t N>
constexpr T& get(Array<T, N>& t) noexcept
{
    static_assert(I < N, "I exceeds the Array size \"N\"");
    return t[I];
}

template <size_t I, class T, uint32_t N>
constexpr const T& get(const Array<T, N>& t) noexcept
{
    static_assert(I < N, "I exceeds the Array size \"N\"");
    return t[I];
}

template <size_t I, class T, uint32_t N>
constexpr T&& get(Array<T, N>&& t) noexcept
{
    static_assert(I < N, "I exceeds the Array size \"N\"");
    return std::move(t[I]);
}

namespace std
{
    template <class T, uint32_t N>
    struct tuple_size<Array<T, N>> : std::integral_constant<size_t, N>
    {};

    template <size_t I, class T, uint32_t N>
    struct tuple_element<I, Array<T, N>>
    {
        using type = T;
    };
}

template<class T, uint32_t N>
template<class... Args>
MR_PF_DEF_V
Array<T, N>::Array(Args... args)
    : v{std::forward<Args>(args)...}
{}

template<class T, uint32_t N>
MR_PF_DEF_V
Array<T, N>::Array(const std::array<T, N>& right)
{
    MRAY_UNROLL_LOOP_N(N)
    for(uint32_t i = 0; i < N; i++)
    {
        v[i] = right[i];
    }
}

template<class T, uint32_t N>
MR_PF_DEF
T& Array<T, N>::operator[](uint32_t i)
{
    // All this class is created due to this because of AMD HIP
    // does not compile the MSVC STL due to CRTDebug...
    assert(i < N && "Out of bounds access on Array");
    return v[i];
}

template<class T, uint32_t N>
MR_PF_DEF
const T& Array<T, N>::operator[](uint32_t i) const
{
    assert(i < N && "Out of bounds access on Array");
    return v[i];
}

template<class T, uint32_t N>
MR_PF_DEF
T* Array<T, N>::data()
{
    return v;
}

template<class T, uint32_t N>
MR_PF_DEF
const T* Array<T, N>::data() const
{
    return v;
}

template<class T, uint32_t N>
MR_PF_DEF
T& Array<T, N>::front()
{
    return *v;
}

template<class T, uint32_t N>
MR_PF_DEF
const T& Array<T, N>::front() const
{
    return *v;
}

template<class T, uint32_t N>
MR_PF_DEF
T& Array<T, N>::back()
{
    return *(v + (N - 1));
}

template<class T, uint32_t N>
MR_PF_DEF
const T& Array<T, N>::back() const
{
    return *(v + (N - 1));
}

template<class T, uint32_t N>
MR_PF_DEF
T* Array<T, N>::begin()
{
    return v;
}

template<class T, uint32_t N>
MR_PF_DEF
const T* Array<T, N>::begin() const
{
    return v;
}

template<class T, uint32_t N>
MR_PF_DEF
T* Array<T, N>::end()
{
    return v + N;
}

template<class T, uint32_t N>
MR_PF_DEF
const T* Array<T, N>::end() const
{
    return v + N;
}

template<class T, uint32_t N>
MR_PF_DEF
const T* Array<T, N>::cbegin() const
{
    return v;
}

template<class T, uint32_t N>
MR_PF_DEF
const T* Array<T, N>::cend() const
{
    return v + N;
}

template<class T, uint32_t N>
MR_PF_DEF
bool Array<T, N>::empty() const
{
    return false;
}

template<class T, uint32_t N>
MR_PF_DEF
uint32_t Array<T, N>::size() const
{
    return N;
}

template<class T, uint32_t N>
MR_PF_DEF
uint32_t Array<T, N>::max_size() const
{
    return N;
}

template<class T, uint32_t N>
MR_PF_DEF_V
void Array<T, N>::fill(const T& val)
{
    for(uint32_t i = 0; i < N; i++)
        v[i] = val;
}
