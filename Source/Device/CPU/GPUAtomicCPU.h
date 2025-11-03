#pragma once
// IWYU pragma: private, include "GPUAtomic.h"

#include "Core/Vector.h"
#include "../GPUTypes.h"

#include <atomic>

namespace mray::host::atomic
{
    template<class T, class Func>
    MR_GF_DECL T EmulateAtomicOp(T&, Func&&);

    template<class T>
    MR_GF_DECL T AtomicAdd(T& t, T v);

    template<class T>
    requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
    MR_GF_DECL T AtomicMax(T& t, T v);

    template<class T>
    requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
    MR_GF_DECL T AtomicMin(T& t, T v);

    template<class T>
    requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
    MR_GF_DECL T AtomicAnd(T& t, T v);

    template<class T>
    requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
    MR_GF_DECL T AtomicOr(T& t, T v);

    template<class T>
    requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
    MR_GF_DECL T AtomicXor(T& t, T v);

    template<class T>
    MR_GF_DECL T AtomicCompSwap(T& t, T compVal, T val);
}

namespace mray::host::atomic
{

template<class T, class Func>
MR_GF_DECL
T EmulateAtomicOp(T& address, Func&& F)
{
    // Trigger instantiation of IntegralOf
    // to check if this is "lock-free" in comp time.
    // is_lock_free is runtime thing on c++ also CUDA etc
    // only have 16, 32, 64 bit (later ones have 128-bit)
    // atomics. So this will make it is comparable.
    static_assert(!std::is_same_v<typename IntegralOf<T>::type, void>);

    std::atomic_ref<T> ref(address);
    T expected = ref.load();
    T result;
    do
    {
        result = F(expected);
    }
    while(!ref.compare_exchange_strong(expected, result));
    return expected;
}

template<unsigned int D, class T>
MR_GF_DEF
Vector<D, T> AtomicAdd(Vector<D, T>& t, Vector<D, T> v)
{
    Vector<D, T> result;
    MRAY_UNROLL_LOOP
    for(unsigned int i = 0; i < D; i++)
        result[i] = std::atomic_ref<T>(t[i]).fetch_add(v[i]);

    return result;
}

template<class T>
MR_GF_DEF
T AtomicAdd(T& t, T v)
{
    return std::atomic_ref<T>(t).fetch_add(v);
}

template<class T>
requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
MR_GF_DEF
T AtomicMax(T& t, T v)
{
    // No fetch_max until c++26 :(
    return EmulateAtomicOp(t, [v](T in)
    {
        return Math::Max(v, in);
    });
}

template<class T>
requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
MR_GF_DEF T AtomicMin(T& t, T v)
{
    // No fetch_min until c++26 :(
    return EmulateAtomicOp(t, [v](T in)
    {
        return Math::Min(v, in);
    });
}

template<class T>
requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
MR_GF_DEF
T AtomicAnd(T& t, T v)
{
    return std::atomic_ref<T>(t).fetch_and(v);
}

template<class T>
requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
MR_GF_DEF
T AtomicOr(T& t, T v)
{
    return std::atomic_ref<T>(t).fetch_or(v);
}

template<class T>
requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
MR_GF_DEF
T AtomicXor(T& t, T v)
{
    return std::atomic_ref<T>(t).fetch_xor(v);
}

template<class T>
MR_GF_DEF
T AtomicCompSwap(T& t, T compVal, T val)
{
    std::atomic_ref<T> ref(t);
    ref.compare_exchange_strong(compVal, val);
    return compVal;
}

}