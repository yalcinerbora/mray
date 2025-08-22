#pragma once
// IWYU pragma: private, include "GPUAtomic.h"

#include "Core/Vector.h"
#include <atomic>

namespace mray::host::atomic
{
    template<class T>
    MR_GF_DECL T AtomicAdd(T& t, T v);

    template<class T>
    requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
    MR_GF_DECL T AtomicAnd(T& t, T v);

    template<class T>
    requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
    MR_GF_DECL T AtomicOr(T& t, T v);

    template<class T>
    requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
    MR_GF_DECL T AtomicXor(T& t, T v);
}

namespace mray::host::atomic
{

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

}