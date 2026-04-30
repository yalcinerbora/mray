
#include "Core/Vector.h"

namespace mray::hip::atomic::detail
{
    template <class T> struct IntegralOf;

    template <class T>
    requires (sizeof(T) == 8 && alignof(T) >= alignof(uint64_t))
    struct IntegralOf<T> { using type = uint64_t; };

    template <class T>
    requires (sizeof(T) == 4 && alignof(T) >= alignof(uint32_t))
    struct IntegralOf<T> { using type = uint32_t; };

    template <class T>
    requires (sizeof(T) == 2 && alignof(T) >= alignof(uint16_t))
    struct IntegralOf<T> { using type = uint16_t; };
}

namespace mray::hip::atomic
{
    template<class T, class Func>
    MR_GF_DECL T EmulateAtomicOp(T&, Func&& f);

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
    MR_GF_DECL T AtomicCompSwap(T& t, T compVal, T storeVal);

    template<class T>
    MR_GF_DECL T AtomicLoad(T& t);

    template<class T>
    MR_GF_DECL void AtomicStore(T& t, T val);
}

// A dirty fix to host side to not whine about
// undefined "atomicXXX" functions.
#ifdef __HIP_DEVICE_COMPILE__

namespace mray::hip::atomic
{

template<class T, class Func>
MR_GF_DEF
T EmulateAtomicOp(T& address, Func&& F)
{
    // TODO:
    // __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__
    // __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__
    // Compile-time check these macros

    using I = typename detail::template IntegralOf<T>::type;
    // Classic CAS wrapper for the operation
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
    I* integralAddress = std::launder(reinterpret_cast<I*>(&address));
    I old = Bit::BitCast<I>(integralAddress);
    I assumed;
    do
    {
        assumed = old;
        T r = F(Bit::BitCast<T>(assumed));
        I rI = Bit::BitCast<I>(r);
        old = atomicCAS(integralAddress, assumed, rI);
    }
    while(assumed != old);
    return Bit::BitCast<T>(old);
}


template<>
MR_GF_DEF
double AtomicAdd(double& t, double v)
{
    return EmulateAtomicOp(t, [v](double r)
    {
        return r + v;
    });
}

template<>
MR_GF_DEF
Vector2 AtomicAdd(Vector2& t, Vector2 v)
{
    static_assert(std::is_same_v<Float, float>,
                  "\"Float\" must be 32-bit for this function! (TODO)");
    Vector2 r;
    // Emulate with 2 atomics
    r[0] = AtomicAdd(t[0], v[0]);
    r[1] = AtomicAdd(t[1], v[1]);
    return r;
}

template<>
MR_GF_DEF
Vector3 AtomicAdd(Vector3& t, Vector3 v)
{
    static_assert(std::is_same_v<Float, float>,
                  "\"Float\" must be 32-bit for this function! (TODO)");

    Vector3 r;
    // Emulate with 3 atomics
    r[0] = AtomicAdd(t[0], v[0]);
    r[1] = AtomicAdd(t[1], v[1]);
    r[2] = AtomicAdd(t[2], v[2]);
    return r;
}

template<>
MR_GF_DEF
Vector4 AtomicAdd(Vector4& t, Vector4 v)
{
    static_assert(std::is_same_v<Float, float>,
                  "\"Float\" must be 32-bit for this function! (TODO)");

    Vector4 r;
    // Emulate with 4 atomics
    r[0] = AtomicAdd(t[0], v[0]);
    r[1] = AtomicAdd(t[1], v[1]);
    r[2] = AtomicAdd(t[2], v[2]);
    r[3] = AtomicAdd(t[3], v[3]);
    return r;
}

template<class T>
MR_GF_DEF
T AtomicAdd(T& t, T v)
{
    // TODO: Check proper template instantiations
    // overload resolutions to remove this static assert
    static_assert(std::is_same_v<T, int32_t> ||
                  std::is_same_v<T, uint32_t> ||
                  std::is_same_v<T, uint64_t> ||
                  std::is_same_v<T, float>,
                  "T must be an /float32/int32/uint32/uint64 "
                  "type!");
    return atomicAdd(&t, v);
}

template<class T>
requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
MR_GF_DEF
T AtomicMax(T& t, T v)
{
    return atomicMax(&t, v);
}

template<class T>
requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
MR_GF_DEF
T AtomicMin(T& t, T v)
{
    return atomicMin(&t, v);
}

template<class T>
requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
MR_GF_DEF
T AtomicAnd(T& t, T v)
{
    return atomicAnd(&t, v);
}

template<class T>
requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
MR_GF_DEF
T AtomicOr(T& t, T v)
{
    return atomicOr(&t, v);
}

template<class T>
requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
MR_GF_DEF
T AtomicXor(T& t, T v)
{
    return atomicXor(&t, v);
}

template<class T>
MR_GF_DEF
T AtomicCompSwap(T& t, T compVal, T storeVal)
{
    using Int = typename detail::template IntegralOf<T>::type;
    auto* tp = reinterpret_cast<Int*>(&t);
    return Bit::BitCast<T>(atomicCAS(tp,
                           Bit::BitCast<Int>(compVal),
                           Bit::BitCast<Int>(storeVal)));
}

template<class T>
MR_GF_DECL T AtomicLoad(T& t)
{
    // TODO: Use proper load when it is available on windows
    return atomicAdd(&t, 0);
}

template<class T>
MR_GF_DECL void AtomicStore(T& t, T val)
{
    // TODO: Use proper store when it is available on windows
    atomicExch(&t, val);
}

}

#endif