
#include "Core/Vector.h"

namespace mray::hip::atomic::detail
{
    template <class T> struct IntegralOf;

    template <class T>
    requires (sizeof(T) == 8)
    struct IntegralOf<T> { using type = uint64_t; };

    template <class T>
    requires (sizeof(T) == 4)
    struct IntegralOf<T> { using type = uint32_t; };

    template <class T>
    requires (sizeof(T) == 2)
    struct IntegralOf<T> { using type = uint16_t; };

    template<class T, class Func>
    MRAY_GPU T EmulateAtomicOp(T*, T val, Func&& f);
}

namespace mray::hip::atomic
{
    template<class T>
    MRAY_GPU T AtomicAdd(T& t, T v);

    template<class T>
    requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
    MRAY_GPU T AtomicAnd(T& t, T v);

    template<class T>
    requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
    MRAY_GPU T AtomicOr(T& t, T v);

    template<class T>
    requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
    MRAY_GPU T AtomicXor(T& t, T v);
}

// A dirty fix to host side to not whine about
// undefined "atomicXXX" functions.
#ifdef __HIP_DEVICE_COMPILE__
namespace mray::hip::atomic::detail
{

template<class T, class Func>
MRAY_GPU MRAY_GPU_INLINE
T EmulateAtomicOp(T* address, T val, Func&& F)
{
    // TODO:
    // __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__
    // __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__
    // Compile-time check these macros

    using I = typename IntegralOf<T>::type;
    // Classic CAS wrapper for the operation
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
    I* integralAddress = std::launder(reinterpret_cast<I*>(address));
    I old = *integralAddress;
    I assumed;
    do
    {
        assumed = old;
        T r = F(std::bit_cast<T>(assumed), val);
        I rI = std::bit_cast<I>(r);
        old = atomicCAS(integralAddress, assumed, rI);
    }
    while(assumed != old);
    return std::bit_cast<T>(old);
}

}

namespace mray::hip::atomic
{

template<>
MRAY_GPU MRAY_GPU_INLINE
double AtomicAdd(double& t, double v)
{
    return detail::EmulateAtomicOp(&t, v, [](double l, double r)
    {
        return l + r;
    });
}

template<>
MRAY_GPU MRAY_GPU_INLINE
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
MRAY_GPU MRAY_GPU_INLINE
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
MRAY_GPU MRAY_GPU_INLINE
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
MRAY_GPU MRAY_GPU_INLINE
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
MRAY_GPU MRAY_GPU_INLINE
T AtomicAnd(T& t, T v)
{
    return atomicAnd(&t, v);
}

template<class T>
requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
MRAY_GPU MRAY_GPU_INLINE
T AtomicOr(T& t, T v)
{
    return atomicOr(&t, v);
}

template<class T>
requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
MRAY_GPU MRAY_GPU_INLINE
T AtomicXor(T& t, T v)
{
    return atomicXor(&t, v);
}

}

#endif