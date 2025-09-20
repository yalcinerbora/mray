
#include "Core/Vector.h"

namespace mray::cuda::atomic::detail
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
    MR_GF_DECL T EmulateAtomicOp(T*, T val, Func&& f);
}

namespace mray::cuda::atomic
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

// A dirty fix to host side to not whine about
// undefined "atomicXXX" functions.

namespace mray::cuda::atomic::detail
{

template<class T, class Func>
MR_GF_DEF
T EmulateAtomicOp(T* address, T val, Func&& F)
{
    #ifdef __CUDA_ARCH__
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
    #else
        return F(*address + val);
    #endif
}

}

namespace mray::cuda::atomic
{

template<>
MR_GF_DEF
double AtomicAdd(double& t, double v)
{
    #ifdef __CUDA_ARCH__
        #if __CUDA_ARCH__ >= 600
            return atomicAdd(&t, v);
        #else
            return detail::EmulateAtomicOp(&t, v, [](double l, double r)
            {
                return l + r;
            });
    #endif
    #else
        return t + v;
    #endif
}

template<class T>
MR_GF_DEF
Vector<2, T> AtomicAdd(Vector<2, T>& t, Vector<2, T> v)
{
    #ifdef __CUDA_ARCH__
        Vector<2, T> r;
        if constexpr(std::is_same_v<T, float>)
        {
            #if __CUDA_ARCH__ >= 900
                auto* tp = reinterpret_cast<float2*>(&t);
                auto vv = float2{v[0], v[1]};
                float2 out = atomicAdd(tp, vv);
                r[0] = out.x;
                r[1] = out.y;
            #else
                // Emulate with 2 atomics
                r[0] = AtomicAdd(t[0], v[0]);
                r[1] = AtomicAdd(t[1], v[1]);
            #endif
        }
        else
        {
            // Emulate with 2 atomics
            r[0] = AtomicAdd(t[0], v[0]);
            r[1] = AtomicAdd(t[1], v[1]);
        }
        return r;
    #else
        return t + v;
    #endif
}

template<class T>
MR_GF_DEF
Vector<3, T> AtomicAdd(Vector<3, T>& t, Vector<3, T> v)
{
    Vector<3, T> r;
    // Emulate with 3 atomics
    r[0] = AtomicAdd(t[0], v[0]);
    r[1] = AtomicAdd(t[1], v[1]);
    r[2] = AtomicAdd(t[2], v[2]);
    return r;
}

template<class T>
MR_GF_DEF
Vector<4, T> AtomicAdd(Vector<4, T>& t, Vector<4, T> v)
{
    #ifdef __CUDA_ARCH__
        Vector<4, T> r;
        if constexpr(std::is_same_v<T, float>)
        {
            #if __CUDA_ARCH__ >= 900
                auto* tp = reinterpret_cast<float4*>(&t);
                auto vv = float4{v[0], v[1], v[2], v[3]};
                float4 out = atomicAdd(tp, vv);
                r[0] = out.x;
                r[1] = out.y;
                r[2] = out.z;
                r[3] = out.w;
            #else
                // Emulate with 4 atomics
                r[0] = AtomicAdd(t[0], v[0]);
                r[1] = AtomicAdd(t[1], v[1]);
                r[2] = AtomicAdd(t[2], v[2]);
                r[3] = AtomicAdd(t[3], v[3]);
            #endif
        }
        else
        {
            r[0] = AtomicAdd(t[0], v[0]);
            r[1] = AtomicAdd(t[1], v[1]);
            r[2] = AtomicAdd(t[2], v[2]);
            r[3] = AtomicAdd(t[3], v[3]);
        }
        return r;
    #else
        return t + v;
    #endif
}

template<class T>
MR_GF_DEF
T AtomicAdd(T& t, T v)
{
    #ifdef __CUDA_ARCH__
        // TODO: Check proper template instantiations
        // overload resolutions to remove this static assert
        static_assert(std::is_same_v<T, int32_t> ||
                      std::is_same_v<T, uint32_t> ||
                      std::is_same_v<T, uint64_t> ||
                      std::is_same_v<T, float>,
                      "T must be an /float32/int32/uint32/uint64 "
                      "type!");
        return atomicAdd(&t, v);
    #else
        return t + v;
    #endif
}

template<class T>
requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
MR_GF_DEF
T AtomicAnd(T& t, T v)
{
    #ifdef __CUDA_ARCH__
        return atomicAnd(&t, v);
    #else
        return t + v;
    #endif
}

template<class T>
requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
MR_GF_DEF
T AtomicOr(T& t, T v)
{
    #ifdef __CUDA_ARCH__
        return atomicOr(&t, v);
    #else
        return t + v;
    #endif
}

template<class T>
requires(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)
MR_GF_DEF
T AtomicXor(T& t, T v)
{
    #ifdef __CUDA_ARCH__
        return atomicXor(&t, v);
    #else
        return t + v;
    #endif
}

}