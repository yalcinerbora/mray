
#include "GPUSystemCUDA.h"

namespace mray::cuda::warp::detail
{
    template<class T> struct TransferringTypeT;

    template<class T>
    requires ((sizeof(T) > 4) && (sizeof(T) <= 8))
    struct TransferringTypeT<T> { using type = uint64_t; };

    template<class T>
    requires (sizeof(T) <= 4)
    struct TransferringTypeT<T> { using type = uint32_t; };

    template<class T>
    using TransferringType = typename TransferringTypeT<T>::type;

    template<class T>
    constexpr TransferringType<T> CopyToIntegral(T t)
    {
        using ResultT = TransferringType<T>;
        if constexpr(sizeof(T) == sizeof(ResultT))
            return Bit::BitCast<ResultT>(t);
        else
        {
            ResultT r;
            std::memcpy(&r, &t, sizeof(T));
            return r;
        }
    }

    template<class T>
    constexpr T CopyFromIntegral(TransferringType<T> i)
    {
        if constexpr(sizeof(T) == sizeof(TransferringType<T>))
            return Bit::BitCast<T>(i);
        else
        {
            T r;
            std::memcpy(&r, &i, sizeof(T));
            return r;
        }
    }
}

namespace mray::cuda::warp
{
    template<class T>
    concept WarpTransferrableC = requires(const T& t)
    {
        requires ImplicitLifetimeC<T>;
        typename detail::TransferringType<T>;
    };

    static constexpr unsigned int ALL_WARP_MASK = std::numeric_limits<unsigned int>::max();

    template<uint32_t LogicalWarpSize = WarpSize(), WarpTransferrableC T>
    MR_GF_DECL T WarpBroadcast(T varName, int laneId,
                               unsigned int mask = ALL_WARP_MASK);

    template<uint32_t LogicalWarpSize = WarpSize(), WarpTransferrableC T>
    MR_GF_DECL T WarpFetchForward(T varName, int offset,
                                  unsigned int mask = ALL_WARP_MASK);

    template<uint32_t LogicalWarpSize = WarpSize(), WarpTransferrableC T>
    MR_GF_DECL T WarpFetchBackward(T varName, int offset,
                                  unsigned int mask = ALL_WARP_MASK);

    template<uint32_t LogicalWarpSize = WarpSize(),
             WarpTransferrableC T, class BinaryFunc>
    MR_GF_DECL T WarpReduce(T varName, BinaryFunc&&,
                            unsigned int mask = ALL_WARP_MASK);

    template<uint32_t LogicalWarpSize = WarpSize(), WarpTransferrableC T>
    MR_GF_DECL T WarpReduceAdd(T varName, unsigned int mask = ALL_WARP_MASK);

    template<uint32_t LogicalWarpSize = WarpSize(), WarpTransferrableC T>
    MR_GF_DECL T WarpReduceMin(T varName, unsigned int mask = ALL_WARP_MASK);

    template<uint32_t LogicalWarpSize = WarpSize(), WarpTransferrableC T>
    MR_GF_DECL T WarpReduceMax(T varName, unsigned int mask = ALL_WARP_MASK);

    template<uint32_t LogicalWarpSize = WarpSize(), WarpTransferrableC T>
    MR_GF_DECL T WarpReduceAnd(T varName, unsigned int mask = ALL_WARP_MASK);

    template<uint32_t LogicalWarpSize = WarpSize(), WarpTransferrableC T>
    MR_GF_DECL T WarpReduceOr(T varName, unsigned int mask = ALL_WARP_MASK);

    template<uint32_t LogicalWarpSize = WarpSize(), WarpTransferrableC T>
    MR_GF_DECL T WarpReduceXor(T varName, unsigned int mask = ALL_WARP_MASK);
}

namespace mray::cuda::warp
{

template<uint32_t LogicalWarpSize , WarpTransferrableC T>
MR_GF_DEF
T WarpBroadcast(T varName, int laneId, unsigned int mask)
{
    using namespace detail;
    return CopyFromIntegral<T>(__shfl_sync(mask, laneId, CopyToIntegral(varName), LogicalWarpSize));
}

template<uint32_t LogicalWarpSize, WarpTransferrableC T>
MR_GF_DEF
T WarpFetchForward(T varName, int offset, unsigned int mask)
{
    using namespace detail;
    return CopyFromIntegral<T>(__shfl_down_sync(mask, offset, CopyToIntegral(varName),
                                                LogicalWarpSize));
}

template<uint32_t LogicalWarpSize, WarpTransferrableC T>
MR_GF_DEF
T WarpFetchBackward(T varName, int offset, unsigned int mask)
{
    using namespace detail;
    return CopyFromIntegral<T>(__shfl_up_sync(mask, offset, CopyToIntegral(varName),
                                              LogicalWarpSize));
}

template<uint32_t LogicalWarpSize, WarpTransferrableC T, class BinaryFunc>
MR_GF_DEF
T WarpReduce(T varName, unsigned int mask, BinaryFunc&& ReduceOp)
{
    T result = varName;

    MRAY_UNROLL_LOOP
    for(uint32_t j = (LogicalWarpSize >> 1); j != 0; j >>= 1)
        result = ReduceOp(result, WarpFetchForward<LogicalWarpSize>(result, j, mask));

    return result;
}

template<uint32_t LogicalWarpSize, WarpTransferrableC T>
MR_GF_DEF
T WarpReduceAdd(T varName, unsigned int mask)
{
    #ifdef __CUDA_ARCH__
        #if __CUDA_ARCH__ >= 800
            using namespace detail;
            return CopyFromIntegral<T>(__reduce_add_sync(mask, CopyToIntegral<T>(varName)));
        #else
            return WarpReduce(varName, mask, []MRAY_GPU(const T & a, const T & b)
            {
                return a + b
            });
        #endif
    #else
        return (mask == 0) ? varName : (varName + 1);
    #endif
}

template<uint32_t LogicalWarpSize, WarpTransferrableC T>
MR_GF_DEF
T WarpReduceMin(T varName, unsigned int mask)
{
    #ifdef __CUDA_ARCH__
        #if __CUDA_ARCH__ >= 800
            using namespace detail;
            return CopyFromIntegral<T>(__reduce_min_sync(mask, CopyToIntegral<T>(varName)));
        #else
            return WarpReduce(varName, mask, []MRAY_GPU(const T & a, const T & b)
            {
                return Math::Min(a, b);
            });
        #endif
    #else
        return (mask == 0) ? varName : Math::Min(varName, varName);
    #endif
}

template<uint32_t LogicalWarpSize, WarpTransferrableC T>
MR_GF_DEF
T WarpReduceMax(T varName, unsigned int mask)
{
    #ifdef __CUDA_ARCH__
        #if __CUDA_ARCH__ >= 800
            using namespace detail;
            return CopyFromIntegral<T>(__reduce_max_sync(mask, CopyToIntegral<T>(varName)));
        #else
            return WarpReduce(varName, mask, []MRAY_GPU(const T & a, const T & b)
            {
                return Math::Max(a, b);
            });
        #endif
    #else
        return (mask == 0) ? varName : Math::Max(varName, varName);
    #endif
}

template<uint32_t LogicalWarpSize, WarpTransferrableC T>
MR_GF_DECL T WarpReduceAnd(T varName, unsigned int mask)
{
    #ifdef __CUDA_ARCH__
        #if __CUDA_ARCH__ >= 800
            using namespace detail;
            return CopyFromIntegral<T>(__reduce_and_sync(mask, CopyToIntegral<T>(varName)));
        #else
            return WarpReduce(varName, mask, []MRAY_GPU(const T& a, const T& b)
            {
                return a & b;
            });
        #endif
    #else
        return (mask == 0) ? varName : (varName & varName);
    #endif
}

template<uint32_t LogicalWarpSize, WarpTransferrableC T>
MR_GF_DECL T WarpReduceOr(T varName, unsigned int mask)
{
    #ifdef __CUDA_ARCH__
        #if __CUDA_ARCH__ >= 800
            using namespace detail;
            return CopyFromIntegral<T>(__reduce_or_sync(mask, CopyToIntegral<T>(varName)));
        #else
            return WarpReduce(varName, mask, []MRAY_GPU(const T& a, const T& b)
            {
                return a | b;
            });
        #endif
    #else
        return (mask == 0) ? varName : (varName | varName);
    #endif
}

template<uint32_t LogicalWarpSize, ImplicitLifetimeC T>
MR_GF_DECL T WarpReduceXor(T varName, unsigned int mask)
{
    #ifdef __CUDA_ARCH__
        #if __CUDA_ARCH__ >= 800
            using namespace detail;
            return CopyFromIntegral<T>(__reduce_xor_sync(mask, CopyToIntegral<T>(varName)));
        #else
            return WarpReduce(varName, mask, []MRAY_GPU(const T& a, const T& b)
            {
                return a ^ b;
            });
        #endif
    #else
        return (mask == 0) ? varName : (varName ^ varName);
    #endif
}

}