#pragma once

#include "GPUSystemCPU.hpp"

namespace mray::host::warp
{
    static constexpr unsigned int ALL_WARP_MASK = std::numeric_limits<unsigned int>::max();

    inline uint32_t ActiveLaneMask() { return uint32_t(1); }

    template<uint32_t LogicalWarpSize = WarpSize()>
    inline uint32_t WarpBallot(bool, unsigned int = ALL_WARP_MASK)
    {
        static_assert(LogicalWarpSize == UINT32_MAX,
                      "Warp operation does not makes sense in CPU mode! "
                      "Please guard the code via \"MRAY_DEVICE_CODE_PATH*\"");
    }

    template<uint32_t LogicalWarpSize = WarpSize()>
    inline bool WarpAny(bool, unsigned int = ALL_WARP_MASK)
    {
        static_assert(LogicalWarpSize == UINT32_MAX,
                      "Warp operation does not makes sense in CPU mode! "
                      "Please guard the code via \"MRAY_DEVICE_CODE_PATH*\"");
    }

    template<uint32_t LogicalWarpSize = WarpSize()>
    inline bool WarpAll(bool, unsigned int = ALL_WARP_MASK)
    {
        static_assert(LogicalWarpSize == UINT32_MAX,
                      "Warp operation does not makes sense in CPU mode! "
                      "Please guard the code via \"MRAY_DEVICE_CODE_PATH*\"");
    }

    template<uint32_t LogicalWarpSize = WarpSize(), class T>
    inline T WarpBroadcast(T, int, unsigned int = ALL_WARP_MASK)
    {
        static_assert(!std::is_same_v<T, T>,
                      "Warp operation does not makes sense in CPU mode! "
                      "Please guard the code via \"MRAY_DEVICE_CODE_PATH*\"");
    }

    template<uint32_t LogicalWarpSize = WarpSize(), class T>
    inline T WarpFetchForward(T, int, unsigned int = ALL_WARP_MASK)
    {
        static_assert(!std::is_same_v<T, T>,
                      "Warp operation does not makes sense in CPU mode! "
                      "Please guard the code via \"MRAY_DEVICE_CODE_PATH*\"");
    }

    template<uint32_t LogicalWarpSize = WarpSize(), class T>
    inline T WarpFetchBackward(T, int, unsigned int = ALL_WARP_MASK)
    {
        static_assert(!std::is_same_v<T, T>,
                      "Warp operation does not makes sense in CPU mode! "
                      "Please guard the code via \"MRAY_DEVICE_CODE_PATH*\"");
    }

    template<uint32_t LogicalWarpSize = WarpSize(),
             class T, class BinaryFunc>
    inline T WarpReduce(T, BinaryFunc&&, unsigned int = ALL_WARP_MASK)
    {
        static_assert(!std::is_same_v<T, T>,
                      "Warp operation does not makes sense in CPU mode! "
                      "Please guard the code via \"MRAY_DEVICE_CODE_PATH*\"");
    }

    template<class T>
    inline T WarpReduceAdd(T, unsigned int = ALL_WARP_MASK)
    {
        static_assert(!std::is_same_v<T, T>,
                      "Warp operation does not makes sense in CPU mode! "
                      "Please guard the code via \"MRAY_DEVICE_CODE_PATH*\"");
    }

    template<class T>
    inline T WarpReduceMin(T, unsigned int = ALL_WARP_MASK)
    {
        static_assert(!std::is_same_v<T, T>,
                      "Warp operation does not makes sense in CPU mode! "
                      "Please guard the code via \"MRAY_DEVICE_CODE_PATH*\"");
    }

    template<class T>
    inline T WarpReduceMax(T, unsigned int = ALL_WARP_MASK)
    {
        static_assert(!std::is_same_v<T, T>,
                      "Warp operation does not makes sense in CPU mode! "
                      "Please guard the code via \"MRAY_DEVICE_CODE_PATH*\"");
    }

    template<class T>
    inline T WarpReduceAnd(T, unsigned int = ALL_WARP_MASK)
    {
        static_assert(!std::is_same_v<T, T>,
                      "Warp operation does not makes sense in CPU mode! "
                      "Please guard the code via \"MRAY_DEVICE_CODE_PATH*\"");
    }

    template<class T>
    inline T WarpReduceOr(T, unsigned int = ALL_WARP_MASK)
    {
        static_assert(!std::is_same_v<T, T>,
                      "Warp operation does not makes sense in CPU mode! "
                      "Please guard the code via \"MRAY_DEVICE_CODE_PATH*\"");
    }

    template<class T>
    inline T WarpReduceXor(T, unsigned int = ALL_WARP_MASK)
    {
        static_assert(!std::is_same_v<T, T>,
                      "Warp operation does not makes sense in CPU mode! "
                      "Please guard the code via \"MRAY_DEVICE_CODE_PATH*\"");
    }
}