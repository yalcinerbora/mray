#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCUDA.h"

#include <cub/cub.cuh>
#include <cuda/functional>

// Direct wrappers over CUB at the moment
// Probably be refactored later
// Add as you need
namespace mray::cuda::algorithms
{
    template <class T>
    MRAY_HOST
    size_t ReduceTMSize(size_t elementCount);

    template <class T, class BinaryOp>
    MRAY_HOST
    void Reduce(Span<T, 1> dReducedValue,
                Span<Byte> dTempMemory,
                Span<const T> dValues,
                const T& initialValue,
                const GPUQueueCUDA& queue,
                BinaryOp&&);

    template <bool IsAscending, class K, class V>
    MRAY_HOST
    size_t RadixSortTMSize(size_t elementCount);

    template <bool IsAscending, class K, class V>
    MRAY_HOST
    uint32_t RadixSort(Span<Span<K>, 2> dKeyDoubleBuffer,
                       Span<Span<V>, 2> dValueDoubleBuffer,
                       Span<Byte> dTempMemory,
                       const GPUQueueCUDA& queue,
                       const Vector2ui& bitRange = Vector2ui(0, sizeof(K) * CHAR_BIT));
}

namespace mray::cuda::algorithms
{
    template <class T>
    MRAY_HOST inline
    size_t ReduceTMSize(size_t elementCount)
    {
        using namespace cub;

        T* dIn = nullptr;
        T* dOut = nullptr;
        void* dTM = nullptr;
        size_t result;
        CUDA_CHECK(DeviceReduce::Reduce(dTM, result, dIn, dOut,
                                        static_cast<int>(elementCount),
                                        [] MRAY_HYBRID(T, T)->T{ return T{}; }, T{}));
        return result;
    }

    template <class T, class BinaryOp>
    MRAY_HOST inline
    void Reduce(Span<T, 1> dReducedValue,
                Span<Byte> dTempMemory,
                Span<const T> dValues,
                const T& initialValue,
                const GPUQueueCUDA& queue,
                BinaryOp&& op)
    {
        using namespace cub;
        //reinterpret_cast<void*>(dTempMemory.data()),
        size_t size;
        CUDA_CHECK(DeviceReduce::Reduce(dTempMemory.data(), size,
                                        dValues.data(), dReducedValue.data(),
                                        static_cast<int>(dValues.size()),
                                        std::forward<BinaryOp>(op), initialValue,
                                        ToHandleCUDA(queue)));
    }

    template <bool IsAscending, class K, class V>
    MRAY_HOST inline
    size_t RadixSortTMSize(size_t elementCount)
    {
        using namespace cub;

        cub::DoubleBuffer<K> keys(nullptr, nullptr);
        cub::DoubleBuffer<V> values(nullptr, nullptr);
        void* dTM = nullptr;
        size_t result;
        if constexpr(IsAscending)
            CUDA_CHECK(DeviceRadixSort::SortPairs(dTM, result, keys,
                                                  values, elementCount));
        else
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(dTM, result, keys,
                                                            values, elementCount));
        return result;
    }

    template <bool IsAscending, class K, class V>
    MRAY_HOST inline
    uint32_t RadixSort(Span<Span<K>, 2> dKeyDoubleBuffer,
                       Span<Span<V>, 2> dValueDoubleBuffer,
                       Span<Byte> dTempMemory,
                       const GPUQueueCUDA& queue,
                       const Vector2ui& bitRange)
    {
        using namespace cub;

        assert(dKeyDoubleBuffer[0].size() == dKeyDoubleBuffer[1].size());
        assert(dValueDoubleBuffer[0].size() == dValueDoubleBuffer[1].size());
        assert(dKeyDoubleBuffer[0].size() == dValueDoubleBuffer[0].size());

        cub::DoubleBuffer<K> keys(dKeyDoubleBuffer[0].data(),
                                  dKeyDoubleBuffer[1].data());
        cub::DoubleBuffer<V> values(dValueDoubleBuffer[0].data(),
                                    dValueDoubleBuffer[1].data());

        size_t size;
        if constexpr(IsAscending)
            CUDA_CHECK(DeviceRadixSort::SortPairs(dTempMemory.data(), size, keys, values,
                                                  dKeyDoubleBuffer[0].size(),
                                                  bitRange[0],
                                                  bitRange[1],
                                                  ToHandleCUDA(queue)));
        else
            CUDA_CHECK(DeviceRadixSort::SortPairsDescending(dTempMemory.data(), size, keys, values,
                                                            dKeyDoubleBuffer[0].size(),
                                                            bitRange[0],
                                                            bitRange[1],
                                                            ToHandleCUDA(queue)));

        return (keys.Current() == dKeyDoubleBuffer[0].data()) ? 0 : 1;
    }
}