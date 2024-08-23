#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCUDA.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

namespace mray::cuda::algorithms
{

template <bool IsAscending, class K, class V>
MRAY_HOST inline
size_t SegmentedRadixSortTMSize(size_t totalElementCount,
                                size_t totalSegments)
{
    using namespace cub;

    cub::DoubleBuffer<K> keys(nullptr, nullptr);
    cub::DoubleBuffer<V> values(nullptr, nullptr);
    void* dTM = nullptr;
    uint32_t* dStartOffsets = nullptr;
    uint32_t* dEndOffsets = nullptr;
    size_t result;
    if constexpr(IsAscending)
        CUDA_CHECK(DeviceSegmentedRadixSort::SortPairs(dTM, result,
                                                       keys, values,
                                                       totalElementCount, totalSegments,
                                                       dStartOffsets, dEndOffsets));
    else
        CUDA_CHECK(DeviceRadixSort::SortPairsDescending(dTM, result,
                                                        keys, values,
                                                        totalElementCount, totalSegments,                                                        dStartOffsets, dEndOffsets));
    return result;
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
    using namespace std::literals;
    static const auto annotation = queue.CreateAnnotation("KCRadixSort"sv);
    GPUAnnotationCUDA::Scope _ = annotation.AnnotateScope();

    assert(dKeyDoubleBuffer[0].size() == dKeyDoubleBuffer[1].size());
    assert(dValueDoubleBuffer[0].size() == dValueDoubleBuffer[1].size());
    assert(dKeyDoubleBuffer[0].size() == dValueDoubleBuffer[0].size());

    DoubleBuffer<K> keys(dKeyDoubleBuffer[0].data(),
                            dKeyDoubleBuffer[1].data());
    DoubleBuffer<V> values(dValueDoubleBuffer[0].data(),
                            dValueDoubleBuffer[1].data());

    size_t size = dTempMemory.size();
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


template <bool IsAscending, class K, class V>
MRAY_HOST inline
size_t SegmentedRadixSort(Span<Span<K>, 2> dKeyDoubleBuffer,
                          Span<Span<V>, 2> dValueDoubleBuffer,
                          Span<Byte> dTempMemory,
                          Span<const uint32_t> dSegmentRanges,
                          const GPUQueueCUDA& queue,
                          const Vector2ui& bitRange)
{
}

}