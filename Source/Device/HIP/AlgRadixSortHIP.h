#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemHIP.h"


// #ifdef MRAY_WINDOWS
//     #pragma warning( push )
//     #pragma warning( disable : 4706)
// #endif

#include <rocprim/device/device_radix_sort.hpp>
#include <rocprim/device/device_segmented_radix_sort.hpp>

#ifdef MRAY_WINDOWS
    #pragma warning( pop )
#endif

namespace mray::hip::algorithms
{

template <bool IsAscending, class K, class V>
MRAY_HOST inline
size_t SegmentedRadixSortTMSize(size_t totalElementCount,
                                size_t totalSegments)
{
    using namespace rocprim;

    rocprim::double_buffer<K> keys(nullptr, nullptr);
    rocprim::double_buffer<V> values(nullptr, nullptr);
    void* dTM = nullptr;
    uint32_t* dStartOffsets = nullptr;
    uint32_t* dEndOffsets = nullptr;
    size_t result;
    if constexpr(IsAscending)
        HIP_CHECK(segmented_radix_sort_pairs(dTM, result,
                                             keys, values,
                                             static_cast<int>(totalElementCount),
                                             static_cast<int>(totalSegments),
                                             dStartOffsets, dEndOffsets));
    else
        HIP_CHECK(segmented_radix_sort_pairs_desc(dTM, result,
                                                  keys, values,
                                                  totalElementCount, totalSegments,
                                                  dStartOffsets, dEndOffsets));
    return result;
}

template <bool IsAscending, class K, class V>
MRAY_HOST inline
size_t RadixSortTMSize(size_t elementCount)
{
    using namespace rocprim;

    rocprim::double_buffer<K> keys(nullptr, nullptr);
    rocprim::double_buffer<V> values(nullptr, nullptr);
    void* dTM = nullptr;
    size_t result;
    if constexpr(IsAscending)
        HIP_CHECK(radix_sort_pairs(dTM, result, keys,
                                   values, elementCount));
    else
        HIP_CHECK(radix_sort_pairs_desc(dTM, result, keys,
                                        values, elementCount));
    return result;
}

template <bool IsAscending, class K, class V>
MRAY_HOST inline
uint32_t RadixSort(Span<Span<K>, 2> dKeyDoubleBuffer,
                   Span<Span<V>, 2> dValueDoubleBuffer,
                   Span<Byte> dTempMemory,
                   const GPUQueueHIP& queue,
                   const Vector2ui& bitRange)
{
    using namespace rocprim;
    using namespace std::literals;
    static const auto annotation = queue.CreateAnnotation("KCRadixSort"sv);
    const auto _ = annotation.AnnotateScope();

    assert(dKeyDoubleBuffer[0].size() == dKeyDoubleBuffer[1].size());
    assert(dValueDoubleBuffer[0].size() == dValueDoubleBuffer[1].size());
    assert(dKeyDoubleBuffer[0].size() == dValueDoubleBuffer[0].size());

    double_buffer<K> keys(dKeyDoubleBuffer[0].data(),
                          dKeyDoubleBuffer[1].data());
    double_buffer<V> values(dValueDoubleBuffer[0].data(),
                            dValueDoubleBuffer[1].data());

    size_t size = dTempMemory.size();
    if constexpr(IsAscending)
        HIP_CHECK(radix_sort_pairs(dTempMemory.data(), size, keys, values,
                                   dKeyDoubleBuffer[0].size(),
                                   bitRange[0],
                                   bitRange[1],
                                   ToHandleHIP(queue)));
    else
        HIP_CHECK(radix_sort_pairs_desc(dTempMemory.data(), size, keys, values,
                                        dKeyDoubleBuffer[0].size(),
                                        bitRange[0],
                                        bitRange[1],
                                        ToHandleHIP(queue)));

    uint32_t result = (keys.current() == dKeyDoubleBuffer[0].data()) ? 0u : 1u;
    assert(((values.current() == dValueDoubleBuffer[0].data()) ? 0u : 1u) == result);
    return result;
}

template <bool IsAscending, class K, class V>
MRAY_HOST inline
uint32_t SegmentedRadixSort(Span<Span<K>, 2> dKeyDoubleBuffer,
                            Span<Span<V>, 2> dValueDoubleBuffer,
                            Span<Byte> dTempMemory,
                            Span<const uint32_t> dSegmentRanges,
                            const GPUQueueHIP& queue,
                            const Vector2ui& bitRange)
{
    using namespace rocprim;

    rocprim::double_buffer<K> keys(dKeyDoubleBuffer[0].data(),
                                   dKeyDoubleBuffer[1].data());
    rocprim::double_buffer<V> values(dValueDoubleBuffer[0].data(),
                                     dValueDoubleBuffer[1].data());
    size_t tmSize = dTempMemory.size();
    int totalElemCount = static_cast<int>(dKeyDoubleBuffer[0].size());
    int totalSegments = static_cast<int>(dSegmentRanges.size() - 1);

    if constexpr(IsAscending)
        HIP_CHECK(segmented_radix_sort_pairs(dTempMemory.data(), tmSize,
                                             keys, values,
                                             totalElemCount, totalSegments,
                                             dSegmentRanges.data(),
                                             dSegmentRanges.data() + 1,
                                             bitRange[0], bitRange[1],
                                             ToHandleHIP(queue)));
    else
        HIP_CHECK(segmented_radix_sort_pairs_desc(dTempMemory.data(), tmSize,
                                                  keys, values,
                                                  totalElemCount, totalSegments,
                                                  dSegmentRanges.data(),
                                                  dSegmentRanges.data() + 1,
                                                  bitRange[0], bitRange[1],
                                                  ToHandleHIP(queue)));
    uint32_t result = (keys.current() == dKeyDoubleBuffer[0].data()) ? 0u : 1u;
    assert(((values.current() == dValueDoubleBuffer[0].data()) ? 0u : 1u) == result);
    return result;
}

}