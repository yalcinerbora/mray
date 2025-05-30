#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemHIP.h"

#include <rocprim/device/device_partition.hpp>
#include <rocprim/device/device_select.hpp>

namespace mray::hip::algorithms
{

template <class T>
MRAY_HOST inline
size_t BinPartitionTMSize(size_t elementCount)
{
    using namespace rocprim;

    T* dOut = nullptr;
    T* dIn = nullptr;
    void* dTM = nullptr;
    uint32_t* dEndOffset = nullptr;
    size_t result;
    HIP_CHECK(partition(dTM, result, dIn, dOut, dEndOffset,
                        static_cast<int>(elementCount),
                        [] MRAY_HYBRID(T)->bool{ return false; }));

    return result;
}

template <class T, class UnaryOp>
MRAY_HOST inline
void BinaryPartition(Span<T> dOutput,
                     Span<uint32_t, 1> dEndOffset,
                     Span<Byte> dTempMemory,
                     Span<const T> dInput,
                     const GPUQueueHIP& queue,
                     UnaryOp&& op)
{
    using namespace rocprim;
    using namespace std::literals;
    static const auto annotation = queue.CreateAnnotation("KCBinaryPartition"sv);
    const auto _ = annotation.AnnotateScope();
    assert(dInput.size() == dOutput.size());

    size_t size = dTempMemory.size();
    HIP_CHECK(partition(dTempMemory.data(), size,
                        dInput.data(), dOutput.data(), dEndOffset.data(),
                        static_cast<int>(dInput.size()),
                        std::forward<UnaryOp>(op),
                        ToHandleHIP(queue)));
}
}