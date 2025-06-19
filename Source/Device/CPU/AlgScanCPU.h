#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCPU.h"

namespace mray::host::algorithms
{

static constexpr uint32_t TPB = StaticThreadPerBlock1D();

template <class T, class BinaryOp>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCInclusiveSegmentedScan(Span<T> dOut,
                              Span<const T> dIn,
                              uint32_t segmentSize,
                              uint32_t totalBlocks,
                              T identityElement,
                              BinaryOp op)
{
}

template <class T>
MRAY_HOST
size_t ExclusiveScanTMSize(size_t elementCount)
{
    return 0u;
}

template <class T, class BinaryOp>
MRAY_HOST inline
void InclusiveSegmentedScan(Span<T> dScannedValues,
                            Span<const T> dValues,
                            uint32_t segmentSize,
                            const T& identityElement,
                            const GPUQueueCPU& queue,
                            BinaryOp&& op)
{
}

template <class T, class BinaryOp>
MRAY_HOST
void ExclusiveScan(Span<T> dScannedValues,
                   Span<Byte> dTempMem,
                   Span<const T> dValues,
                   const T& initialValue,
                   const GPUQueueCPU& queue,
                   BinaryOp&& op)
{
}

}