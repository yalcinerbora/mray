#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCPU.h"

namespace mray::host::algorithms
{

template <class T>
MRAY_HOST inline
size_t ReduceTMSize(size_t elementCount)
{
    return 0u;
}

template <class OutT, class InT>
MRAY_HOST inline
size_t TransformReduceTMSize(size_t elementCount)
{
    return 0u;
}

template <class OutT, class InT>
MRAY_HOST inline
size_t SegmentedTransformReduceTMSize(size_t numSegments)
{
    return 0u;
}

template <class T, class BinaryOp>
MRAY_HOST inline
void Reduce(Span<T, 1> dReducedValue,
            Span<Byte> dTempMemory,
            Span<const T> dValues,
            const T& initialValue,
            const GPUQueueCPU& queue,
            BinaryOp&& op)
{
}

template <class OutT, class InT, class BinaryOp, class TransformOp>
MRAY_HOST inline
void TransformReduce(Span<OutT, 1> dReducedValue,
                     Span<Byte> dTempMemory,
                     Span<const InT> dValues,
                     const OutT& initialValue,
                     const GPUQueueCPU& queue,
                     BinaryOp&& binaryOp,
                     TransformOp&& transformOp)
{
}

template <class OutT, class InT, class BinaryOp, class TransformOp>
MRAY_HOST inline
void SegmentedTransformReduce(Span<OutT> dReducedValues,
                              Span<Byte> dTempMemory,
                              Span<const InT> dValues,
                              Span<const uint32_t> dSegmentRanges,
                              const OutT& initialValue,
                              const GPUQueueCPU& queue,
                              BinaryOp&& binaryOp,
                              TransformOp&& transformOp)
{
}

}