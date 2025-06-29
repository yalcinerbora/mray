#pragma once

#include "../GPUSystemForward.h"
#include "Core/MathForward.h"

namespace mray::host::algorithms
{
    //====================//
    //  Binary Partition  //
    //====================//
    template <class T>
    MRAY_HOST
    size_t BinPartitionTMSize(size_t elementCount, const GPUQueueCPU&);

    template <class T, class UnaryOp>
    MRAY_HOST
    void BinaryPartition(Span<T> dOutput,
                         Span<uint32_t, 1> dEndOffset,
                         Span<Byte> dTempMemory,
                         Span<const T> dInput,
                         const GPUQueueCPU& queue,
                         UnaryOp&&);

    //====================//
    //   Binary Search    //
    //====================//
    template <class T>
    MRAY_HYBRID
    size_t LowerBound(Span<const T>, const T& value);

    //====================//
    //     Radix Sort     //
    //====================//
    template <bool IsAscending, class K, class V>
    MRAY_HOST
    size_t RadixSortTMSize(size_t elementCount, const GPUQueueCPU&);

    template <bool IsAscending, class K, class V>
    MRAY_HOST
    uint32_t RadixSort(Span<Span<K>, 2> dKeyDoubleBuffer,
                       Span<Span<V>, 2> dValueDoubleBuffer,
                       Span<Byte> dTempMemory,
                       const GPUQueueCPU& queue,
                       const Vector2ui& bitRange = Vector2ui(0, sizeof(K) * CHAR_BIT));

    template <bool IsAscending, class K, class V>
    MRAY_HOST
    size_t SegmentedRadixSortTMSize(size_t totalElementCount,
                                    size_t totalSegments,
                                    const GPUQueueCPU&);

    template <bool IsAscending, class K, class V>
    MRAY_HOST
    uint32_t SegmentedRadixSort(Span<Span<K>, 2> dKeyDoubleBuffer,
                               Span<Span<V>, 2> dValueDoubleBuffer,
                               Span<Byte> dTempMemory,
                               Span<const uint32_t> dSegmentRanges,
                               const GPUQueueCPU& queue,
                               const Vector2ui& bitRange = Vector2ui(0, sizeof(K) * CHAR_BIT));

    //====================//
    //       Reduce       //
    //====================//
    template <class T>
    MRAY_HOST
    size_t ReduceTMSize(size_t elementCount, const GPUQueueCPU&);

    template <class OutT, class InT>
    MRAY_HOST
    size_t TransformReduceTMSize(size_t elementCount, const GPUQueueCPU&);

    template <class OutT, class InT>
    MRAY_HOST
    size_t SegmentedTransformReduceTMSize(size_t numSegments, const GPUQueueCPU&);

    template <class T, class BinaryOp>
    MRAY_HOST
    void Reduce(Span<T, 1> dReducedValue,
                Span<Byte> dTempMemory,
                Span<const T> dValues,
                const T& initialValue,
                const GPUQueueCPU& queue,
                BinaryOp&&);

    template <class OutT, class InT, class BinaryOp, class TransformOp>
    MRAY_HOST
    void TransformReduce(Span<OutT, 1> dReducedValue,
                         Span<Byte> dTempMemory,
                         Span<const InT> dValues,
                         const OutT& initialValue,
                         const GPUQueueCPU& queue,
                         BinaryOp&&,
                         TransformOp&&);

    template <class OutT, class InT, class BinaryOp, class TransformOp>
    MRAY_HOST
    void SegmentedTransformReduce(Span<OutT> dReducedValue,
                                  Span<Byte> dTempMemory,
                                  Span<const InT> dValues,
                                  Span<const uint32_t> dSegmentRanges,
                                  const OutT& initialValue,
                                  const GPUQueueCPU& queue,
                                  BinaryOp&&,
                                  TransformOp&&);

    //====================//
    //        Scan        //
    //====================//
    template <class T>
    MRAY_HOST
    size_t ExclusiveScanTMSize(size_t elementCount, const GPUQueueCPU&);

    template <class T, class BinaryOp>
    MRAY_HOST
    void InclusiveSegmentedScan(Span<T> dScannedValues,
                                Span<const T> dValues,
                                uint32_t segmentSize,
                                const T& identityElement,
                                const GPUQueueCPU& queue,
                                BinaryOp&& op);

    template <class T, class BinaryOp>
    MRAY_HOST
    void ExclusiveScan(Span<T> dScannedValues,
                       Span<Byte> dTempMem,
                       Span<const T> dValues,
                       const T& initialValue,
                       const GPUQueueCPU& queue,
                       BinaryOp&& op);


}