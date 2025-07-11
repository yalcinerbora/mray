#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCUDA.h"

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <thrust/iterator/transform_iterator.h>

namespace mray::cuda::algorithms
{

template <class T>
MRAY_HOST inline
size_t ReduceTMSize(size_t elementCount, const GPUQueueCUDA& q)
{
    using namespace cub;

    T* dIn = nullptr;
    T* dOut = nullptr;
    void* dTM = nullptr;
    size_t result;
    CUDA_CHECK(DeviceReduce::Reduce(dTM, result, dIn, dOut,
                                    static_cast<int>(elementCount),
                                    [] MRAY_HYBRID(T, T)->T{ return T{}; }, T{},
                                    ToHandleCUDA(q)));
    return result;
}

template <class OutT, class InT>
MRAY_HOST inline
size_t TransformReduceTMSize(size_t elementCount, const GPUQueueCUDA& q)
{
    using namespace cub;

    auto TFunc = [] MRAY_HYBRID(InT) -> OutT{ return OutT{}; };
    using TransIt = thrust::transform_iterator<decltype(TFunc), const InT*, OutT>;
    TransIt dIn = TransIt(nullptr, TFunc);
    OutT* dOut = nullptr;
    void* dTM = nullptr;
    size_t result;
    CUDA_CHECK(DeviceReduce::Reduce(dTM, result, dIn, dOut,
                                    static_cast<int>(elementCount),
                                    [] MRAY_HYBRID(OutT, OutT)-> OutT { return OutT{}; },
                                    OutT{}, ToHandleCUDA(q)));
    return result;
}

template <class OutT, class InT>
MRAY_HOST inline
size_t SegmentedTransformReduceTMSize(size_t numSegments, const GPUQueueCUDA& q)
{
    using namespace cub;
    auto TFunc = [] MRAY_HYBRID(InT) -> OutT{ return OutT{}; };
    using TransIt = thrust::transform_iterator<decltype(TFunc), const InT*, OutT>;
    TransIt dIn = TransIt(nullptr, TFunc);
    uint32_t* dStartOffsets = nullptr;
    uint32_t* dEndOffsets = nullptr;
    OutT* dOut = nullptr;
    void* dTM = nullptr;

    size_t result;
    CUDA_CHECK(DeviceSegmentedReduce::Reduce(dTM, result, dIn, dOut,
                                             static_cast<int>(numSegments),
                                             dStartOffsets, dEndOffsets,
                                             [] MRAY_HYBRID(OutT, OutT)-> OutT { return OutT{}; },
                                             OutT{}, ToHandleCUDA(q)));
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
    using namespace std::literals;
    static const auto annotation = queue.CreateAnnotation("KCReduce"sv);
    const auto _ = annotation.AnnotateScope();

    size_t size = dTempMemory.size();
    CUDA_CHECK(DeviceReduce::Reduce(dTempMemory.data(), size,
                                    dValues.data(), dReducedValue.data(),
                                    static_cast<int>(dValues.size()),
                                    std::forward<BinaryOp>(op), initialValue,
                                    ToHandleCUDA(queue)));
}

template <class OutT, class InT, class BinaryOp, class TransformOp>
MRAY_HOST inline
void TransformReduce(Span<OutT, 1> dReducedValue,
                     Span<Byte> dTempMemory,
                     Span<const InT> dValues,
                     const OutT& initialValue,
                     const GPUQueueCUDA& queue,
                     BinaryOp&& binaryOp,
                     TransformOp&& transformOp)
{
    using namespace cub;
    using namespace std::literals;
    static const auto annotation = queue.CreateAnnotation("KCTransformReduce"sv);
    const auto _ = annotation.AnnotateScope();

    using TransIt = thrust::transform_iterator<TransformOp, const InT*, OutT>;
    TransIt dIn = TransIt(dValues.data(), std::forward<TransformOp>(transformOp));

    size_t size = dTempMemory.size();
    CUDA_CHECK(DeviceReduce::Reduce(dTempMemory.data(), size,
                                    dIn, dReducedValue.data(),
                                    static_cast<int>(dValues.size()),
                                    std::forward<BinaryOp>(binaryOp),
                                    initialValue, ToHandleCUDA(queue)));
}

template <class OutT, class InT, class BinaryOp, class TransformOp>
MRAY_HOST inline
void SegmentedTransformReduce(Span<OutT> dReducedValues,
                              Span<Byte> dTempMemory,
                              Span<const InT> dValues,
                              Span<const uint32_t> dSegmentRanges,
                              const OutT& initialValue,
                              const GPUQueueCUDA& queue,
                              BinaryOp&& binaryOp,
                              TransformOp&& transformOp)
{
    using namespace cub;
    using namespace std::literals;
    static const auto annotation = queue.CreateAnnotation("KCSegmentedTransformReduce"sv);
    const auto _ = annotation.AnnotateScope();
    //using TransIt = thrust::transform_iterator<OutT, TransformOp, const InT*>;
    using TransIt = thrust::transform_iterator<TransformOp, const InT*, OutT>;
    TransIt dIn = TransIt(dValues.data(), std::forward<TransformOp>(transformOp));

    int segmentCount = static_cast<int>(dSegmentRanges.size() - 1);
    size_t size = dTempMemory.size();
    CUDA_CHECK(DeviceSegmentedReduce::Reduce(dTempMemory.data(), size,
                                             dIn,
                                             dReducedValues.data(),
                                             segmentCount,
                                             dSegmentRanges.data(),
                                             dSegmentRanges.data() + 1,
                                             std::forward<BinaryOp>(binaryOp),
                                             initialValue, ToHandleCUDA(queue)));
}

}