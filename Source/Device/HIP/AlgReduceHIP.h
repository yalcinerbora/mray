#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemHIP.h"

#include <rocprim/device/device_reduce.hpp>
#include <rocprim/device/device_segmented_reduce.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

namespace mray::hip::algorithms
{

template <class T>
MRAY_HOST inline
size_t ReduceTMSize(size_t elementCount)
{
    using namespace rocprim;

    T* dIn = nullptr;
    T* dOut = nullptr;
    void* dTM = nullptr;
    size_t result;
    HIP_CHECK(reduce(dTM, result, dIn, dOut,
                     T{}, static_cast<int>(elementCount),
                     [] MRAY_HYBRID(T, T)->T{ return T{}; }));
    return result;
}

template <class OutT, class InT>
MRAY_HOST inline
size_t TransformReduceTMSize(size_t elementCount)
{
    using namespace rocprim;

    auto TFunc = [] MRAY_HYBRID(InT) -> OutT{ return OutT{}; };
    using TransIt = transform_iterator<OutT, decltype(TFunc), const InT*>;
    TransIt dIn = TransIt(nullptr, TFunc);
    OutT* dOut = nullptr;
    void* dTM = nullptr;
    size_t result;
    HIP_CHECK(reduce(dTM, result, dIn, dOut,
                     static_cast<int>(elementCount),
                     [] MRAY_HYBRID(OutT, OutT)-> OutT { return OutT{}; },
                     OutT{}));
    return result;
}

template <class OutT, class InT>
MRAY_HOST inline
size_t SegmentedTransformReduceTMSize(size_t numSegments)
{
    using namespace rocprim;
    auto TFunc = [] MRAY_HYBRID(InT) -> OutT{ return OutT{}; };
    using TransIt = transform_iterator<OutT, decltype(TFunc), const InT*>;
    TransIt dIn = TransIt(nullptr, TFunc);
    uint32_t* dStartOffsets = nullptr;
    uint32_t* dEndOffsets = nullptr;
    OutT* dOut = nullptr;
    void* dTM = nullptr;

    size_t result;
    HIP_CHECK(segmented_reduce(dTM, result, dIn, dOut,
                               static_cast<int>(numSegments),
                               dStartOffsets, dEndOffsets,
                               [] MRAY_HYBRID(OutT, OutT)-> OutT { return OutT{}; },
                               OutT{}));
    return result;
}

template <class T, class BinaryOp>
MRAY_HOST inline
void Reduce(Span<T, 1> dReducedValue,
            Span<Byte> dTempMemory,
            Span<const T> dValues,
            const T& initialValue,
            const GPUQueueHIP& queue,
            BinaryOp&& op)
{
    using namespace rocprim;
    using namespace std::literals;
    static const auto annotation = queue.CreateAnnotation("KCReduce"sv);
    const auto _ = annotation.AnnotateScope();

    size_t size = dTempMemory.size();
    HIP_CHECK(reduce(dTempMemory.data(), size,
                     dValues.data(), dReducedValue.data(),
                     initialValue,
                     static_cast<int>(dValues.size()),
                     std::forward<BinaryOp>(op),
                     ToHandleHIP(queue)));
}

template <class OutT, class InT, class BinaryOp, class TransformOp>
MRAY_HOST inline
void TransformReduce(Span<OutT, 1> dReducedValue,
                     Span<Byte> dTempMemory,
                     Span<const InT> dValues,
                     const OutT& initialValue,
                     const GPUQueueHIP& queue,
                     BinaryOp&& binaryOp,
                     TransformOp&& transformOp)
{
    using namespace rocprim;
    using namespace std::literals;
    static const auto annotation = queue.CreateAnnotation("KCTransformReduce"sv);
    const auto _ = annotation.AnnotateScope();

    using TransIt = transform_iterator<OutT, TransformOp, const InT*>;
    TransIt dIn = TransIt(dValues.data(), std::forward<TransformOp>(transformOp));

    size_t size = dTempMemory.size();
    HIP_CHECK(reduce(dTempMemory.data(), size,
                     dIn, dReducedValue.data(),
                     static_cast<int>(dValues.size()),
                     std::forward<BinaryOp>(binaryOp),
                     initialValue, ToHandleHIP(queue)));
}

template <class OutT, class InT, class BinaryOp, class TransformOp>
MRAY_HOST inline
void SegmentedTransformReduce(Span<OutT> dReducedValues,
                              Span<Byte> dTempMemory,
                              Span<const InT> dValues,
                              Span<const uint32_t> dSegmentRanges,
                              const OutT& initialValue,
                              const GPUQueueHIP& queue,
                              BinaryOp&& binaryOp,
                              TransformOp&& transformOp)
{
    using namespace rocprim;
    using namespace std::literals;
    static const auto annotation = queue.CreateAnnotation("KCSegmentedTransformReduce"sv);
    const auto _ = annotation.AnnotateScope();
    using TransIt = transform_iterator<OutT, TransformOp, const InT*>;
    TransIt dIn = TransIt(dValues.data(), std::forward<TransformOp>(transformOp));

    int segmentCount = static_cast<int>(dSegmentRanges.size() - 1);
    size_t size = dTempMemory.size();
    HIP_CHECK(segmented_reduce(dTempMemory.data(), size,
                               dIn,
                               dReducedValues.data(),
                               segmentCount,
                               dSegmentRanges.data(),
                               dSegmentRanges.data() + 1,
                               std::forward<BinaryOp>(binaryOp),
                               initialValue, ToHandleHIP(queue)));
}

}