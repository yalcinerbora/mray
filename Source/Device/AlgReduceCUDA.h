#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCUDA.h"

#include <cub/device/device_reduce.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

// Direct wrappers over CUB at the moment
// Probably be refactored later
// Add as you need
namespace mray::cuda::algorithms
{

template <class T>
MRAY_HOST
size_t ReduceTMSize(size_t elementCount);

template <class OutT, class InT>
MRAY_HOST
size_t TransformReduceTMSize(size_t elementCount);

template <class T, class BinaryOp>
MRAY_HOST
void Reduce(Span<T, 1> dReducedValue,
            Span<Byte> dTempMemory,
            Span<const T> dValues,
            const T& initialValue,
            const GPUQueueCUDA& queue,
            BinaryOp&&);

template <class OutT, class InT, class BinaryOp, class TransformOp>
MRAY_HOST
void TransformReduce(Span<OutT, 1> dReducedValue,
                     Span<Byte> dTempMemory,
                     Span<const InT> dValues,
                     const OutT& initialValue,
                     const GPUQueueCUDA& queue,
                     BinaryOp&&,
                     TransformOp&&);
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

template <class OutT, class InT>
MRAY_HOST inline
size_t TransformReduceTMSize(size_t elementCount)
{
    using namespace cub;

    auto TFunc = [] MRAY_HYBRID(InT) -> OutT{ return OutT{}; };
    using TransIt = cub::TransformInputIterator<InT, decltype(TFunc), OutT*>;
    TransIt dIn = TransIt(nullptr, TFunc);
    OutT* dOut = nullptr;
    void* dTM = nullptr;
    size_t result;
    CUDA_CHECK(DeviceReduce::Reduce(dTM, result, dIn, dOut,
                                    static_cast<int>(elementCount),
                                    [] MRAY_HYBRID(OutT, OutT)-> OutT{ return OutT{}; },
                                    OutT{}));
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
    static const NVTXKernelName kernelName = NVTXKernelName(queue.ProfilerDomain(), "KCReduce"sv);
    NVTXAnnotate annotate = kernelName.Annotate();

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
    static const NVTXKernelName kernelName = NVTXKernelName(queue.ProfilerDomain(),
                                                            "KCTransformReduce"sv);
    NVTXAnnotate annotate = kernelName.Annotate();

    using TransIt = cub::TransformInputIterator<InT, TransformOp, OutT*>;
    TransIt dIn = TransIt(dValues.data(), std::forward<TransformOp>(transformOp));

    size_t size = dTempMemory.size();
    CUDA_CHECK(DeviceReduce::Reduce(dTempMemory.data(), size,
                                    dIn, dReducedValue.data(),
                                    static_cast<int>(dValues.size()),
                                    std::forward<BinaryOp>(binaryOp),
                                    initialValue, ToHandleCUDA(queue)));
}

}