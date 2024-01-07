#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCUDA.h"

#include <cub/device/device_reduce.cuh>

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
        size_t size = dTempMemory.size();
        CUDA_CHECK(DeviceReduce::Reduce(dTempMemory.data(), size,
                                        dValues.data(), dReducedValue.data(),
                                        static_cast<int>(dValues.size()),
                                        std::forward<BinaryOp>(op), initialValue,
                                        ToHandleCUDA(queue)));
    }

}