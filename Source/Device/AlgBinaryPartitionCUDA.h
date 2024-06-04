#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCUDA.h"

#include <cub/device/device_partition.cuh>

#include <cub/device/device_select.cuh>

// Direct wrappers over CUB at the moment
// Probably be refactored later
// Add as you need
namespace mray::cuda::algorithms
{

template <class T>
MRAY_HOST
size_t BinPartitionTMSize(size_t elementCount);

template <class T, class UnaryOp>
MRAY_HOST
void BinaryPartition(Span<T> dOutput,
                     Span<uint32_t, 1> dEndOffset,
                     Span<Byte> dTempMemory,
                     Span<const T> dInput,
                     const GPUQueueCUDA& queue,
                     UnaryOp&&);
}

namespace mray::cuda::algorithms
{

template <class T>
MRAY_HOST inline
size_t BinPartitionTMSize(size_t elementCount)
{
    using namespace cub;

    T* dOut = nullptr;
    T* dIn = nullptr;
    void* dTM = nullptr;
    uint32_t* dEndOffset = nullptr;
    size_t result;
    CUDA_CHECK(DevicePartition::If(dTM, result, dIn, dOut, dEndOffset,
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
                     const GPUQueueCUDA& queue,
                     UnaryOp&& op)
{
    using namespace cub;
    using namespace std::literals;
    static const NVTXKernelName kernelName = NVTXKernelName(queue.ProfilerDomain(), "KCBinaryPartition"sv);
    NVTXAnnotate annotate = kernelName.Annotate();

    assert(dInput.size() == dOutput.size());


    size_t size = dTempMemory.size();
    CUDA_CHECK(DevicePartition::If(dTempMemory.data(), size,
                                   dInput.data(), dOutput.data(), dEndOffset.data(),
                                   static_cast<int>(dInput.size()),
                                   std::forward<UnaryOp>(op),
                                   ToHandleCUDA(queue)));
}
}