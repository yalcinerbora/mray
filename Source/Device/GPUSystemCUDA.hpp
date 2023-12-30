
#pragma once

#include "GPUSystemCUDA.h"
#include "GPUSystem.h"

// Cuda Kernel Optimization Hints
// Since we call all of the kernels in a static manner
// (in case of Block Size) hint the compiler
// using __launch_bounds__ expression
#define MRAY_DEVICE_LAUNCH_BOUNDS(X) __launch_bounds__(X)
#define MRAY_DEVICE_LAUNCH_BOUNDS_1D \
        MRAY_DEVICE_LAUNCH_BOUNDS __launch_bounds__(StaticThreadPerBlock1D())

#define MRAY_GRID_CONSTANT __grid_constant__

namespace CudaKernelCalls
{
    template <class K, class... Args>
    MRAY_KERNEL static
    void KernelCallCUDA(K&& Kernel, Args&&... args)
    {
        // Grid-stride loop
        KernelCallParameters1D launchParams
        {
            .gridSize = gridDim.x,
            .blockSize = blockDim.x,
            .blockId = blockIdx.x,
            .threadId = threadIdx.x
        };
        std::forward<K>(Kernel)(launchParams, std::forward<Args>(args)...);
    }

    template <uint32_t TPB, class K, class... Args>
    __launch_bounds__(TPB)
    MRAY_KERNEL static
    void KernelCallCUDA(K&& Kernel, Args&&... args)
    {
        // Grid-stride loop
        KernelCallParameters1D launchParams
        {
            .gridSize = gridDim.x,
            .blockSize = blockDim.x,
            .blockId = blockIdx.x,
            .threadId = threadIdx.x
        };
        std::forward<K>(Kernel)(launchParams, std::forward<Args>(args)...);
    }
}

namespace mray::cuda
{

template<class Function, class... Args>
MRAY_HYBRID inline
void GPUQueueCUDA::IssueKernel(uint32_t sharedMemSize, uint32_t workCount,
                               Function&& f, Args&&... args) const
{
    using namespace CudaKernelCalls;
    uint32_t blockCount = MathFunctions::AlignToMultiple(workCount, StaticThreadPerBlock1D());
    uint32_t blockSize = StaticThreadPerBlock1D();

    KernelCallCUDA<StaticThreadPerBlock1D()>
    <<<blockCount, blockSize, sharedMemSize, stream>>>
    (
        std::forward<Function>(f),
        std::forward<Args>(args)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
MRAY_HOST inline
void GPUQueueCUDA::IssueKernel(uint32_t workCount,
                               Function&& f, Args&&... args) const
{
    KC_X(0, workCount, std::forward<Function>(f), std::forward<Args>(args)...);
}

template<class Function, class... Args>
MRAY_HOST inline
void GPUQueueCUDA::IssueSaturatingKernel(uint32_t sharedMemSize,
                                         uint32_t workCount,
                                         //
                                         Function&& f, Args&&... args) const
{
    using namespace CudaKernelCalls;

    const void* kernelPtr = reinterpret_cast<const void*>
        (&KernelCallCUDA<StaticThreadPerBlock1D(), Function, Args...>);
    uint32_t threadCount = StaticThreadPerBlock1D();
    uint32_t blockCount = DetermineGridStrideBlock(kernelPtr,
                                                   sharedMemSize,
                                                   threadCount,
                                                   workCount);


    KernelCallCUDA<StaticThreadPerBlock1D(), Function, Args...>
    <<<blockCount, threadCount, sharedMemSize, stream>>>
    (
        std::forward<Function>(f),
        std::forward<Args>(args)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
MRAY_HOST inline
void GPUQueueCUDA::IssueSaturatingKernel(uint32_t workCount,
                                         //
                                         Function&& f, Args&&... args) const
{
    SaturatingKC_X(0, workCount,
                   std::forward<Function>(f),
                   std::forward<Args>(args)...);
}

template<class Function, class... Args>
MRAY_HOST inline
void GPUQueueCUDA::IssueExactKernel(uint32_t sharedMemSize,
                                    uint32_t blockCount, uint32_t blockSize,
                                    //
                                    Function&& f, Args&&... args) const
{
    using namespace CudaKernelCalls;

    KernelCallCUDA
    <<<blockCount, blockSize, sharedMemSize, stream>>>
    (
        std::forward<Function>(f),
        std::forward<Args>(args)...
    );
    CUDA_KERNEL_CHECK();
}

template<class Function, class... Args>
MRAY_HOST inline
void GPUQueueCUDA::IssueExactKernel(uint32_t blockCount, uint32_t blockSize,
                                    //
                                    Function&& f, Args&&... args) const
{
    ExactKC_X(0, blockCount, blockSize,
              std::forward<Function>(f),
              std::forward<Args>(args)...);
}

}


