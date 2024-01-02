
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "Core/MathFunctions.h"
#include "DefinitionsCUDA.h"
#include "GPUTypes.h"

// Cuda Kernel Optimization Hints
// Since we call all of the kernels in a static manner
// (in case of Block Size) hint the compiler
// using __launch_bounds__ expression
#define MRAY_DEVICE_LAUNCH_BOUNDS(X) __launch_bounds__(X)
#define MRAY_DEVICE_LAUNCH_BOUNDS_1D \
        MRAY_DEVICE_LAUNCH_BOUNDS __launch_bounds__(StaticThreadPerBlock1D())

#define MRAY_GRID_CONSTANT __grid_constant__

static constexpr uint32_t WarpSize()
{
    return 32u;
}

// A Good guess for TPB
static constexpr uint32_t StaticThreadPerBlock1D()
{
    return 512u;
}

// TODO: This should not be compile time static
static constexpr uint32_t TotalQueuePerDevice()
{
    return 4;
}

namespace mray::cuda
{

class GPUDeviceCUDA;
class DeviceMemoryCUDA;
class DeviceLocalMemoryCUDA;
template<uint32_t DIMS, class T>
class DeviceTextureCUDA;

class GPUQueueCUDA
{
    friend cudaStream_t ToHandleCUDA(const GPUQueueCUDA&);

    private:
    cudaStream_t        stream;
    uint32_t            multiprocessorCount;

    MRAY_HYBRID
    uint32_t            DetermineGridStrideBlock(const void* kernelPtr,
                                                 uint32_t sharedMemSize,
                                                 uint32_t threadCount,
                                                 uint32_t workCount) const;

    public:
    // Constructors & Destructor
    MRAY_HYBRID                 GPUQueueCUDA(uint32_t multiprocessorCount,
                                             DeviceQueueType t = DeviceQueueType::NORMAL);
                                GPUQueueCUDA(const GPUQueueCUDA&) = delete;
    MRAY_HYBRID                 GPUQueueCUDA(GPUQueueCUDA&&);
    GPUQueueCUDA&               operator=(const GPUQueueCUDA&) = delete;
    MRAY_HYBRID GPUQueueCUDA&   operator=(GPUQueueCUDA&&);
    MRAY_HYBRID                 ~GPUQueueCUDA();

    // Classic GPU Calls
    // Create just enough blocks according to work size
    template<class Function, class... Args>
    MRAY_HYBRID void    IssueKernel(uint32_t sharedMemSize,
                                    uint32_t workCount,
                                    //
                                    Function&&, Args&&...) const;
    template<class Function, class... Args>
    MRAY_HYBRID void    IssueKernel(uint32_t workCount,
                                    //
                                    Function&&, Args&&...) const;
    // Grid-Stride Kernels
    // Kernel is launched just enough blocks to
    // fully saturate the GPU.
    template<class Function, class... Args>
    MRAY_HOST void      IssueSaturatingKernel(uint32_t sharedMemSize,
                                              uint32_t workCount,
                                              //
                                              Function&&, Args&&...) const;
    template<class Function, class... Args>
    MRAY_HOST void      IssueSaturatingKernel(uint32_t workCount,
                                              //
                                              Function&&, Args&&...) const;
    // Exact Kernel Calls
    // You 1-1 specify block and grid dimensions
    // Important: These can not be annottated with launch_bounds
    template<class Function, class... Args>
    MRAY_HYBRID void    IssueExactKernel(uint32_t sharedMemSize,
                                         uint32_t gridDim, uint32_t blockDim,
                                         //
                                         Function&&, Args&&...) const;
    template<class Function, class... Args>
    MRAY_HYBRID void    IssueExactKernel(uint32_t gridDim, uint32_t blockDim,
                                         //
                                         Function&&, Args&&...) const;

    MRAY_HYBRID
    static uint32_t     RecommendedBlockCountPerSM(const void* kernelPtr,
                                                   uint32_t threadsPerBlock,
                                                   uint32_t sharedMemSize);
};

class GPUDeviceCUDA
{
    using DeviceQueues = std::vector<GPUQueueCUDA>;

    private:
    int                     deviceId;
    cudaDeviceProp          props;
    DeviceQueues            queues;

    protected:
    public:
    // Constructors & Destructor
    explicit                GPUDeviceCUDA(int deviceId);
                            GPUDeviceCUDA(const GPUDeviceCUDA&) = delete;
                            GPUDeviceCUDA(GPUDeviceCUDA&&) = default;
    GPUDeviceCUDA&          operator=(const GPUDeviceCUDA&) = delete;
    GPUDeviceCUDA&          operator=(GPUDeviceCUDA&&) = default;
                            ~GPUDeviceCUDA() = default;

    bool                    operator==(const GPUDeviceCUDA&) const;

    int                     DeviceId() const;
    std::string             Name() const;
    std::string             ComputeCapability() const;
    size_t                  TotalMemory() const;

    uint32_t                SMCount() const;
    uint32_t                MaxActiveBlockPerSM(uint32_t threadsPerBlock = StaticThreadPerBlock1D()) const;

    const GPUQueueCUDA&     GetQueue(uint32_t index) const;

    // Memory Related
    DeviceLocalMemoryCUDA   GetAMemory() const;
    //template <uint32_t DIM, class T>
    //TextureCUDA<DIM,T>      GetATexture() const;

};

class GPUSystemCUDA
{
    public:
    using GPUList = std::vector<GPUDeviceCUDA>;

    private:
    GPUList                 systemGPUs;

    protected:
    public:
    // Constructors & Destructor
                            GPUSystemCUDA();
                            GPUSystemCUDA(const GPUSystemCUDA&) = delete;
                            GPUSystemCUDA(GPUSystemCUDA&&) = delete;
    GPUSystemCUDA&          operator=(const GPUSystemCUDA&) = delete;
    GPUSystemCUDA&          operator=(GPUSystemCUDA&&) = delete;

    // Multi-Device Splittable Smart GPU Calls
    // Automatic device split and stream split on devices
    std::vector<size_t>     SplitWorkToMultipleGPU(uint32_t workCount,
                                                   uint32_t threadCount,
                                                   uint32_t sharedMemSize,
                                                   void* f) const;

    // Misc
    const GPUList&          SystemDevices() const;
    const GPUDeviceCUDA&    BestDevice() const;

    // Get Kernel Attributes & Set Dynamic Shared Memory Size
    KernelAttributes        GetKernelAttributes(const void* kernelPtr) const;
    bool                    SetKernelShMemSize(const void* kernelPtr,
                                               int sharedMemConfigSize) const;

    size_t                  TotalMemory() const;

    // Simple & Slow System Synchronization
    void                    SyncAll() const;

    // MemoryRelated
    DeviceMemoryCUDA        GetAMemory() const;
};

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t GPUQueueCUDA::RecommendedBlockCountPerSM(const void* kernelPtr,
                                                  uint32_t threadsPerBlock,
                                                  uint32_t sharedMemSize)
{
    int numBlocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
                                                             kernelPtr,
                                                             threadsPerBlock,
                                                             sharedMemSize));
    return static_cast<uint32_t>(numBlocks);
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t GPUQueueCUDA::DetermineGridStrideBlock(const void* kernelPtr,
                                                uint32_t sharedMemSize,
                                                uint32_t threadCount,
                                                uint32_t workCount) const
{
    // TODO: Make better SM determination
    uint32_t blockPerSM = RecommendedBlockCountPerSM(kernelPtr, threadCount, sharedMemSize);
    // Only call enough SM
    uint32_t totalRequiredBlocks = MathFunctions::DivideUp(workCount, threadCount);
    uint32_t requiredSMCount = (totalRequiredBlocks + blockPerSM - 1) / blockPerSM;
    uint32_t smCount = std::min(multiprocessorCount, requiredSMCount);
    uint32_t blockCount = std::min(requiredSMCount, smCount * blockPerSM);
    return blockCount;
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCUDA::GPUQueueCUDA(GPUQueueCUDA&& other)
    : stream(other.stream)
    , multiprocessorCount(other.multiprocessorCount)
{
    other.stream = (cudaStream_t)0;
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCUDA& GPUQueueCUDA::operator=(GPUQueueCUDA&& other)
{
    multiprocessorCount = other.multiprocessorCount;
    stream = other.stream;
    other.stream = (cudaStream_t)0;
}


inline cudaStream_t ToHandleCUDA(const GPUQueueCUDA& q)
{
    return q.stream;
}

}


