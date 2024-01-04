
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "Core/Types.h"
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

class GPUQueueCUDA;
class GPUDeviceCUDA;

class GPUFenceCUDA
{
    MRAY_HYBRID
    friend cudaEvent_t ToHandleCUDA(const GPUFenceCUDA&);

    private:
    cudaEvent_t                 eventC;

    public:
    MRAY_HYBRID                 GPUFenceCUDA(const GPUQueueCUDA&);
                                GPUFenceCUDA(const GPUFenceCUDA&) = delete;
    MRAY_HYBRID                 GPUFenceCUDA(GPUFenceCUDA&&) noexcept;
    GPUFenceCUDA&               operator=(const GPUFenceCUDA&) = delete;
    MRAY_HYBRID GPUFenceCUDA&   operator=(GPUFenceCUDA&&) noexcept;
    MRAY_HYBRID                 ~GPUFenceCUDA();

    MRAY_HYBRID void            Wait() const;
};

class GPUQueueCUDA
{
    MRAY_HYBRID
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
    MRAY_HYBRID                 GPUQueueCUDA(GPUQueueCUDA&&) noexcept;
    GPUQueueCUDA&               operator=(const GPUQueueCUDA&) = delete;
    MRAY_HYBRID GPUQueueCUDA&   operator=(GPUQueueCUDA&&) noexcept;
    MRAY_HYBRID                 ~GPUQueueCUDA();

    // Classic GPU Calls
    // Create just enough blocks according to work size
    template<auto DeviceFunction, class... Args>
    MRAY_HYBRID void    IssueKernel(KernelIssueParams,
                                    //
                                    Args&&...) const;
    template<class Lambda>
    MRAY_HYBRID void    IssueKernelL(KernelIssueParams,
                                     //
                                     Lambda&&) const;
    // Grid-Stride Kernels
    // Kernel is launched just enough blocks to
    // fully saturate the GPU.
    template<auto DeviceFunction, class... Args>
    MRAY_HYBRID void    IssueSaturatingKernel(KernelIssueParams,
                                              //
                                              Args&&...) const;
    template<class Lambda>
    MRAY_HYBRID void    IssueSaturatingKernelL(KernelIssueParams,
                                               //
                                               Lambda&&) const;
    // Exact Kernel Calls
    // You 1-1 specify block and grid dimensions
    // Important: These can not be annottated with launch_bounds
    template<auto DeviceFunction, class... Args>
    MRAY_HYBRID void    IssueExactKernel(KernelExactIssueParams,
                                         //
                                         Args&&...) const;
    template<class Lambda>
    MRAY_HYBRID void    IssueExactKernelL(KernelExactIssueParams,
                                         //
                                         Lambda&&) const;

    // Memory Movement (Async)
    template <class T>
    MRAY_HOST void      CopyAsync(Span<T> regionTo, Span<const T> regionFrom) const;

    template <uint32_t D, class T>
    MRAY_HOST void      CopyAsync(Texture<D, T>& texTo,
                                  const TextureDim<D>& offset,
                                  Span<const T> regionFrom) const;

    // Synchronization
    MRAY_HYBRID
    GPUFenceCUDA        Barrier() const;

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
                            GPUDeviceCUDA(GPUDeviceCUDA&&) noexcept = default;
    GPUDeviceCUDA&          operator=(const GPUDeviceCUDA&) = delete;
    GPUDeviceCUDA&          operator=(GPUDeviceCUDA&&) noexcept = default;
                            ~GPUDeviceCUDA() = default;

    bool                    operator==(const GPUDeviceCUDA&) const;

    int                     DeviceId() const;
    std::string             Name() const;
    std::string             ComputeCapability() const;
    size_t                  TotalMemory() const;

    uint32_t                SMCount() const;
    uint32_t                MaxActiveBlockPerSM(uint32_t threadsPerBlock = StaticThreadPerBlock1D()) const;

    const GPUQueueCUDA&     GetQueue(uint32_t index) const;
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
};

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCUDA::GPUFenceCUDA(const GPUQueueCUDA& q)
    : eventC((cudaEvent_t)0)
{
    CUDA_CHECK(cudaEventCreateWithFlags(&eventC, cudaEventDisableTiming));
    cudaStream_t stream = ToHandleCUDA(q);
    CUDA_CHECK(cudaEventRecord(eventC, stream));
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCUDA::GPUFenceCUDA(GPUFenceCUDA&& other) noexcept
    : eventC(other.eventC)
{
    other.eventC = (cudaEvent_t)0;
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCUDA& GPUFenceCUDA::operator=(GPUFenceCUDA&& other) noexcept
{
    eventC = other.eventC;
    other.eventC = (cudaEvent_t)0;
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCUDA::~GPUFenceCUDA()
{
    if(eventC != (cudaEvent_t)0)
        CUDA_CHECK(cudaEventDestroy(eventC));
}

MRAY_HYBRID MRAY_CGPU_INLINE
void GPUFenceCUDA::Wait() const
{
    #ifndef __CUDA_ARCH__
        CUDA_CHECK(cudaEventSynchronize(eventC));
    #else
        // TODO: Reason about this
        //
        // CUDA_CHECK(cudaStreamWaitEvent(stream, event));
        __trap();
    #endif
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCUDA::GPUQueueCUDA(uint32_t multiprocessorCount,
                           DeviceQueueType t)
    : multiprocessorCount(multiprocessorCount)
{

    switch(t)
    {
        case DeviceQueueType::NORMAL:
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream,
                                                 cudaStreamNonBlocking));
            break;
        default:
            assert(false);
            break;

        // Only valid on device
        #ifdef __CUDA_ARCH__
            case DeviceQueueType::FIRE_AND_FORGET:
                stream = cudaStreamFireAndForget;
                break;
            case DeviceQueueType::TAIL_LAUNCH:
                stream = cudaStreamTailLaunch;
                break;
        #endif
    }
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCUDA::~GPUQueueCUDA()
{
    #ifdef __CUDA_ARCH__
        if(stream != cudaStreamTailLaunch ||
           stream != cudaStreamFireAndForget ||
           stream != (cudaStream_t)0)
            CUDA_CHECK(cudaStreamDestroy(stream));
    #else
        if(stream != (cudaStream_t)0)
            CUDA_CHECK(cudaStreamDestroy(stream));
    #endif

}

// Memory Movement (Async)
template <class T>
MRAY_HOST void GPUQueueCUDA::CopyAsync(Span<T> regionTo, Span<const T> regionFrom) const
{
    assert(regionTo.size_bytes() == regionFrom.size_bytes());
    CUDA_CHECK(cudaMemcpyAsync(regionTo.data(), region.from.data(),
                          regionFrom.size_bytes(),
                          cudaMemcpyDefault, stream));
}

//template <uint32_t D, class T>
//MRAY_HOST void GPUQueueCUDA::CopyAsync(Texture<D, T>& texTo, uint32_t mipLevel,
//                                       const TextureSize<T>& texOffset,
//                                       Span<const T> regionFrom) const
//{
//    // Implementation is templated so its complex
//    texTo.Set()
//}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCUDA GPUQueueCUDA::Barrier() const
{
    return GPUFenceCUDA(*this);
}

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
GPUQueueCUDA::GPUQueueCUDA(GPUQueueCUDA&& other) noexcept
    : stream(other.stream)
    , multiprocessorCount(other.multiprocessorCount)
{
    other.stream = (cudaStream_t)0;
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCUDA& GPUQueueCUDA::operator=(GPUQueueCUDA&& other) noexcept
{
    multiprocessorCount = other.multiprocessorCount;
    stream = other.stream;
    other.stream = (cudaStream_t)0;
}

MRAY_HYBRID MRAY_CGPU_INLINE
cudaStream_t ToHandleCUDA(const GPUQueueCUDA& q)
{
    return q.stream;
}

MRAY_HYBRID MRAY_CGPU_INLINE
cudaEvent_t ToHandleCUDA(const GPUFenceCUDA& f)
{
    return f.eventC;
}

}


