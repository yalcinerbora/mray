
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "Core/Types.h"
#include "Core/MathFunctions.h"
#include "DefinitionsCUDA.h"

#include "../GPUTypes.h"

// Cuda Kernel Optimization Hints
// Since we call all of the kernels in a static manner
// (in case of Block Size) hint the compiler
// using __launch_bounds__ expression
#define MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(X) __launch_bounds__(X)
#define MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT \
        MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(StaticThreadPerBlock1D())

#define MRAY_DEVICE_BLOCK_SYNC() __syncthreads()

#if __CUDA_ARCH__ >= 700
    #define MRAY_GRID_CONSTANT __grid_constant__
#else
    #define MRAY_GRID_CONSTANT
#endif

#define MRAY_SHARED_MEMORY __shared__

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

using AnnotationHandle = void*;

class GPUQueueCUDA;
class GPUDeviceCUDA;

// Generic Call Parameters
struct KernelCallParamsCUDA
{
    uint32_t gridSize;
    uint32_t blockSize;
    uint32_t blockId;
    uint32_t threadId;

    MRAY_GPU                KernelCallParamsCUDA();
    MRAY_HYBRID uint32_t    GlobalId() const;
    MRAY_HYBRID uint32_t    TotalSize() const;
};

class GPUSemaphoreCUDA
{
    friend class GPUQueueCUDA;

    private:
    cudaExternalSemaphore_t semCUDA;
    SystemSemaphoreHandle   semSystem;

    public:
    // Constructors & Destructor
                        GPUSemaphoreCUDA(SystemSemaphoreHandle sem);
                        GPUSemaphoreCUDA(const GPUSemaphoreCUDA&) = delete;
                        GPUSemaphoreCUDA(GPUSemaphoreCUDA&&);
    GPUSemaphoreCUDA&   operator=(const GPUSemaphoreCUDA&) = delete;
    GPUSemaphoreCUDA&   operator=(GPUSemaphoreCUDA&&);
                        ~GPUSemaphoreCUDA();
};

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
    cudaStream_t            stream;
    uint32_t                multiprocessorCount;
    AnnotationHandle        nvtxDomain;
    const GPUDeviceCUDA*    myDevice = nullptr;

    MRAY_HYBRID
    uint32_t            DetermineGridStrideBlock(const void* kernelPtr,
                                                 uint32_t sharedMemSize,
                                                 uint32_t threadCount,
                                                 uint32_t workCount) const;

    public:
    // Constructors & Destructor
    MRAY_HOST                   GPUQueueCUDA(uint32_t multiprocessorCount,
                                             AnnotationHandle domain,
                                             const GPUDeviceCUDA* device);
    MRAY_GPU                    GPUQueueCUDA(uint32_t multiprocessorCount,
                                             AnnotationHandle domain,
                                             DeviceQueueType t);
                                GPUQueueCUDA(const GPUQueueCUDA&) = delete;
    MRAY_HYBRID                 GPUQueueCUDA(GPUQueueCUDA&&) noexcept;
    GPUQueueCUDA&               operator=(const GPUQueueCUDA&) = delete;
    MRAY_HYBRID GPUQueueCUDA&   operator=(GPUQueueCUDA&&) noexcept;
    MRAY_HYBRID                 ~GPUQueueCUDA();

    // Classic GPU Calls
    // Create just enough blocks according to work size
    template<auto Kernel, class... Args>
    MRAY_HOST void  IssueKernel(std::string_view name,
                                KernelIssueParams,
                                //
                                Args&&...) const;
    template<class Lambda>
    MRAY_HOST void  IssueLambda(std::string_view name,
                                KernelIssueParams,
                                //
                                Lambda&&) const;
    // Grid-Stride Kernels
    // Kernel is launched just enough blocks to
    // fully saturate the GPU.
    template<auto Kernel, class... Args>
    MRAY_HOST void  IssueSaturatingKernel(std::string_view name,
                                          KernelIssueParams,
                                          //
                                          Args&&...) const;
    template<class Lambda>
    MRAY_HOST void  IssueSaturatingLambda(std::string_view name,
                                          KernelIssueParams,
                                          //
                                          Lambda&&) const;
    // Exact Kernel Calls
    // You 1-1 specify block and grid dimensions
    // Important: These can not be annottated with launch_bounds
    template<auto Kernel, class... Args>
    MRAY_HOST void  IssueExactKernel(std::string_view name,
                                     KernelExactIssueParams,
                                     //
                                     Args&&...) const;
    template<class Lambda, uint32_t Bounds = StaticThreadPerBlock1D()>
    MRAY_HOST void  IssueExactLambda(std::string_view name,
                                     KernelExactIssueParams,
                                     //
                                     Lambda&&) const;

    // Split the device side kernel calls. glibc++ span has
    // explicitly defined defaulted destructor and NVCC errors
    // because of that even if we dont call the kernel from the
    // device.
    template<auto Kernel, class... Args>
    MRAY_GPU void   DeviceIssueKernel(std::string_view name,
                                      KernelIssueParams,
                                      //
                                      Args&&...) const;
    template<class Lambda>
    MRAY_GPU void   DeviceIssueLambda(std::string_view name,
                                      KernelIssueParams,
                                      //
                                      Lambda&&) const;
    template<auto Kernel, class... Args>
    MRAY_GPU void   DeviceIssueSaturatingKernel(std::string_view name,
                                                KernelIssueParams,
                                                //
                                                Args&&...) const;
    template<class Lambda>
    MRAY_GPU void   DeviceIssueSaturatingLambda(std::string_view name,
                                                KernelIssueParams,
                                                //
                                                Lambda&&) const;
    template<auto Kernel, class... Args>
    MRAY_GPU void   DeviceIssueExactKernel(std::string_view name,
                                           KernelExactIssueParams,
                                           //
                                           Args&&...) const;
    template<class Lambda, uint32_t Bounds = StaticThreadPerBlock1D()>
    MRAY_GPU void   DeviceIssueExactLambda(std::string_view name,
                                           KernelExactIssueParams,
                                           //
                                           Lambda&&) const;

    // Memory Movement (Async)
    template <class T>
    MRAY_HOST void      MemcpyAsync(Span<T> regionTo, Span<const T> regionFrom) const;
    template <class T>
    MRAY_HOST void      MemcpyAsyncStrided(Span<T> regionTo, size_t outputByteStride,
                                           Span<const T> regionFrom, size_t inputByteStride) const;
    template <class T>
    MRAY_HOST void      MemsetAsync(Span<T> region, uint8_t perByteValue) const;

    MRAY_HOST void      IssueBufferForDestruction(TransientData data) const;

    // Synchronization
    MRAY_HYBRID
    GPUFenceCUDA        Barrier() const;
    MRAY_HOST
    void                IssueSemaphoreWait(GPUSemaphoreCUDA&, uint64_t waitValue) const;
    MRAY_HOST
    void                IssueSemaphoreSignal(GPUSemaphoreCUDA&, uint64_t signalValue) const;

    MRAY_HYBRID
    uint32_t            SMCount() const;

    MRAY_HYBRID
    static uint32_t     RecommendedBlockCountPerSM(const void* kernelPtr,
                                                   uint32_t threadsPerBlock,
                                                   uint32_t sharedMemSize);

    MRAY_HOST
    AnnotationHandle        ProfilerDomain() const;
    MRAY_HOST
    const GPUDeviceCUDA*    Device() const;
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
    explicit                GPUDeviceCUDA(int deviceId, AnnotationHandle);
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
    using GPUPtrList = std::vector<const GPUDeviceCUDA*>;

    private:
    GPUList             systemGPUs;
    GPUPtrList          systemGPUPtrs;
    AnnotationHandle    nvtxDomain;

    // TODO: Check designs for this, this made the GPUSystem global
    // which is fine
    static GPUList*     globalGPUListPtr;
    static void         ThreadInitFunction();

    protected:
    public:
    // Constructors & Destructor
                        GPUSystemCUDA();
                        GPUSystemCUDA(const GPUSystemCUDA&) = delete;
                        GPUSystemCUDA(GPUSystemCUDA&&) = delete;
    GPUSystemCUDA&      operator=(const GPUSystemCUDA&) = delete;
    GPUSystemCUDA&      operator=(GPUSystemCUDA&&) = delete;
                        ~GPUSystemCUDA();

    // Multi-Device Splittable Smart GPU Calls
    // Automatic device split and stream split on devices
    std::vector<size_t> SplitWorkToMultipleGPU(uint32_t workCount,
                                               uint32_t threadCount,
                                               uint32_t sharedMemSize,
                                               void* f) const;

    // Misc
    const GPUList&          SystemDevices() const;
    const GPUDeviceCUDA&    BestDevice() const;
    const GPUPtrList&       AllGPUs() const;

    // Get Kernel Attributes & Set Dynamic Shared Memory Size
    KernelAttributes        GetKernelAttributes(const void* kernelPtr) const;
    bool                    SetKernelShMemSize(const void* kernelPtr,
                                               int sharedMemConfigSize) const;

    size_t                  TotalMemory() const;

    template <class T>
    MRAY_HOST void          Memcpy(Span<T> regionTo, Span<const T> regionFrom) const;
    template <class T>
    MRAY_HOST void          Memset(Span<T> region, uint8_t perByteValue) const;

    // Simple & Slow System Synchronization
    void                    SyncAll() const;

    // Thread Initialization Function, should be called for every thread
    // that will run GPU code
    [[nodiscard]]
    GPUThreadInitFunction   GetThreadInitFunction() const;
};


MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t KernelCallParamsCUDA::GlobalId() const
{
    return blockId * blockSize + threadId;
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t KernelCallParamsCUDA::TotalSize() const
{
    return gridSize * blockSize;
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCUDA::GPUFenceCUDA(const GPUQueueCUDA& q)
    : eventC(cudaEvent_t(0))
{
    CUDA_CHECK(cudaEventCreateWithFlags(&eventC, cudaEventDisableTiming));
    cudaStream_t stream = ToHandleCUDA(q);
    CUDA_CHECK(cudaEventRecord(eventC, stream));
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCUDA::GPUFenceCUDA(GPUFenceCUDA&& other) noexcept
    : eventC(other.eventC)
{
    other.eventC = cudaEvent_t(0);
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCUDA& GPUFenceCUDA::operator=(GPUFenceCUDA&& other) noexcept
{
    eventC = other.eventC;
    other.eventC = cudaEvent_t(0);
    return *this;
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCUDA::~GPUFenceCUDA()
{
    if(eventC != cudaEvent_t(0))
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

MRAY_HOST inline
GPUQueueCUDA::GPUQueueCUDA(uint32_t multiprocessorCount,
                           AnnotationHandle domain,
                           const GPUDeviceCUDA* device)
    : multiprocessorCount(multiprocessorCount)
    , nvtxDomain(domain)
    , myDevice(device)
{
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream,
                                         cudaStreamNonBlocking));
}

MRAY_GPU MRAY_GPU_INLINE
GPUQueueCUDA::GPUQueueCUDA(uint32_t multiprocessorCount,
                           AnnotationHandle domain,
                           DeviceQueueType t)
    : multiprocessorCount(multiprocessorCount)
    , nvtxDomain(domain)
    , myDevice(nullptr)
{
    switch(t)
    {
        case DeviceQueueType::NORMAL:
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream,
                                                 cudaStreamNonBlocking));
            break;

        // These are semantically valid only on device
        #ifdef __CUDA_ARCH__
            case DeviceQueueType::FIRE_AND_FORGET:
                stream = cudaStreamFireAndForget;
                break;
            case DeviceQueueType::TAIL_LAUNCH:
                stream = cudaStreamTailLaunch;
                break;
            default: __trap(); break;
        #endif
    }
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCUDA::GPUQueueCUDA(GPUQueueCUDA&& other) noexcept
    : stream(other.stream)
    , multiprocessorCount(other.multiprocessorCount)
    , nvtxDomain(other.nvtxDomain)
    , myDevice(other.myDevice)
{
    other.stream = cudaStream_t(0);
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCUDA& GPUQueueCUDA::operator=(GPUQueueCUDA&& other) noexcept
{
    multiprocessorCount = other.multiprocessorCount;
    nvtxDomain = other.nvtxDomain;
    stream = other.stream;
    myDevice = other.myDevice;
    other.stream = cudaStream_t(0);
    return *this;
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCUDA::~GPUQueueCUDA()
{
    #ifdef __CUDA_ARCH__
        if(stream != cudaStreamTailLaunch &&
           stream != cudaStreamFireAndForget &&
           stream != cudaStream_t(0))
            CUDA_CHECK(cudaStreamDestroy(stream));
    #else
        if(stream != cudaStream_t(0))
            CUDA_CHECK(cudaStreamDestroy(stream));
    #endif

}

// Memory Movement (Async)
template <class T>
MRAY_HOST
void GPUQueueCUDA::MemcpyAsync(Span<T> regionTo, Span<const T> regionFrom) const
{
    assert(regionTo.size_bytes() >= regionFrom.size_bytes());
    CUDA_CHECK(cudaMemcpyAsync(regionTo.data(), regionFrom.data(),
                               regionFrom.size_bytes(),
                               cudaMemcpyDefault, stream));
}

template <class T>
MRAY_HOST void GPUQueueCUDA::MemcpyAsyncStrided(Span<T> regionTo, size_t outputByteStride,
                                                Span<const T> regionFrom, size_t inputByteStride) const
{
    // TODO: This may have performance implications maybe,
    // test it. We utilize "1" width 2D copy to emulate strided memcpy.
    size_t actualInStride = (inputByteStride == 0) ? sizeof(T) : inputByteStride;
    size_t actualOutStride = (outputByteStride == 0) ? sizeof(T) : outputByteStride;

    size_t elemCountIn = MathFunctions::DivideUp(regionFrom.size_bytes(), actualInStride);
    assert(elemCountIn == MathFunctions::DivideUp(regionTo.size_bytes(), actualOutStride));

    cudaMemcpy2DAsync(regionTo.data(),
                      actualOutStride,
                      regionFrom.data(),
                      actualInStride,
                      sizeof(T), elemCountIn,
                      cudaMemcpyDefault,
                      stream);
}

template <class T>
MRAY_HOST
void GPUQueueCUDA::MemsetAsync(Span<T> region, uint8_t perByteValue) const
{
    // TODO: Check if memory is not pure-host memory
    CUDA_CHECK(cudaMemsetAsync(region.data(), perByteValue,
                               region.size_bytes(), stream));
}

MRAY_HOST inline
void GPUQueueCUDA::IssueBufferForDestruction(TransientData data) const
{
    void* ptr = TransientPoolIssueBufferForDestruction(std::move(data));
    CUDA_CHECK(cudaLaunchHostFunc(stream, &TransientPoolDestroyCallback, ptr));
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCUDA GPUQueueCUDA::Barrier() const
{
    return GPUFenceCUDA(*this);
}

MRAY_HOST inline
void GPUQueueCUDA::IssueSemaphoreWait(GPUSemaphoreCUDA& sem, uint64_t waitValue) const
{
    cudaExternalSemaphoreWaitParams waitParams = {};
    waitParams.params.fence.value = waitValue;
    CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&sem.semCUDA, &waitParams,
                                               1, stream));
}

MRAY_HOST inline
void GPUQueueCUDA::IssueSemaphoreSignal(GPUSemaphoreCUDA& sem, uint64_t signalValue) const
{
    cudaExternalSemaphoreSignalParams signalParams = {};
    signalParams.params.fence.value = signalValue;
    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&sem.semCUDA, &signalParams,
                                                 1, stream));
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t GPUQueueCUDA::SMCount() const
{
    return multiprocessorCount;
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t GPUQueueCUDA::RecommendedBlockCountPerSM(const void* kernelPtr,
                                                  uint32_t threadsPerBlock,
                                                  uint32_t sharedMemSize)
{
    int numBlocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
                                                             kernelPtr,
                                                             static_cast<int>(threadsPerBlock),
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

template <class T>
MRAY_HOST
void GPUSystemCUDA::Memcpy(Span<T> regionTo, Span<const T> regionFrom) const
{
    assert(regionTo.size_bytes() >= regionFrom.size_bytes());
    CUDA_CHECK(cudaMemcpy(regionTo.data(), regionFrom.data(),
                          regionTo.size_bytes(), cudaMemcpyDefault));
}

template <class T>
MRAY_HOST
void GPUSystemCUDA::Memset(Span<T> region, uint8_t perByteValue) const
{
    // TODO: Check if memory is not pure-host memory
    CUDA_CHECK(cudaMemset(region.data(), perByteValue,
                          region.size_bytes()));
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


