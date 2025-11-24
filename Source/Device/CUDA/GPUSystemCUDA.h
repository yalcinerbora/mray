
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "Core/Types.h"
#include "Core/Math.h"
#include "DefinitionsCUDA.h"

#include "../GPUTypes.h"

#include "TransientPool/TransientPool.h"

#ifdef MRAY_ENABLE_TRACY
    namespace tracy { class CUDACtx;}
#endif

class TimelineSemaphore;

// Cuda Kernel Optimization Hints
// Since we call all of the kernels in a static manner
// (in case of Block Size) hint the compiler
// using __launch_bounds__ expression
//
// MSVC Intellisense (realtime compiler) goes insane
// when it sees this macro so providing it as empty
#ifdef __INTELLISENSE__
    #define MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(X)
#else
    #define MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(X) __launch_bounds__(X)
#endif

// Simplified version for default configuration
#define MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT \
            MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(StaticThreadPerBlock1D())


#if __CUDA_ARCH__ >= 700
    #define MRAY_GRID_CONSTANT __grid_constant__
#else
    #define MRAY_GRID_CONSTANT
#endif

#define MRAY_SHARED_MEMORY __shared__

#define MRAY_KERNEL __global__

static constexpr uint32_t WarpSize()
{
    return 32u;
}

template<uint32_t LOGICAL_WARP_SIZE = WarpSize()>
MR_GF_DECL
inline void WarpSynchronize()
{
    // Dirty fix to make host side happy
    #ifdef __CUDA_ARCH__
    static_assert(LOGICAL_WARP_SIZE == 1 || LOGICAL_WARP_SIZE == 2 ||
                  LOGICAL_WARP_SIZE == 4 || LOGICAL_WARP_SIZE == 8 ||
                  LOGICAL_WARP_SIZE == 16 || LOGICAL_WARP_SIZE == 32,
                  "Logical warps must be power of 2 and \"<32\"");

    // Technically single-threaded logical warp is self synchronizing,
    // so no need to do sync.
    if constexpr(LOGICAL_WARP_SIZE != 1)
    {
        static constexpr uint32_t FULL_MASK = std::numeric_limits<uint32_t>::max();
        // Creating all FF's is UB (when doing it via shift, is there any other way to do it)
        // since we shift out of bounds so...
        static constexpr uint32_t MASK = (LOGICAL_WARP_SIZE == 32)
            ? FULL_MASK
            : (1u << LOGICAL_WARP_SIZE) - 1u;
        uint32_t localWarpId = threadIdx.x % WarpSize();
        uint32_t logicalWarpId = localWarpId / LOGICAL_WARP_SIZE;
        uint32_t localMask = MASK << (logicalWarpId * LOGICAL_WARP_SIZE);
        __syncwarp(localMask);
    }
    #endif
}

MR_GF_DECL
inline void BlockSynchronize()
{
    // Dirty fix to make host side happy
    #ifdef __CUDA_ARCH__
    __syncthreads();
    #endif
}

MR_GF_DECL
inline void ThreadFenceGrid()
{
    // Dirty fix to make host side happy
    #ifdef __CUDA_ARCH__
    __threadfence();
    #endif
}

// A Good guess for TPB
constexpr uint32_t StaticThreadPerBlock1D()
{
    return 256u;
}

// TODO: This should not be compile time static
constexpr uint32_t TotalQueuePerDevice()
{
    return 4;
}

namespace mray::cuda
{

using AnnotationHandle = void*;

class GPUQueueCUDA;
class GPUDeviceCUDA;

// Semaphore related namespace global functions
void TimelineSemAcquireInternal(void*);
void TimelineSemReleaseInternal(void*);

// Generic Call Parameters
struct KernelCallParamsCUDA
{
    uint32_t gridSize;
    uint32_t blockSize;
    uint32_t blockId;
    uint32_t threadId;

    MR_PF_DECL_V        KernelCallParamsCUDA();
    MR_PF_DECL uint32_t GlobalId() const;
    MR_PF_DECL uint32_t TotalSize() const;
};

using AnnotationHandle = void*;
using AnnotationStringHandle = void*;

class GPUAnnotationCUDA
{
    public:
    friend class GPUSystemCUDA;
    friend class GPUQueueCUDA;

    class Scope
    {
        friend GPUAnnotationCUDA;

        private:
        AnnotationHandle domain;

        Scope(AnnotationHandle);
        public:
        // Constructors & Destructor
                Scope(const Scope&) = delete;
                Scope(Scope&&) = delete;
        Scope&  operator=(const Scope&) = delete;
        Scope&  operator=(Scope&&) = delete;
                ~Scope();
    };

    private:
    AnnotationHandle        domainHandle;
    AnnotationStringHandle  stringHandle;

    GPUAnnotationCUDA(AnnotationHandle, std::string_view name);

    public:
    // Constructors & Destructor
                        GPUAnnotationCUDA(const GPUAnnotationCUDA&) = delete;
                        GPUAnnotationCUDA(GPUAnnotationCUDA&&) = delete;
    GPUAnnotationCUDA&  operator=(const GPUAnnotationCUDA&) = delete;
    GPUAnnotationCUDA&  operator=(GPUAnnotationCUDA&&) = delete;

    [[nodiscard]]
    Scope               AnnotateScope() const;
};

class GPUSemaphoreViewCUDA
{
    private:
    TimelineSemaphore*  externalSemaphore;
    uint64_t            acquireValue;

    public:
    // Constructors & Destructor
                GPUSemaphoreViewCUDA(TimelineSemaphore* sem,
                                     uint64_t acqValue);

    // Change to next acquisition state (acquireValue + 1)
    // we will not send anything to visor.
    // This is used to acquire the image for memory realloc
    void        SkipAState();
    // Change to next acquisition state (acquireValue + 2)
    // return the Visor's wait state (acquireValue + 1)
    uint64_t    ChangeToNextState();

    [[nodiscard]]
    bool        HostAcquire();
    void        HostRelease();
};

class GPUFenceCUDA
{
    MRAY_HYBRID
    friend cudaEvent_t ToHandleCUDA(const GPUFenceCUDA&);

    private:
    cudaEvent_t                 eventC;

    public:
    MR_HF_DECL                  GPUFenceCUDA(const GPUQueueCUDA&);
                                GPUFenceCUDA(const GPUFenceCUDA&) = delete;
    MR_HF_DECL                  GPUFenceCUDA(GPUFenceCUDA&&) noexcept;
    GPUFenceCUDA&               operator=(const GPUFenceCUDA&) = delete;
    MR_HF_DECL GPUFenceCUDA&    operator=(GPUFenceCUDA&&) noexcept;
    MR_HF_DECL                  ~GPUFenceCUDA();

    MR_HF_DECL void             Wait() const;
};

class GPUQueueCUDA
{
    MRAY_HYBRID
    friend cudaStream_t ToHandleCUDA(const GPUQueueCUDA&);

    private:
    cudaStream_t            stream              = nullptr;
    uint32_t                multiprocessorCount = 0;
    AnnotationHandle        nvtxDomain          = nullptr;
    const GPUDeviceCUDA*    myDevice            = nullptr;

    MR_HF_DECL
    uint32_t            DetermineGridStrideBlock(const void* kernelPtr,
                                                 uint32_t sharedMemSize,
                                                 uint32_t threadCount,
                                                 uint32_t workCount) const;

    MR_HF_DECL
    uint32_t            RecommendedBlockCountDevice(const void* kernelPtr,
                                                    uint32_t threadsPerBlock,
                                                    uint32_t sharedMemSize) const;

    MR_HF_DECL
    static uint32_t     RecommendedBlockCountSM(const void* kernelPtr,
                                                uint32_t threadsPerBlock,
                                                uint32_t sharedMemSize);

    public:
    // Constructors & Destructor
                                GPUQueueCUDA() = default;
    MRAY_HOST                   GPUQueueCUDA(uint32_t multiprocessorCount,
                                             AnnotationHandle domain,
                                             const GPUDeviceCUDA* device);
    MR_GF_DECL                  GPUQueueCUDA(uint32_t multiprocessorCount,
                                             AnnotationHandle domain,
                                             DeviceQueueType t);
                                GPUQueueCUDA(const GPUQueueCUDA&) = delete;
    MR_HF_DECL                  GPUQueueCUDA(GPUQueueCUDA&&) noexcept;
    GPUQueueCUDA&               operator=(const GPUQueueCUDA&) = delete;
    MR_HF_DECL GPUQueueCUDA&    operator=(GPUQueueCUDA&&) noexcept;
    MR_HF_DECL                  ~GPUQueueCUDA();

    // Grid-Stride Kernels
    // Kernel is launched just enough blocks to
    // fully saturate the GPU.
    template<auto Kernel, class... Args>
    MRAY_HOST void  IssueWorkKernel(std::string_view name,
                                    DeviceWorkIssueParams,
                                    //
                                    Args&&...) const;
    template<class Lambda>
    MRAY_HOST void  IssueWorkLambda(std::string_view name,
                                    DeviceWorkIssueParams,
                                    //
                                    Lambda&&) const;
    // Block-Stride Kernels
    // You 1-1 specify block and grid dimensions
    // Kernel is launched with just enough blocks
    // to fully saturate the GPU.
    template<auto Kernel, class... Args>
    MRAY_HOST void  IssueBlockKernel(std::string_view name,
                                     DeviceBlockIssueParams,
                                     //
                                     Args&&...) const;
    template<class Lambda, uint32_t Bounds = StaticThreadPerBlock1D()>
    MRAY_HOST void  IssueBlockLambda(std::string_view name,
                                     DeviceBlockIssueParams,
                                     //
                                     Lambda&&) const;

    // Split the device side kernel calls. glibc++ span has
    // explicitly defined defaulted destructor and NVCC errors
    // because of that even if we don't call the kernel from the
    // device.
    template<auto Kernel, class... Args>
    MR_GF_DECL void DeviceIssueWorkKernel(std::string_view name,
                                          DeviceWorkIssueParams,
                                          //
                                          Args&&...) const;
    template<class Lambda>
    MR_GF_DECL void DeviceIssueWorkLambda(std::string_view name,
                                          DeviceWorkIssueParams,
                                          //
                                          Lambda&&) const;
    template<auto Kernel, class... Args>
    MR_GF_DECL void DeviceIssueBlockKernel(std::string_view name,
                                           DeviceBlockIssueParams,
                                           //
                                           Args&&...) const;
    template<class Lambda, uint32_t Bounds = StaticThreadPerBlock1D()>
    MR_GF_DECL void DeviceIssueBlockLambda(std::string_view name,
                                           DeviceBlockIssueParams,
                                           //
                                           Lambda&&) const;

    // Memory Movement (Async)
    template <class T>
    MRAY_HOST void      MemcpyAsync(Span<T> regionTo, Span<const T> regionFrom) const;
    template <class T>
    MRAY_HOST void      MemcpyAsync2D(Span<T> regionTo, size_t toStride,
                                      Span<const T> regionFrom, size_t fromStride,
                                      Vector2ui copySize) const;
    template <class T>
    MRAY_HOST void      MemcpyAsyncStrided(Span<T> regionTo, size_t outputByteStride,
                                           Span<const T> regionFrom, size_t inputByteStride) const;
    template <class T>
    MRAY_HOST void      MemsetAsync(Span<T> region, uint8_t perByteValue) const;

    // Misc
    MRAY_HOST void      IssueBufferForDestruction(TransientData data) const;

    // Synchronization
    MR_HF_DECL [[nodiscard]]
    GPUFenceCUDA        Barrier() const;
    MRAY_HOST
    void                IssueSemaphoreWait(GPUSemaphoreViewCUDA&) const;
    MRAY_HOST
    void                IssueSemaphoreSignal(GPUSemaphoreViewCUDA&) const;
    MRAY_HOST
    void                IssueWait(const GPUFenceCUDA&) const;

    MRAY_HYBRID
    uint32_t            SMCount() const;

    public:
    // Annotation for profiling etc. (uses NVTX)
    MRAY_HOST
    GPUAnnotationCUDA       CreateAnnotation(std::string_view) const;

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
    GPUQueueCUDA            transferQueue;

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

    const GPUQueueCUDA&     GetComputeQueue(uint32_t index) const;
    const GPUQueueCUDA&     GetTransferQueue() const;

};

class GPUSystemCUDA
{
    public:
    using GPUList = std::vector<GPUDeviceCUDA>;
    using GPUPtrList = std::vector<const GPUDeviceCUDA*>;

    #ifdef MRAY_ENABLE_TRACY
       tracy::CUDACtx*  tracyCUDACtx;
    #endif

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
                        GPUSystemCUDA(bool logBanner = false);
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

    // Annotation for profiling etc. (uses NVTX)
    GPUAnnotationCUDA       CreateAnnotation(std::string_view) const;
};


MR_PF_DEF
uint32_t KernelCallParamsCUDA::GlobalId() const
{
    return blockId * blockSize + threadId;
}

MR_PF_DEF
uint32_t KernelCallParamsCUDA::TotalSize() const
{
    return gridSize * blockSize;
}

MR_HF_DEF
GPUFenceCUDA::GPUFenceCUDA(const GPUQueueCUDA& q)
    : eventC(cudaEvent_t(0))
{
    CUDA_CHECK(cudaEventCreateWithFlags(&eventC, cudaEventDisableTiming));
    cudaStream_t stream = ToHandleCUDA(q);
    CUDA_CHECK(cudaEventRecord(eventC, stream));
}

MR_HF_DEF
GPUFenceCUDA::GPUFenceCUDA(GPUFenceCUDA&& other) noexcept
    : eventC(other.eventC)
{
    other.eventC = cudaEvent_t(0);
}

MR_HF_DEF
GPUFenceCUDA& GPUFenceCUDA::operator=(GPUFenceCUDA&& other) noexcept
{
    eventC = other.eventC;
    other.eventC = cudaEvent_t(0);
    return *this;
}

MR_HF_DEF
GPUFenceCUDA::~GPUFenceCUDA()
{
    if(eventC != cudaEvent_t(0))
        CUDA_CHECK(cudaEventDestroy(eventC));
}

MR_HF_DEF
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

MR_GF_DEF
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
        #else
            default: assert(false); break;
        #endif
    }
}

MR_HF_DEF
GPUQueueCUDA::GPUQueueCUDA(GPUQueueCUDA&& other) noexcept
    : stream(other.stream)
    , multiprocessorCount(other.multiprocessorCount)
    , nvtxDomain(other.nvtxDomain)
    , myDevice(other.myDevice)
{
    other.stream = cudaStream_t(0);
}

MR_HF_DEF
GPUQueueCUDA& GPUQueueCUDA::operator=(GPUQueueCUDA&& other) noexcept
{
    multiprocessorCount = other.multiprocessorCount;
    nvtxDomain = other.nvtxDomain;
    stream = other.stream;
    myDevice = other.myDevice;
    other.stream = cudaStream_t(0);
    return *this;
}

MR_HF_DEF
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
MRAY_HOST inline
void GPUQueueCUDA::MemcpyAsync(Span<T> regionTo, Span<const T> regionFrom) const
{
    assert(regionTo.size_bytes() >= regionFrom.size_bytes());
    CUDA_CHECK(cudaMemcpyAsync(regionTo.data(), regionFrom.data(),
                               regionFrom.size_bytes(),
                               cudaMemcpyDefault, stream));
}

template <class T>
MRAY_HOST inline
void GPUQueueCUDA::MemcpyAsync2D(Span<T> regionTo, size_t toStride,
                                 Span<const T> regionFrom, size_t fromStride,
                                 Vector2ui copySize) const
{
    assert(toStride * (copySize[1] - 1) + copySize[0] <=
           regionTo.size());
    assert(fromStride * (copySize[1] - 1) + copySize[0] <=
           regionFrom.size());
    assert(toStride >= copySize[0]);
    assert(fromStride >= copySize[0]);

    size_t inStrideBytes = toStride * sizeof(T);
    size_t outStrideBytes = fromStride * sizeof(T);
    size_t copyWidthBytes = copySize[0] * sizeof(T);

    cudaMemcpy2DAsync(regionTo.data(),
                      inStrideBytes,
                      regionFrom.data(),
                      outStrideBytes,
                      copyWidthBytes, copySize[1],
                      cudaMemcpyDefault,
                      stream);
}

template <class T>
MRAY_HOST inline
void GPUQueueCUDA::MemcpyAsyncStrided(Span<T> regionTo, size_t outputByteStride,
                                      Span<const T> regionFrom, size_t inputByteStride) const
{
    // TODO: This may have performance implications maybe,
    // test it. We utilize "1" width 2D copy to emulate strided memcpy.
    size_t actualInStride = (inputByteStride == 0) ? sizeof(T) : inputByteStride;
    size_t actualOutStride = (outputByteStride == 0) ? sizeof(T) : outputByteStride;

    size_t elemCountIn = Math::DivideUp(size_t(regionFrom.size_bytes()), actualInStride);
    assert(elemCountIn == Math::DivideUp(size_t(regionTo.size_bytes()), actualOutStride));

    cudaMemcpy2DAsync(regionTo.data(),
                      actualOutStride,
                      regionFrom.data(),
                      actualInStride,
                      sizeof(T), elemCountIn,
                      cudaMemcpyDefault,
                      stream);
}

template <class T>
MRAY_HOST inline
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

MR_HF_DEF
GPUFenceCUDA GPUQueueCUDA::Barrier() const
{
    return GPUFenceCUDA(*this);
}

MRAY_HOST inline
void GPUQueueCUDA::IssueSemaphoreWait(GPUSemaphoreViewCUDA& sem) const
{
    CUDA_CHECK(cudaLaunchHostFunc(stream, &TimelineSemAcquireInternal, &sem));
}

MRAY_HOST inline
void GPUQueueCUDA::IssueSemaphoreSignal(GPUSemaphoreViewCUDA& sem) const
{
    CUDA_CHECK(cudaLaunchHostFunc(stream, &TimelineSemReleaseInternal, &sem));
}

MRAY_HOST inline
void GPUQueueCUDA::IssueWait(const GPUFenceCUDA& barrier) const
{
    CUDA_CHECK(cudaStreamWaitEvent(stream, ToHandleCUDA(barrier),
                                   cudaEventWaitDefault));
}

MR_HF_DEF
uint32_t GPUQueueCUDA::SMCount() const
{
    return multiprocessorCount;
}

MR_HF_DEF
uint32_t GPUQueueCUDA::RecommendedBlockCountSM(const void* kernelPtr,
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

MR_HF_DEF
uint32_t GPUQueueCUDA::RecommendedBlockCountDevice(const void* kernelPtr,
                                                   uint32_t threadsPerBlock,
                                                   uint32_t sharedMemSize) const
{
    uint32_t blockPerSM = RecommendedBlockCountSM(kernelPtr, threadsPerBlock,
                                                  sharedMemSize);
    return multiprocessorCount * blockPerSM;
}

MR_HF_DEF
uint32_t GPUQueueCUDA::DetermineGridStrideBlock(const void* kernelPtr,
                                                uint32_t sharedMemSize,
                                                uint32_t threadCount,
                                                uint32_t workCount) const
{
    // TODO: Make better SM determination
    uint32_t blockPerSM = RecommendedBlockCountSM(kernelPtr, threadCount, sharedMemSize);
    // Only call enough SM
    uint32_t totalRequiredBlocks = Math::DivideUp(workCount, threadCount);
    uint32_t requiredSMCount = Math::DivideUp(totalRequiredBlocks, blockPerSM);
    uint32_t smCount = Math::Min(multiprocessorCount, requiredSMCount);
    uint32_t blockCount = smCount * blockPerSM;
    return blockCount;
}

MRAY_HOST inline
GPUAnnotationCUDA GPUQueueCUDA::CreateAnnotation(std::string_view name) const
{
    return GPUAnnotationCUDA(nvtxDomain, name);
}

MRAY_HOST inline
const GPUDeviceCUDA* GPUQueueCUDA::Device() const
{
    return myDevice;
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

MR_HF_DEF
cudaStream_t ToHandleCUDA(const GPUQueueCUDA& q)
{
    return q.stream;
}

MR_HF_DEF
cudaEvent_t ToHandleCUDA(const GPUFenceCUDA& f)
{
    return f.eventC;
}

}


