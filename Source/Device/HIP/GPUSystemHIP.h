
#pragma once

#include <hip/hip_runtime.h>
#include <vector>

#include "Core/Types.h"
#include "Core/Math.h"
#include "DefinitionsHIP.h"

#include "../GPUTypes.h"

#include "TransientPool/TransientPool.h"

class TimelineSemaphore;

// Hip Kernel Optimization Hints
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


// TODO: Check this for rocm (all versions support this?)
#define MRAY_GRID_CONSTANT

#define MRAY_SHARED_MEMORY __shared__

#define MRAY_KERNEL __global__

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

namespace mray::hip
{

using AnnotationHandle = void*;

class GPUQueueHIP;
class GPUDeviceHIP;

// Semaphore related namespace global functions
void TimelineSemAcquireInternal(void*);
void TimelineSemReleaseInternal(void*);

// Generic Call Parameters
struct KernelCallParamsHIP
{
    uint32_t gridSize;
    uint32_t blockSize;
    uint32_t blockId;
    uint32_t threadId;

    MRAY_GPU            KernelCallParamsHIP();
    MR_PF_DECL uint32_t GlobalId() const;
    MR_PF_DECL uint32_t TotalSize() const;
};

using AnnotationHandle = void*;
using AnnotationStringHandle = void*;

class GPUAnnotationHIP
{
    public:
    friend class GPUSystemHIP;
    friend class GPUQueueHIP;

    class Scope
    {
        friend GPUAnnotationHIP;

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

    GPUAnnotationHIP(AnnotationHandle, std::string_view name);

    public:
    // Constructors & Destructor
                        GPUAnnotationHIP(const GPUAnnotationHIP&) = delete;
                        GPUAnnotationHIP(GPUAnnotationHIP&&) = delete;
    GPUAnnotationHIP&   operator=(const GPUAnnotationHIP&) = delete;
    GPUAnnotationHIP&   operator=(GPUAnnotationHIP&&) = delete;

    [[nodiscard]]
    Scope               AnnotateScope() const;
};

class GPUSemaphoreViewHIP
{
    private:
    TimelineSemaphore*  externalSemaphore;
    uint64_t            acquireValue;

    public:
    // Constructors & Destructor
                GPUSemaphoreViewHIP(TimelineSemaphore* sem,
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

class GPUFenceHIP
{
    MRAY_HYBRID
    friend hipEvent_t           ToHandleHIP(const GPUFenceHIP&);

    private:
    hipEvent_t                  eventC;

    public:
    MR_HF_DECL                 GPUFenceHIP(const GPUQueueHIP&);
                               GPUFenceHIP(const GPUFenceHIP&) = delete;
    MR_HF_DECL                 GPUFenceHIP(GPUFenceHIP&&) noexcept;
    GPUFenceHIP&               operator=(const GPUFenceHIP&) = delete;
    MR_HF_DECL GPUFenceHIP&    operator=(GPUFenceHIP&&) noexcept;
    MR_HF_DECL                 ~GPUFenceHIP();

    MR_HF_DECL void            Wait() const;
};

class GPUQueueHIP
{
    MRAY_HYBRID
    friend hipStream_t      ToHandleHIP(const GPUQueueHIP&);

    private:
    hipStream_t             stream              = nullptr;
    uint32_t                multiprocessorCount = 0;
    AnnotationHandle        roctxDomain         = nullptr;
    const GPUDeviceHIP*     myDevice            = nullptr;

    MR_HF_DECL
    uint32_t            DetermineGridStrideBlock(const void* kernelPtr,
                                                 uint32_t sharedMemSize,
                                                 uint32_t threadCount,
                                                 uint32_t workCount) const;

    public:
    // Constructors & Destructor
                                GPUQueueHIP() = default;
    MRAY_HOST                   GPUQueueHIP(uint32_t multiprocessorCount,
                                            AnnotationHandle domain,
                                            const GPUDeviceHIP* device);
    MR_GF_DECL                  GPUQueueHIP(uint32_t multiprocessorCount,
                                            AnnotationHandle domain,
                                            DeviceQueueType t);
                                GPUQueueHIP(const GPUQueueHIP&) = delete;
    MR_HF_DECL                  GPUQueueHIP(GPUQueueHIP&&) noexcept;
    GPUQueueHIP&                operator=(const GPUQueueHIP&) = delete;
    MR_HF_DECL GPUQueueHIP&     operator=(GPUQueueHIP&&) noexcept;
    MR_HF_DECL                  ~GPUQueueHIP();


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
    // Exact Kernel Calls
    // You 1-1 specify block and grid dimensions
    // Important: These can not be annotated with launch_bounds
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
    // because of that even if we dont call the kernel from the
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
    MR_HF_DECL NO_DISCARD
    GPUFenceHIP         Barrier() const;
    MRAY_HOST
    void                IssueSemaphoreWait(GPUSemaphoreViewHIP&) const;
    MRAY_HOST
    void                IssueSemaphoreSignal(GPUSemaphoreViewHIP&) const;
    MRAY_HOST
    void                IssueWait(const GPUFenceHIP&) const;

    MRAY_HYBRID
    uint32_t            SMCount() const;

    MR_HF_DECL
    uint32_t            RecommendedBlockCountDevice(const void* kernelPtr,
                                                    uint32_t threadsPerBlock,
                                                    uint32_t sharedMemSize) const;

    MR_HF_DECL
    static uint32_t     RecommendedBlockCountSM(const void* kernelPtr,
                                                uint32_t threadsPerBlock,
                                                uint32_t sharedMemSize);

    // Annotation for profiling etc. (uses rocTX)
    MRAY_HOST
    GPUAnnotationHIP        CreateAnnotation(std::string_view) const;

    MRAY_HOST
    const GPUDeviceHIP*     Device() const;
};

class GPUDeviceHIP
{
    using DeviceQueues = std::vector<GPUQueueHIP>;

    private:
    int                     deviceId;
    hipDeviceProp_t         props;
    DeviceQueues            queues;
    GPUQueueHIP             transferQueue;

    protected:
    public:
    // Constructors & Destructor
    explicit                GPUDeviceHIP(int deviceId, AnnotationHandle);
                            GPUDeviceHIP(const GPUDeviceHIP&) = delete;
                            GPUDeviceHIP(GPUDeviceHIP&&) noexcept = default;
    GPUDeviceHIP&           operator=(const GPUDeviceHIP&) = delete;
    GPUDeviceHIP&           operator=(GPUDeviceHIP&&) noexcept = default;
                            ~GPUDeviceHIP() = default;

    bool                    operator==(const GPUDeviceHIP&) const;

    int                     DeviceId() const;
    std::string             Name() const;
    std::string             ComputeCapability() const;
    size_t                  TotalMemory() const;

    uint32_t                SMCount() const;
    uint32_t                MaxActiveBlockPerSM(uint32_t threadsPerBlock = StaticThreadPerBlock1D()) const;

    const GPUQueueHIP&      GetComputeQueue(uint32_t index) const;
    const GPUQueueHIP&      GetTransferQueue() const;

};

class GPUSystemHIP
{
    public:
    using GPUList = std::vector<GPUDeviceHIP>;
    using GPUPtrList = std::vector<const GPUDeviceHIP*>;

    private:
    GPUList             systemGPUs;
    GPUPtrList          systemGPUPtrs;
    AnnotationHandle    roctxDomain;

    // TODO: Check designs for this, this made the GPUSystem global
    // which is fine
    static GPUList*     globalGPUListPtr;
    static void         ThreadInitFunction();

    protected:
    public:
    // Constructors & Destructor
                        GPUSystemHIP(bool logBanner = false);
                        GPUSystemHIP(const GPUSystemHIP&) = delete;
                        GPUSystemHIP(GPUSystemHIP&&) = delete;
    GPUSystemHIP&       operator=(const GPUSystemHIP&) = delete;
    GPUSystemHIP&       operator=(GPUSystemHIP&&) = delete;
                        ~GPUSystemHIP();

    // Multi-Device Splittable Smart GPU Calls
    // Automatic device split and stream split on devices
    std::vector<size_t> SplitWorkToMultipleGPU(uint32_t workCount,
                                               uint32_t threadCount,
                                               uint32_t sharedMemSize,
                                               void* f) const;

    // Misc
    const GPUList&          SystemDevices() const;
    const GPUDeviceHIP&     BestDevice() const;
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

    // Annotation for profiling etc. (uses rocTX)
    GPUAnnotationHIP        CreateAnnotation(std::string_view) const;
};


MR_PF_DEF
uint32_t KernelCallParamsHIP::GlobalId() const
{
    return blockId * blockSize + threadId;
}

MR_PF_DEF
uint32_t KernelCallParamsHIP::TotalSize() const
{
    return gridSize * blockSize;
}

MR_HF_DEF
GPUFenceHIP::GPUFenceHIP(const GPUQueueHIP& q)
    : eventC(hipEvent_t(0))
{
    HIP_CHECK(hipEventCreateWithFlags(&eventC, hipEventDisableTiming));
    hipStream_t stream = ToHandleHIP(q);
    HIP_CHECK(hipEventRecord(eventC, stream));
}

MR_HF_DEF
GPUFenceHIP::GPUFenceHIP(GPUFenceHIP&& other) noexcept
    : eventC(other.eventC)
{
    other.eventC = hipEvent_t(0);
}

MR_HF_DEF
GPUFenceHIP& GPUFenceHIP::operator=(GPUFenceHIP&& other) noexcept
{
    eventC = other.eventC;
    other.eventC = hipEvent_t(0);
    return *this;
}

MR_HF_DEF
GPUFenceHIP::~GPUFenceHIP()
{
    if(eventC != hipEvent_t(0))
        HIP_CHECK(hipEventDestroy(eventC));
}

MR_HF_DEF
void GPUFenceHIP::Wait() const
{
    #ifndef __HIP_DEVICE_COMPILE__
        HIP_CHECK(hipEventSynchronize(eventC));
    #else
        // TODO: Reason about this
        //
        // HIP_CHECK(hipStreamWaitEvent(stream, event));
        // TODO: hip does not have breakpoint or trap
        // directly aborting
        abort();
    #endif
}

MRAY_HOST inline
GPUQueueHIP::GPUQueueHIP(uint32_t multiprocessorCount,
                         AnnotationHandle domain,
                         const GPUDeviceHIP* device)
    : multiprocessorCount(multiprocessorCount)
    , roctxDomain(domain)
    , myDevice(device)
{
    HIP_CHECK(hipStreamCreateWithFlags(&stream,
                                       hipStreamNonBlocking));
}

MR_GF_DEF
GPUQueueHIP::GPUQueueHIP(uint32_t multiprocessorCount,
                         AnnotationHandle domain,
                         DeviceQueueType)
    : multiprocessorCount(multiprocessorCount)
    , roctxDomain(domain)
    , myDevice(nullptr)
{
    // TODO: Hip do not support device queues?
    // Or at least hipStreamFireAndForget etc.
    // switch(t)
    // {
    //     case DeviceQueueType::NORMAL:
    //         HIP_CHECK(hipStreamCreateWithFlags(&stream,
    //                                            hipStreamNonBlocking));
    //         break;

    //     // These are semantically valid only on device
    //     #ifdef __HIP_DEVICE_COMPILE__
    //         case DeviceQueueType::FIRE_AND_FORGET:
    //             stream = hipStreamFireAndForget;
    //             break;
    //         case DeviceQueueType::TAIL_LAUNCH:
    //             stream = hipStreamTailLaunch;
    //             break;
    //         default: __trap(); break;
    //     #else
    //         default: assert(false); break;
    //     #endif
    // }
}

MR_HF_DEF
GPUQueueHIP::GPUQueueHIP(GPUQueueHIP&& other) noexcept
    : stream(other.stream)
    , multiprocessorCount(other.multiprocessorCount)
    , roctxDomain(other.roctxDomain)
    , myDevice(other.myDevice)
{
    other.stream = hipStream_t(0);
}

MR_HF_DEF
GPUQueueHIP& GPUQueueHIP::operator=(GPUQueueHIP&& other) noexcept
{
    multiprocessorCount = other.multiprocessorCount;
    roctxDomain = other.roctxDomain;
    stream = other.stream;
    myDevice = other.myDevice;
    other.stream = hipStream_t(0);
    return *this;
}

MR_HF_DEF
GPUQueueHIP::~GPUQueueHIP()
{

    // TODO: Hip do not support device queues?
    // #ifdef __HIP_DEVICE_COMPILE__
    //     if(stream != hipStreamTailLaunch &&
    //        stream != hipStreamFireAndForget &&
    //        stream != hipStream_t(0))
    //         HIP_CHECK(hipStreamDestroy(stream));
    // #else
        if(stream != hipStream_t(0))
            HIP_CHECK(hipStreamDestroy(stream));
    // #endif
}

// Memory Movement (Async)
template <class T>
MRAY_HOST
void GPUQueueHIP::MemcpyAsync(Span<T> regionTo, Span<const T> regionFrom) const
{
    assert(regionTo.size_bytes() >= regionFrom.size_bytes());
    HIP_CHECK(hipMemcpyAsync(regionTo.data(), regionFrom.data(),
                             regionFrom.size_bytes(),
                             hipMemcpyDefault, stream));
}

template <class T>
MRAY_HOST
void GPUQueueHIP::MemcpyAsync2D(Span<T> regionTo, size_t toStride,
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

    hipMemcpy2DAsync(regionTo.data(),
                     inStrideBytes,
                     regionFrom.data(),
                     outStrideBytes,
                     copyWidthBytes, copySize[1],
                     hipMemcpyDefault,
                     stream);
}

template <class T>
MRAY_HOST
void GPUQueueHIP::MemcpyAsyncStrided(Span<T> regionTo, size_t outputByteStride,
                                     Span<const T> regionFrom, size_t inputByteStride) const
{
    // TODO: This may have performance implications maybe,
    // test it. We utilize "1" width 2D copy to emulate strided memcpy.
    size_t actualInStride = (inputByteStride == 0) ? sizeof(T) : inputByteStride;
    size_t actualOutStride = (outputByteStride == 0) ? sizeof(T) : outputByteStride;

    size_t elemCountIn = Math::DivideUp(regionFrom.size_bytes(), actualInStride);
    assert(elemCountIn == Math::DivideUp(regionTo.size_bytes(), actualOutStride));

    hipMemcpy2DAsync(regionTo.data(),
                     actualOutStride,
                     regionFrom.data(),
                     actualInStride,
                     sizeof(T), elemCountIn,
                     hipMemcpyDefault,
                     stream);
}

template <class T>
MRAY_HOST
void GPUQueueHIP::MemsetAsync(Span<T> region, uint8_t perByteValue) const
{
    // TODO: Check if memory is not pure-host memory
    HIP_CHECK(hipMemsetAsync(region.data(), perByteValue,
                               region.size_bytes(), stream));
}

MRAY_HOST inline
void GPUQueueHIP::IssueBufferForDestruction(TransientData data) const
{
    void* ptr = TransientPoolIssueBufferForDestruction(std::move(data));
    HIP_CHECK(hipLaunchHostFunc(stream, &TransientPoolDestroyCallback, ptr));
}

MR_HF_DEF
GPUFenceHIP GPUQueueHIP::Barrier() const
{
    return GPUFenceHIP(*this);
}

MRAY_HOST inline
void GPUQueueHIP::IssueSemaphoreWait(GPUSemaphoreViewHIP& sem) const
{
    HIP_CHECK(hipLaunchHostFunc(stream, &TimelineSemAcquireInternal, &sem));
}

MRAY_HOST inline
void GPUQueueHIP::IssueSemaphoreSignal(GPUSemaphoreViewHIP& sem) const
{
    HIP_CHECK(hipLaunchHostFunc(stream, &TimelineSemReleaseInternal, &sem));
}

MRAY_HOST inline
void GPUQueueHIP::IssueWait(const GPUFenceHIP& barrier) const
{
    HIP_CHECK(hipStreamWaitEvent(stream, ToHandleHIP(barrier)));
}

MR_HF_DEF
uint32_t GPUQueueHIP::SMCount() const
{
    return multiprocessorCount;
}

MR_HF_DEF
uint32_t GPUQueueHIP::RecommendedBlockCountSM(const void* kernelPtr,
                                              uint32_t threadsPerBlock,
                                              uint32_t sharedMemSize)
{
    int numBlocks = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
                                                           kernelPtr,
                                                           static_cast<int>(threadsPerBlock),
                                                           sharedMemSize));
    return static_cast<uint32_t>(numBlocks);
}

MR_HF_DEF
uint32_t GPUQueueHIP::RecommendedBlockCountDevice(const void* kernelPtr,
                                                  uint32_t threadsPerBlock,
                                                  uint32_t sharedMemSize) const
{
    uint32_t blockPerSM = RecommendedBlockCountSM(kernelPtr, threadsPerBlock,
                                                  sharedMemSize);
    return multiprocessorCount* blockPerSM;
}

MR_HF_DEF
uint32_t GPUQueueHIP::DetermineGridStrideBlock(const void* kernelPtr,
                                               uint32_t sharedMemSize,
                                               uint32_t threadCount,
                                               uint32_t workCount) const
{
    // TODO: Make better SM determination
    uint32_t blockPerSM = RecommendedBlockCountSM(kernelPtr, threadCount, sharedMemSize);
    // Only call enough SM
    uint32_t totalRequiredBlocks = Math::DivideUp(workCount, threadCount);
    uint32_t requiredSMCount = Math::DivideUp(totalRequiredBlocks, blockPerSM);
    uint32_t smCount = std::min(multiprocessorCount, requiredSMCount);
    uint32_t blockCount = std::min(requiredSMCount, smCount * blockPerSM);
    return blockCount;
}

MRAY_HOST inline
GPUAnnotationHIP GPUQueueHIP::CreateAnnotation(std::string_view name) const
{
    return GPUAnnotationHIP(roctxDomain, name);
}

MRAY_HOST inline
const GPUDeviceHIP* GPUQueueHIP::Device() const
{
    return myDevice;
}

template <class T>
MRAY_HOST
void GPUSystemHIP::Memcpy(Span<T> regionTo, Span<const T> regionFrom) const
{
    assert(regionTo.size_bytes() >= regionFrom.size_bytes());
    HIP_CHECK(hipMemcpy(regionTo.data(), regionFrom.data(),
                          regionTo.size_bytes(), hipMemcpyDefault));
}

template <class T>
MRAY_HOST
void GPUSystemHIP::Memset(Span<T> region, uint8_t perByteValue) const
{
    // TODO: Check if memory is not pure-host memory
    HIP_CHECK(hipMemset(region.data(), perByteValue,
                        region.size_bytes()));
}

MR_HF_DEF
hipStream_t ToHandleHIP(const GPUQueueHIP& q)
{
    return q.stream;
}

MR_HF_DEF
hipEvent_t ToHandleHIP(const GPUFenceHIP& f)
{
    return f.eventC;
}

}


