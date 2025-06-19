
#pragma once

#include <vector>

#include "Core/Types.h"
#include "Core/Math.h"
#include "DefinitionsCPU.h"

#include "../GPUTypes.h"

class TimelineSemaphore;

// For CPU device these do not makes sense
#define MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(X)

// Simplified version for default configuration
#define MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT \
    MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(StaticThreadPerBlock1D())

#define MRAY_GRID_CONSTANT

#define MRAY_SHARED_MEMORY thread_local

// For CPU targets we make this smaller number
static constexpr uint32_t StaticThreadPerBlock1D()
{
    return 64u;
}

// TODO: This should not be compile time static
static constexpr uint32_t TotalQueuePerDevice()
{
    return 4;
}

namespace mray::host
{

using AnnotationHandle = void*;

class GPUQueueCPU;
class GPUDeviceCPU;

// Semaphore related namespace global functions
void TimelineSemAcquireInternal(void*);
void TimelineSemReleaseInternal(void*);

// Generic Call Parameters
struct KernelCallParamsCPU
{
    uint32_t gridSize;
    uint32_t blockSize;
    uint32_t blockId;
    uint32_t threadId;

    MRAY_GPU                KernelCallParamsCPU();
    MRAY_HYBRID uint32_t    GlobalId() const;
    MRAY_HYBRID uint32_t    TotalSize() const;
};

using AnnotationHandle = void*;
using AnnotationStringHandle = void*;

class GPUAnnotationCPU
{
    public:
    friend class GPUSystemCPU;
    friend class GPUQueueCPU;

    class Scope
    {
        friend GPUAnnotationCPU;

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

    GPUAnnotationCPU(AnnotationHandle, std::string_view name);

    public:
    // Constructors & Destructor
                        GPUAnnotationCPU(const GPUAnnotationCPU&) = delete;
                        GPUAnnotationCPU(GPUAnnotationCPU&&) = delete;
    GPUAnnotationCPU&   operator=(const GPUAnnotationCPU&) = delete;
    GPUAnnotationCPU&   operator=(GPUAnnotationCPU&&) = delete;

    [[nodiscard]]
    Scope               AnnotateScope() const;
};

class GPUSemaphoreViewCPU
{
    private:
    TimelineSemaphore*  externalSemaphore;
    uint64_t            acquireValue;

    public:
    // Constructors & Destructor
                GPUSemaphoreViewCPU(TimelineSemaphore* sem,
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

class GPUFenceCPU
{
    private:
    // TODO: ....

    public:
    MRAY_HYBRID                 GPUFenceCPU(const GPUQueueCPU&);
                                GPUFenceCPU(const GPUFenceCPU&) = delete;
    MRAY_HYBRID                 GPUFenceCPU(GPUFenceCPU&&) noexcept;
    GPUFenceCPU&                operator=(const GPUFenceCPU&) = delete;
    MRAY_HYBRID GPUFenceCPU&    operator=(GPUFenceCPU&&) noexcept;
    MRAY_HYBRID                 ~GPUFenceCPU();

    MRAY_HYBRID void            Wait() const;
};

class GPUQueueCPU
{
    private:
    const GPUDeviceCPU* myDevice            = nullptr;

    MRAY_HYBRID
    uint32_t            DetermineGridStrideBlock(const void* kernelPtr,
                                                 uint32_t sharedMemSize,
                                                 uint32_t threadCount,
                                                 uint32_t workCount) const;

    public:
    // Constructors & Destructor
                                GPUQueueCPU() = default;
    MRAY_HOST                   GPUQueueCPU(uint32_t multiprocessorCount,
                                            AnnotationHandle domain,
                                            const GPUDeviceCPU* device);
    MRAY_GPU                    GPUQueueCPU(uint32_t multiprocessorCount,
                                            AnnotationHandle domain,
                                            DeviceQueueType t);
                                GPUQueueCPU(const GPUQueueCPU&) = delete;
    MRAY_HYBRID                 GPUQueueCPU(GPUQueueCPU&&) noexcept;
    GPUQueueCPU&                operator=(const GPUQueueCPU&) = delete;
    MRAY_HYBRID GPUQueueCPU&    operator=(GPUQueueCPU&&) noexcept;
    MRAY_HYBRID                 ~GPUQueueCPU();

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
    MRAY_HYBRID NO_DISCARD
    GPUFenceCPU         Barrier() const;
    MRAY_HOST
    void                IssueSemaphoreWait(GPUSemaphoreViewCPU&) const;
    MRAY_HOST
    void                IssueSemaphoreSignal(GPUSemaphoreViewCPU&) const;
    MRAY_HOST
    void                IssueWait(const GPUFenceCPU&) const;

    MRAY_HYBRID
    uint32_t            SMCount() const;

    MRAY_HYBRID
    uint32_t            RecommendedBlockCountDevice(const void* kernelPtr,
                                                    uint32_t threadsPerBlock,
                                                    uint32_t sharedMemSize) const;

    MRAY_HYBRID
    static uint32_t     RecommendedBlockCountSM(const void* kernelPtr,
                                                uint32_t threadsPerBlock,
                                                uint32_t sharedMemSize);

    // Annotation for profiling etc. (uses rocTX)
    MRAY_HOST
    GPUAnnotationCPU        CreateAnnotation(std::string_view) const;

    MRAY_HOST
    const GPUDeviceCPU*     Device() const;
};

class GPUDeviceCPU
{
    using DeviceQueues = std::vector<GPUQueueCPU>;

    private:
    DeviceQueues            queues;
    GPUQueueCPU             transferQueue;

    protected:
    public:
    // Constructors & Destructor
    explicit                GPUDeviceCPU(int deviceId, AnnotationHandle);
                            GPUDeviceCPU(const GPUDeviceCPU&) = delete;
                            GPUDeviceCPU(GPUDeviceCPU&&) noexcept = default;
    GPUDeviceCPU&           operator=(const GPUDeviceCPU&) = delete;
    GPUDeviceCPU&           operator=(GPUDeviceCPU&&) noexcept = default;
                            ~GPUDeviceCPU() = default;

    bool                    operator==(const GPUDeviceCPU&) const;

    int                     DeviceId() const;
    std::string             Name() const;
    std::string             ComputeCapability() const;
    size_t                  TotalMemory() const;

    uint32_t                SMCount() const;
    uint32_t                MaxActiveBlockPerSM(uint32_t threadsPerBlock = StaticThreadPerBlock1D()) const;

    const GPUQueueCPU&      GetComputeQueue(uint32_t index) const;
    const GPUQueueCPU&      GetTransferQueue() const;

};

class GPUSystemCPU
{
    public:
    using GPUList = std::vector<GPUDeviceCPU>;
    using GPUPtrList = std::vector<const GPUDeviceCPU*>;

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
                        GPUSystemCPU();
                        GPUSystemCPU(const GPUSystemCPU&) = delete;
                        GPUSystemCPU(GPUSystemCPU&&) = delete;
    GPUSystemCPU&       operator=(const GPUSystemCPU&) = delete;
    GPUSystemCPU&       operator=(GPUSystemCPU&&) = delete;
                        ~GPUSystemCPU();

    // Multi-Device Splittable Smart GPU Calls
    // Automatic device split and stream split on devices
    std::vector<size_t> SplitWorkToMultipleGPU(uint32_t workCount,
                                               uint32_t threadCount,
                                               uint32_t sharedMemSize,
                                               void* f) const;

    // Misc
    const GPUList&          SystemDevices() const;
    const GPUDeviceCPU&     BestDevice() const;
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
    GPUAnnotationCPU        CreateAnnotation(std::string_view) const;
};


MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t KernelCallParamsCPU::GlobalId() const
{
    return blockId * blockSize + threadId;
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t KernelCallParamsCPU::TotalSize() const
{
    return gridSize * blockSize;
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCPU::GPUFenceCPU(const GPUQueueCPU& q)
{
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCPU::GPUFenceCPU(GPUFenceCPU&& other) noexcept
{
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCPU& GPUFenceCPU::operator=(GPUFenceCPU&& other) noexcept
{
    assert(&other != this);
    return *this;

}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCPU::~GPUFenceCPU()
{
}

MRAY_HYBRID MRAY_CGPU_INLINE
void GPUFenceCPU::Wait() const
{

}

MRAY_HOST inline
GPUQueueCPU::GPUQueueCPU(uint32_t multiprocessorCount,
                         AnnotationHandle domain,
                         const GPUDeviceCPU* device)
    : myDevice(device)
{}

MRAY_GPU MRAY_GPU_INLINE
GPUQueueCPU::GPUQueueCPU(uint32_t multiprocessorCount,
                         AnnotationHandle domain,
                         DeviceQueueType)
    : myDevice(nullptr)
{
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCPU::GPUQueueCPU(GPUQueueCPU&& other) noexcept
    : myDevice(other.myDevice)
{}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCPU& GPUQueueCPU::operator=(GPUQueueCPU&& other) noexcept
{
    myDevice = other.myDevice;
    return *this;
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCPU::~GPUQueueCPU()
{
}

// Memory Movement (Async)
template <class T>
MRAY_HOST
void GPUQueueCPU::MemcpyAsync(Span<T> regionTo, Span<const T> regionFrom) const
{
}

template <class T>
MRAY_HOST
void GPUQueueCPU::MemcpyAsync2D(Span<T> regionTo, size_t toStride,
                                Span<const T> regionFrom, size_t fromStride,
                                Vector2ui copySize) const
{
}

template <class T>
MRAY_HOST
void GPUQueueCPU::MemcpyAsyncStrided(Span<T> regionTo, size_t outputByteStride,
                                     Span<const T> regionFrom, size_t inputByteStride) const
{
}

template <class T>
MRAY_HOST
void GPUQueueCPU::MemsetAsync(Span<T> region, uint8_t perByteValue) const
{
}

MRAY_HOST inline
void GPUQueueCPU::IssueBufferForDestruction(TransientData data) const
{
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCPU GPUQueueCPU::Barrier() const
{
    return GPUFenceCPU(*this);
}

MRAY_HOST inline
void GPUQueueCPU::IssueSemaphoreWait(GPUSemaphoreViewCPU& sem) const
{
}

MRAY_HOST inline
void GPUQueueCPU::IssueSemaphoreSignal(GPUSemaphoreViewCPU& sem) const
{
}

MRAY_HOST inline
void GPUQueueCPU::IssueWait(const GPUFenceCPU& barrier) const
{
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t GPUQueueCPU::SMCount() const
{
    return 0u;
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t GPUQueueCPU::RecommendedBlockCountSM(const void* kernelPtr,
                                              uint32_t threadsPerBlock,
                                              uint32_t sharedMemSize)
{
    return StaticThreadPerBlock1D();
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t GPUQueueCPU::RecommendedBlockCountDevice(const void* kernelPtr,
                                                  uint32_t threadsPerBlock,
                                                  uint32_t sharedMemSize) const
{
    return StaticThreadPerBlock1D();
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t GPUQueueCPU::DetermineGridStrideBlock(const void* kernelPtr,
                                               uint32_t sharedMemSize,
                                               uint32_t threadCount,
                                               uint32_t workCount) const
{
    return 1u;
}

MRAY_HOST inline
GPUAnnotationCPU GPUQueueCPU::CreateAnnotation(std::string_view name) const
{
    return GPUAnnotationCPU(nullptr, name);
}

MRAY_HOST inline
const GPUDeviceCPU* GPUQueueCPU::Device() const
{
    return myDevice;
}

template <class T>
MRAY_HOST
void GPUSystemCPU::Memcpy(Span<T> regionTo, Span<const T> regionFrom) const
{
}

template <class T>
MRAY_HOST
void GPUSystemCPU::Memset(Span<T> region, uint8_t perByteValue) const
{
}

}


