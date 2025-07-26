
#pragma once
// IWYU pragma: private; include "GPUSystem.h"

#include <vector>
#include <mutex>

#include "Core/Types.h"
#include "Core/Math.h"
#include "Core/MemAlloc.h"
#include "Core/Profiling.h"

#include "TransientPool/TransientPool.h"

#include "../GPUTypes.h"

#include "AlgForwardCPU.h"

class TimelineSemaphore;
class ThreadPool;

// For CPU device these do not makes sense
#define MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(X)

// Simplified version for default configuration
#define MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT \
    MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(StaticThreadPerBlock1D())

#define MRAY_GRID_CONSTANT

// For CPU devices shared memory is static.
// "thread_local" would be redundant here
// since each block has a single thread
#define MRAY_SHARED_MEMORY static thread_local

#define MRAY_KERNEL MRAY_FORCE_INLINE

static constexpr uint32_t StaticThreadPerBlock1D()
{
    // After manually adjusting this in "Kitchen"
    // scene, it seems 256 is optimal.
    // Interestingly it is one of the optimal paramaters
    // (generic-optimal) for the GPU as well.
    //
    // Tested on asymmetric 20 thread laptop CPU (i7-12700H).
    return 256u;
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

struct KernelCallParamsGlobal
{
    uint32_t gridSize   = 0;
    uint32_t blockSize  = 0;
    uint32_t blockId    = 0;
    uint32_t threadId   = 0;
};

// Global list of KP
inline thread_local KernelCallParamsGlobal globalKCParams;

class GPUAnnotationCPU : public ProfilerAnnotation
{
    using SourceLoc = std::source_location;
    using Base = ProfilerAnnotation;

    public:
    friend class GPUSystemCPU;
    friend class GPUQueueCPU;

    class Scope : public ProfilerAnnotation::Scope
    {
        friend GPUAnnotationCPU;
        using Base = ProfilerAnnotation::Scope;
        using Base::Base;
    };

    private:
    GPUAnnotationCPU(AnnotationHandle, std::string_view name,
                     const SourceLoc& s = SourceLoc::current());

    public:
    [[nodiscard]]
    Scope               AnnotateScope() const;
};

void AtomicWaitValueToAchieve(std::atomic_uint64_t& checkLoc, uint64_t valToAchieve);

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
    uint64_t                valueToWait;
    std::atomic_uint64_t*   completedKernelCounter;

    friend class GPUQueueCPU;

    public:
    MRAY_HYBRID                 GPUFenceCPU(const GPUQueueCPU&);
                                GPUFenceCPU(const GPUFenceCPU&) = delete;
    MRAY_HYBRID                 GPUFenceCPU(GPUFenceCPU&&) noexcept = default;
    GPUFenceCPU&                operator=(const GPUFenceCPU&) = delete;
    MRAY_HYBRID GPUFenceCPU&    operator=(GPUFenceCPU&&) noexcept = default;
    MRAY_HYBRID                 ~GPUFenceCPU() = default;

    MRAY_HYBRID void            Wait() const;
};

class GPUQueueCPU
{
    public:
    struct ControlBlockData
    {
        std::atomic_uint64_t    issuedKernelCounter = 0;
        std::atomic_uint64_t    completedKernelCounter = 0;
        std::mutex              issueMutex;
        //
        std::atomic_uint32_t    curBlockCounter = 0;
    };

    private:
    using ControlBlockPtr = std::unique_ptr<ControlBlockData>;

    friend class GPUFenceCPU;

    template<auto Kernel, class... Args>
    MRAY_HYBRID void IssueKernelInternal(std::string_view name,
                                         uint32_t totalWorkCount,
                                         uint32_t tpb,
                                         //
                                         Args&&...) const;

    template<class Lambda>
    MRAY_HYBRID void IssueLambdaInternal(std::string_view name,
                                         uint32_t totalWorkCount,
                                         uint32_t tpb,
                                         Lambda&&) const;

    private:
    AnnotationHandle    domain;
    const GPUDeviceCPU* myDevice    = nullptr;
    ThreadPool*         tp          = nullptr;
    ControlBlockPtr     cb          = nullptr;


    // Friends that use RecommendedBlockCount... functions
    template<bool, class, class>
    friend size_t algorithms::RadixSortTMSize(size_t, const GPUQueueCPU&);
    template<bool, class K, class V>
    friend uint32_t algorithms::RadixSort(Span<Span<K>, 2>, Span<Span<V>, 2>,
                                          Span<Byte>, const GPUQueueCPU&,
                                          const Vector2ui&);

    template<class>
    friend size_t algorithms::BinPartitionTMSize(size_t, const GPUQueueCPU&);
    template<class T, class U>
    friend void algorithms::BinaryPartition(Span<T>, Span<uint32_t, 1>,
                                            Span<Byte>, Span<const T>,
                                            const GPUQueueCPU&, U&&);
    template<class>
    friend size_t algorithms::ExclusiveScanTMSize(size_t, const GPUQueueCPU&);
    template<class T, class B>
    friend void algorithms::ExclusiveScan(Span<T>, Span<Byte>, Span<const T>,
                                          const T&, const GPUQueueCPU&, B&& op);

    MRAY_HYBRID
    uint32_t    RecommendedBlockCountDevice(const void* kernelPtr,
                                            uint32_t threadsPerBlock,
                                            uint32_t sharedMemSize) const;

    MRAY_HYBRID
    uint32_t    RecommendedBlockCountSM(const void* kernelPtr,
                                        uint32_t threadsPerBlock,
                                        uint32_t sharedMemSize);
    MRAY_HYBRID
    uint32_t    DetermineGridStrideBlock(const void* kernelPtr,
                                         uint32_t sharedMemSize,
                                         uint32_t threadCount,
                                         uint32_t workCount) const;

    public:
    // Constructors & Destructor
                    GPUQueueCPU() = default;
                    GPUQueueCPU(ThreadPool& tp,
                                AnnotationHandle domain,
                                const GPUDeviceCPU* device);
                                GPUQueueCPU(const GPUQueueCPU&) = delete;
                    GPUQueueCPU(GPUQueueCPU&&) noexcept = default;
    GPUQueueCPU&    operator=(const GPUQueueCPU&) = delete;
    GPUQueueCPU&    operator=(GPUQueueCPU&&) noexcept = default;
                    ~GPUQueueCPU();

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
    MRAY_GPU void   DeviceIssueWorkKernel(std::string_view name,
                                          DeviceWorkIssueParams,
                                          //
                                          Args&&...) const;
    template<class Lambda>
    MRAY_GPU void   DeviceIssueWorkLambda(std::string_view name,
                                          DeviceWorkIssueParams,
                                          //
                                          Lambda&&) const;
    template<auto Kernel, class... Args>
    MRAY_GPU void   DeviceIssueBlockKernel(std::string_view name,
                                           DeviceBlockIssueParams,
                                           //
                                           Args&&...) const;
    template<class Lambda, uint32_t Bounds = StaticThreadPerBlock1D()>
    MRAY_GPU void   DeviceIssueBlockLambda(std::string_view name,
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
    //
    int                     deviceId;
    AnnotationHandle        domain;
    ThreadPool*             threadPool;
    //
    std::string             name;
    size_t                  totalMemory;

    protected:
    public:
    // Constructors & Destructor
    explicit                GPUDeviceCPU(ThreadPool& tp, int deviceId, AnnotationHandle);
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
    std::unique_ptr<ThreadPool> localTP;
    GPUList                     systemGPUs;
    GPUPtrList                  systemGPUPtrs;
    AnnotationHandle            cpuDomain;

    // TODO: Check designs for this, this made the GPUSystem global
    // which is fine
    static void         ThreadInitFunction();

    protected:
    public:
    // Constructors & Destructor
                        GPUSystemCPU();
                        GPUSystemCPU(ThreadPool& tp);
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

    // Annotation for profiling etc. (currently nothing is used)
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

inline
void AtomicWaitValueToAchieve(std::atomic_uint64_t& checkLoc, uint64_t valToAchieve)
{
    // I dont understand the new wait functions...
    //
    // We need to wait until "X >= Y" (i.e. completedCount >= achievedCount)
    // but atomic only waits when X == Y returns otherwise.
    //
    // So we awake the threads a lot. Below code somewhat minimizes this.
    // "wait_predicate" would have been better?
    // Probably hard to implement or most CPU's/Kernels does not have
    // this functionality, I dunno.
    //
    // Anyway if this comment stays it works robustly.
    // TODO: Even if this there is no ABA issue, check the perf
    // and fall back to condition variable / mutex
    // compare as such "wait until (curIssueCount >= oldIssueCount)
    // similar implementation of the TimelineSemaphore class.
    //
    // Order is important here, for loop semantics should
    // give what we want
    // Load the data initially,
    //      if it is already >=, return
    //      if not wait
    // When thread awakes
    // reload the value and check if its passed again etc...
    for(uint64_t check = checkLoc; check < valToAchieve; check = checkLoc)
        checkLoc.wait(check);
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCPU::GPUFenceCPU(const GPUQueueCPU& q)
    : valueToWait(q.cb->issuedKernelCounter.load())
    , completedKernelCounter(&q.cb->completedKernelCounter)
{}

MRAY_HYBRID MRAY_CGPU_INLINE
void GPUFenceCPU::Wait() const
{
    AtomicWaitValueToAchieve(*completedKernelCounter, valueToWait);
}

MRAY_HYBRID inline
GPUQueueCPU::GPUQueueCPU(ThreadPool& tp,
                         AnnotationHandle domain,
                         const GPUDeviceCPU* device)
    : domain(domain)
    , myDevice(device)
    , tp(&tp)
    , cb(std::make_unique<ControlBlockData>())
{}

// Memory Movement (Async)
template <class T>
MRAY_HOST
void GPUQueueCPU::MemcpyAsync(Span<T> regionTo, Span<const T> regionFrom) const
{
    using namespace std::string_view_literals;

    assert(regionTo.size() >= regionFrom.size());
    uint32_t elemCount = uint32_t(regionFrom.size());
    uint32_t tpb = StaticThreadPerBlock1D();
    uint32_t blockCount = Math::DivideUp(elemCount, tpb);
    // We find grid size via StaticTPB, we only set
    // single thread for each block. We want to use
    // memcpy, instead of thread copy assigning variables
    // one by one.
    IssueBlockLambda
    (
        "MemcpyAsync"sv,
        DeviceBlockIssueParams
        {
            .gridSize= blockCount,
            .blockSize = 1u
        },
        [=](KernelCallParamsCPU kp)
        {
            uint32_t elemCount = uint32_t(regionFrom.size());

            uint32_t i = kp.GlobalId();
            uint32_t writeBound = std::min(elemCount, (i + 1) * tpb);
            writeBound -= i * tpb;

            auto localFromSpan = regionFrom.subspan(i * tpb, writeBound);
            auto localToSpan = regionTo.subspan(i * tpb, writeBound);
            if(writeBound != 0)
            {
                assert(localFromSpan.size() == localToSpan.size());
                std::memcpy(localToSpan.data(), localFromSpan.data(), localFromSpan.size_bytes());
            }
        }
    );
}

template <class T>
MRAY_HOST
void GPUQueueCPU::MemcpyAsync2D(Span<T> regionTo, size_t toStride,
                                Span<const T> regionFrom, size_t fromStride,
                                Vector2ui copySize) const
{
    using namespace std::string_view_literals;

    assert(toStride * (copySize[1] - 1) + copySize[0] <=
           regionTo.size());
    assert(fromStride * (copySize[1] - 1) + copySize[0] <=
           regionFrom.size());
    assert(toStride >= copySize[0]);
    assert(fromStride >= copySize[0]);

    assert(regionTo.size() >= regionFrom.size());
    //
    IssueBlockLambda
    (
        "MemcpyAsync2D"sv,
        DeviceBlockIssueParams
        {
            .gridSize = copySize[1],
            .blockSize = 1u
        },
        [=](KernelCallParamsCPU kp)
        {
            uint32_t i = kp.GlobalId();
            if(i >= copySize[1]) return;

            auto copyTo = regionTo.subspan(i * toStride, copySize[0]);
            auto copyFrom = regionFrom.subspan(i * fromStride, copySize[0]);
            std::memcpy(copyTo.data(), copyFrom.data(), copyTo.size_bytes());
        }
    );
}

template <class T>
MRAY_HOST
void GPUQueueCPU::MemcpyAsyncStrided(Span<T> regionTo, size_t outputByteStride,
                                     Span<const T> regionFrom, size_t inputByteStride) const
{
    using namespace std::string_view_literals;

    size_t actualInStride = (inputByteStride == 0) ? sizeof(T) : inputByteStride;
    size_t actualOutStride = (outputByteStride == 0) ? sizeof(T) : outputByteStride;

    size_t elemCountIn = Math::DivideUp(regionFrom.size_bytes(), actualInStride);
    assert(elemCountIn == Math::DivideUp(regionTo.size_bytes(), actualOutStride));

    // Strip down to bytes, since copy is strided
    Byte* toPtr = reinterpret_cast<Byte*>(regionTo.data());
    const Byte* fromPtr = reinterpret_cast<const Byte*>(regionFrom.data());

    IssueWorkLambda
    (
        "MemcpyAsyncStrided"sv,
        DeviceWorkIssueParams{ .workCount = static_cast<uint32_t>(elemCountIn)},
        [=](KernelCallParamsCPU kp)
        {
            uint32_t i = kp.GlobalId();
            if(i >= elemCountIn) return;

            // We copy each element manually since it is strided
            // This is probably slow compared to GPU, but it is rarely
            // used. Hopefully, it will not be a problem.
            Byte* itemTo = toPtr + i * actualOutStride;
            const Byte* itemFrom = fromPtr + i * actualInStride;
            // We can lift to type T here
            // but what about alignment???
            std::memcpy(itemTo, itemFrom, sizeof(T));
        }
    );
}

template <class T>
MRAY_HOST
void GPUQueueCPU::MemsetAsync(Span<T> region, uint8_t perByteValue) const
{
    using namespace std::string_view_literals;

    uint32_t elemCount = static_cast<uint32_t>(region.size());
    uint32_t tpb = StaticThreadPerBlock1D();
    uint32_t blockCount = Math::DivideUp(elemCount, tpb);
    IssueBlockLambda
    (
        "MemsetAsync"sv,
        DeviceBlockIssueParams
        {
            .gridSize = blockCount,
            .blockSize = 1u,
        },
        [=](KernelCallParamsCPU kp)
        {
            uint32_t i = kp.GlobalId();
            uint32_t writeBound = std::min(elemCount, (i + 1) * tpb);
            writeBound -= i * tpb;
            if(writeBound != 0)
            {
                auto localRegion = region.subspan(i * tpb, writeBound);
                std::memset(localRegion.data(), perByteValue, localRegion.size_bytes());
            }
        }
    );
}

MRAY_HOST inline
void GPUQueueCPU::IssueBufferForDestruction(TransientData data) const
{
    using namespace std::string_view_literals;
    // We technically do not need to go through delete callbacks etc
    // but ThreadPool accepts non-mutable functions (lambda's)
    // so we rely on that
    // since we are in the control.
    void* ptr = TransientPoolIssueBufferForDestruction(std::move(data));
    IssueBlockLambda
    (
        "DestroyTransientBuffer"sv,
        DeviceBlockIssueParams{.gridSize = 1, .blockSize = 1},
        [ptr](KernelCallParamsCPU)
        {
            TransientPoolDestroyCallback(ptr);
        }
    );
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUFenceCPU GPUQueueCPU::Barrier() const
{
    return GPUFenceCPU(*this);
}

MRAY_HOST inline
void GPUQueueCPU::IssueSemaphoreWait(GPUSemaphoreViewCPU& sem) const
{
    using namespace std::string_view_literals;
    IssueBlockLambda
    (
        "SemWait"sv,
        DeviceBlockIssueParams{.gridSize = 1, .blockSize = 1},
        [&](KernelCallParamsCPU)
        {
            std::ignore = sem.HostAcquire();
        }
    );
}

MRAY_HOST inline
void GPUQueueCPU::IssueSemaphoreSignal(GPUSemaphoreViewCPU& sem) const
{
    using namespace std::string_view_literals;
    IssueBlockLambda
    (
        "SemSignal"sv,
        DeviceBlockIssueParams{.gridSize = 1, .blockSize = 1},
        [&](KernelCallParamsCPU)
        {
            sem.HostRelease();
        }
    );
}


MRAY_HOST inline
void GPUQueueCPU::IssueWait(const GPUFenceCPU& barrier) const
{
    using namespace std::string_view_literals;

    auto* barrierValue = barrier.completedKernelCounter;
    uint64_t valueToAchieve = barrier.valueToWait;

    IssueBlockLambda
    (
        "WaitOtherQueue"sv,
        DeviceBlockIssueParams{.gridSize = 1, .blockSize = 1},
        [barrierValue, valueToAchieve](KernelCallParamsCPU)
        {
            AtomicWaitValueToAchieve(*barrierValue, valueToAchieve);
        }
    );
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t GPUQueueCPU::RecommendedBlockCountSM(const void*, uint32_t, uint32_t)
{
    return 1u;
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t GPUQueueCPU::RecommendedBlockCountDevice(const void*, uint32_t, uint32_t) const
{
    return SMCount();
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t GPUQueueCPU::SMCount() const
{
    return myDevice->SMCount();
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t GPUQueueCPU::DetermineGridStrideBlock(const void*,
                                               uint32_t,
                                               uint32_t tbp,
                                               uint32_t workCount) const
{
    // Limit with SM count
    uint32_t blockCount = Math::DivideUp(workCount, tbp);
    blockCount = std::min(blockCount, SMCount());

    return blockCount;
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
    assert(regionTo.size() >= regionFrom.size());
    std::memcpy(regionTo.data(), regionFrom.data(),
                regionTo.size_bytes());
}

template <class T>
MRAY_HOST
void GPUSystemCPU::Memset(Span<T> region, uint8_t perByteValue) const
{
    std::memset(region.data(), int(perByteValue), region.size_bytes());
}

}


