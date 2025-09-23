#include "GPUSystemCPU.h"
#include "Core/System.h"
#include "DeviceMemoryCPU.h"

#include "Core/TimelineSemaphore.h"
#include "Core/ThreadPool.h"
#include "Core/Log.h"

#ifdef MRAY_WINDOWS
    #include <Windows.h>
    #define CPUId __cpuidex

#elif defined MRAY_LINUX
    #include <sys/sysinfo.h>

    #include <pmmintrin.h>
    #include <xmmintrin.h>

    void cpuidex_func(int* regs, int i, int j)
    {
        asm volatile
        (
            "cpuid" : "=a" (regs[0]),
                      "=b" (regs[1]),
                      "=c" (regs[2]),
                      "=d" (regs[3])
                    : "a" (i), "c" (j)
        );
    }

    #define CPUId cpuidex_func
#endif

std::string GetCPUName()
{
    static constexpr uint32_t MAX_NAME_CHARS = 48u;
    std::string result(MAX_NAME_CHARS + 1, '\0');
    // Intel x86 ISA manual, volume 2 page 347
    // https://cdrdv2.intel.com/v1/dl/getContent/671110
    CPUId(reinterpret_cast<int*>(result.data() + 0 * 4), std::bit_cast<int>(0x80000002), 0);
    CPUId(reinterpret_cast<int*>(result.data() + 4 * 4), std::bit_cast<int>(0x80000003), 0);
    CPUId(reinterpret_cast<int*>(result.data() + 8 * 4), std::bit_cast<int>(0x80000004), 0);
    while(result.back() == '\0')
        result.pop_back();
    result.push_back('\0');

    return result;
}

uint64_t GetTotalCPUMemory()
{
    #ifdef MRAY_WINDOWS

        ULONGLONG totalMemKB;
        GetPhysicallyInstalledSystemMemory(&totalMemKB);
        return size_t(totalMemKB) * 1'000;

    #elif defined MRAY_LINUX

        // Doing a extern C here,
        // since I dunno struct sysinfo
        // and function sysinfo collides or not
        struct sysinfo s;
        sysinfo(&s);
        return size_t(s.totalram);

    #endif
}

namespace mray::host
{

GPUAnnotationCPU::GPUAnnotationCPU(AnnotationHandle,
                                   std::string_view name,
                                   const std::source_location& sLoc)
    : Base(name, sLoc)
{}

typename GPUAnnotationCPU::Scope
GPUAnnotationCPU::AnnotateScope() const
{
    return Scope(*this);
}

GPUSemaphoreViewCPU::GPUSemaphoreViewCPU(TimelineSemaphore* sem,
                                         uint64_t av)
    : externalSemaphore(sem)
    , acquireValue(av)
{}

void GPUSemaphoreViewCPU::SkipAState()
{
    acquireValue += 1;
}

uint64_t GPUSemaphoreViewCPU::ChangeToNextState()
{
    acquireValue += 2;
    return acquireValue - 1;
}

bool GPUSemaphoreViewCPU::HostAcquire()
{
    bool acquired = externalSemaphore->Acquire(acquireValue);
    return acquired;
}

void GPUSemaphoreViewCPU::HostRelease()
{
    externalSemaphore->Release();
}

GPUDeviceCPU::GPUDeviceCPU(ThreadPool& tp, int deviceId,
                           AnnotationHandle domain)
    : deviceId(deviceId)
    , domain(domain)
    , threadPool(&tp)
    , name(GetCPUName())
    , totalMemory(GetTotalCPUMemory())
{
    for(uint32_t i = 0; i < TotalQueuePerDevice(); i++)
        queues.emplace_back(tp, domain, this);
    transferQueue = GPUQueueCPU(tp, domain, this);
}

bool GPUDeviceCPU::operator==(const GPUDeviceCPU& other) const
{
    return deviceId == other.deviceId;
}

int GPUDeviceCPU::DeviceId() const
{
    return deviceId;
}

std::string GPUDeviceCPU::Name() const
{
    return name;
}

std::string GPUDeviceCPU::ComputeCapability() const
{
    return std::string("none");
}

size_t GPUDeviceCPU::TotalMemory() const
{
    return totalMemory;
}

uint32_t GPUDeviceCPU::SMCount() const
{
    return threadPool->ThreadCount();
}

uint32_t GPUDeviceCPU::MaxActiveBlockPerSM(uint32_t) const
{
    return 1u;
}

const GPUQueueCPU& GPUDeviceCPU::GetComputeQueue(uint32_t index) const
{
    assert(index < ComputeQueuePerDevice);
    return queues[index];
}

const GPUQueueCPU& GPUDeviceCPU::GetTransferQueue() const
{
    return transferQueue;
}

GPUQueueCPU::~GPUQueueCPU()
{
    tp->Wait();
}

GPUSystemCPU::GPUSystemCPU(bool logBanner, uint32_t threadCount)
    : localTP(nullptr)
    , cpuDomain(nullptr)
{
    uint32_t queueSize = Math::NextPowerOfTwo(threadCount * 128);
    localTP = std::make_unique<ThreadPool>
    (
        typename ThreadPool::InitParams
        {
            .threadCount = threadCount,
            .queueSize  = queueSize
        },
        [](SystemThreadHandle handle, uint32_t id)
        {
            RenameThread(handle, MRAY_FORMAT("{:03d}_[G]MRay", id));
            ThreadInitFunction();
        }
    );
    // TODO: Check NUMA stuff:
    //  - Put a flag, to auto combine or split
    //    NUMA nodes as separate "Device"
    systemGPUs.emplace_back(*localTP.get(), 0, nullptr);
    systemGPUPtrs.push_back(&systemGPUs.back());

    // Skip the banner if not requested
    if(!logBanner) return;

    std::string banner;
    banner.reserve(1024);
    bool isFirst = true;
    for(const auto& gpu : systemGPUs)
    {
        if(!isFirst)
        {
            banner += "---------------------\n";
            isFirst = false;
        }

        double memGiB = double(gpu.TotalMemory());
        memGiB /= 1024.0;
        memGiB /= 1024.0;
        memGiB /= 1204.0;
        banner += MRAY_FORMAT("Name      : {}\n"
                              "Memory    : {:.3f} GiB\n"
                              "---------------------\n",
                              gpu.Name(),
                              memGiB);
    }
    MRAY_LOG("----Tracer-GPU(s)----\n{}", banner);
}

GPUSystemCPU::~GPUSystemCPU()
{}

std::vector<size_t> GPUSystemCPU::SplitWorkToMultipleGPU(uint32_t workCount,
                                                         uint32_t,
                                                         uint32_t,
                                                         void*) const
{
    // We assume single NUMA currently, it should be changed
    // after that is implemented.
    return std::vector<size_t>(workCount);
}

const GPUSystemCPU::GPUList& GPUSystemCPU::SystemDevices() const
{
    return systemGPUs;
}

const GPUSystemCPU::GPUPtrList& GPUSystemCPU::AllGPUs() const
{
    return systemGPUPtrs;
}

const GPUDeviceCPU& GPUSystemCPU::BestDevice() const
{
    return systemGPUs[0];
}

KernelAttributes GPUSystemCPU::GetKernelAttributes(const void*) const
{
    // TODO: This does not make sense for the CPU
    // It'd be cool if we could've returned
    // Stack usage, total size static variables etc.
    return KernelAttributes
    {
        .localMemoryPerThread = 0,
        .constantMemorySize = 0,
        .maxDynamicSharedMemorySize = 0,
        .maxTBP = 1,
        .registerCountPerThread = 0,
        .staticSharedMemorySize = 0
    };
}

bool GPUSystemCPU::SetKernelShMemSize(const void*, int) const
{
    return true;
}

size_t GPUSystemCPU::TotalMemory() const
{
    return GetTotalCPUMemory();
}

void GPUSystemCPU::SyncAll() const
{
    for(const auto& gpu : AllGPUs())
    {
        for(uint32_t i = 0; i < ComputeQueuePerDevice; i++)
            gpu->GetComputeQueue(0).Barrier().Wait();
        gpu->GetTransferQueue().Barrier().Wait();
    }
}

void GPUSystemCPU::ThreadInitFunction()
{
    // From the embree tutorials...
    // For best performance we need to change these
    // First time setting a control register on x86 device :)
    //
    // GCC complains about this usage, but its all macros etc.
    // nothing we can do about it...
    #ifdef MRAY_GCC
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wsign-conversion"
    #endif

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    #ifdef MRAY_GCC
        #pragma GCC diagnostic pop
    #endif
}

GPUAnnotationCPU GPUSystemCPU::CreateAnnotation(std::string_view name) const
{
    return GPUAnnotationCPU(nullptr, name);
}

GPUThreadInitFunction GPUSystemCPU::GetThreadInitFunction() const
{
    return &GPUSystemCPU::ThreadInitFunction;
}

// Semaphore related namespace global functions
void TimelineSemAcquireInternal(void* params)
{
    GPUSemaphoreViewCPU* ts = static_cast<GPUSemaphoreViewCPU*>(params);
    // Device side acquisition, we cant do much here,
    // because this is async, so we drop the result and on text iteration
    // GPU driving code may check the semaphore before sending an
    // acquisition code to GPU (a host launch)
    std::ignore = ts->HostAcquire();
}

void TimelineSemReleaseInternal(void* params)
{
    GPUSemaphoreViewCPU* ts = static_cast<GPUSemaphoreViewCPU*>(params);
    ts->HostRelease();
}

}