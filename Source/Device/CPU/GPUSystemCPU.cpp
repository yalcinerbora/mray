#include "GPUSystemCPU.h"
#include "DeviceMemoryCPU.h"

#include "Core/Error.h"
#include "Core/TimelineSemaphore.h"

#include "Core/Timer.h"

namespace mray::host
{

GPUAnnotationCPU::Scope::Scope(AnnotationHandle d)
    : domain(d)
{}

GPUAnnotationCPU::Scope::~Scope()
{
}

GPUAnnotationCPU::GPUAnnotationCPU(AnnotationHandle h,
                                   std::string_view name)
    : domainHandle(h)
    , stringHandle(nullptr)
{
}

GPUAnnotationCPU::Scope GPUAnnotationCPU::AnnotateScope() const
{
    return Scope(AnnotationHandle{});
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

GPUDeviceCPU::GPUDeviceCPU(int deviceId, AnnotationHandle domain)
{
}

bool GPUDeviceCPU::operator==(const GPUDeviceCPU& other) const
{
    return true;
}

int GPUDeviceCPU::DeviceId() const
{
    return 0;
}

std::string GPUDeviceCPU::Name() const
{
    return std::string("CPU");
}

std::string GPUDeviceCPU::ComputeCapability() const
{
    return std::string("v0");
}

size_t GPUDeviceCPU::TotalMemory() const
{
    return 0u;
}

uint32_t GPUDeviceCPU::SMCount() const
{
    return 1u;
}

uint32_t GPUDeviceCPU::MaxActiveBlockPerSM(uint32_t threadsPerBlock) const
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

GPUSystemCPU::GPUSystemCPU()

{
    // TODO:
    if(globalGPUListPtr)
        throw MRayError("One process can only have "
                        "a single GPUSystem object!");

    throw MRayError("CPU-backend is not implemented yet!");
}

GPUSystemCPU::~GPUSystemCPU()
{
    globalGPUListPtr = nullptr;
}

std::vector<size_t> GPUSystemCPU::SplitWorkToMultipleGPU(uint32_t workCount,
                                                         uint32_t threadCount,
                                                         uint32_t sharedMemSize,
                                                         void* kernelPtr) const
{
    return std::vector<size_t>(9);
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

KernelAttributes GPUSystemCPU::GetKernelAttributes(const void* kernelPtr) const
{
    return KernelAttributes {};
}

bool GPUSystemCPU::SetKernelShMemSize(const void* kernelPtr,
                                      int sharedMemConfigSize) const
{
    return false;
}

size_t GPUSystemCPU::TotalMemory() const
{
    return 0u;
}

void GPUSystemCPU::SyncAll() const
{
}

typename GPUSystemCPU::GPUList*
GPUSystemCPU::globalGPUListPtr = nullptr;

void GPUSystemCPU::ThreadInitFunction()
{
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
    // Device side acquision, we cant do much here,
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