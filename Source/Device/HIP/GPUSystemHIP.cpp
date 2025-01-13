#include "GPUSystemHIP.h"
#include "DeviceMemoryHIP.h"

#include "Core/Error.hpp"
#include "Core/TimelineSemaphore.h"

#include <hip/hip_runtime.h>

#include "Core/Timer.h"

namespace mray::hip
{

GPUAnnotationHIP::Scope::Scope(AnnotationHandle d)
    : domain(d)
{}

GPUAnnotationHIP::Scope::~Scope()
{
    // TODO:
    //nvtxDomainRangePop(std::bit_cast<roctracer_domain_t>(domain));
}

GPUAnnotationHIP::GPUAnnotationHIP(AnnotationHandle h,
                                   std::string_view name)
    : domainHandle(h)
    , stringHandle(nullptr)
{
    // TODO:
    // roctracer_domain_t roctxDomain = std::bit_cast<roctracer_domain_t>(h);
    // stringHandle = nvtxDomainRegisterStringA(nvtxDomain, name.data());

}

GPUAnnotationHIP::Scope GPUAnnotationHIP::AnnotateScope() const
{
    // TODO:
    // nvtxDomainHandle_t nvtxDomain = std::bit_cast<nvtxDomainHandle_t>(domainHandle);
    // nvtxEventAttributes_t attrib = {};
    // attrib.version = NVTX_VERSION;
    // attrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    // //NVTX_MESSAGE_TYPE_ASCII;
    // attrib.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    // attrib.message.registered = std::bit_cast<nvtxStringHandle_t>(stringHandle);

    // nvtxDomainRangePushEx(nvtxDomain, &attrib);
    // return Scope(domainHandle);
}

GPUSemaphoreViewHIP::GPUSemaphoreViewHIP(TimelineSemaphore* sem,
                                         uint64_t av)
    : externalSemaphore(sem)
    , acquireValue(av)
{}

void GPUSemaphoreViewHIP::SkipAState()
{
    acquireValue += 1;
}

uint64_t GPUSemaphoreViewHIP::ChangeToNextState()
{
    acquireValue += 2;
    return acquireValue - 1;
}

bool GPUSemaphoreViewHIP::HostAcquire()
{
    bool acquired = externalSemaphore->Acquire(acquireValue);
    //MRAY_LOG("[Tracer]: Acquired Img {}", acquireValue);
    return acquired;
}

void GPUSemaphoreViewHIP::HostRelease()
{
    //MRAY_LOG("[Tracer]: Released Img\n"
    //         "----------------------");
    externalSemaphore->Release();
}

GPUDeviceHIP::GPUDeviceHIP(int deviceId, AnnotationHandle domain)
    : deviceId(deviceId)
{
    // Enforce non-async functions to explicitly synchronize
    // TODO: Again this is not available on HIP yet...
    // HIP_CHECK(hipInitDevice(deviceId,
    //                         // TODO: This is not available on HIP yet
    //                         // Enable this later
    //                         // hipDeviceSyncMemops |
    //                         hipDeviceScheduleAuto,
    //                         hipInitDeviceFlagsAreValid));
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    // TODO: ....
    if(true)
    {
        throw MRayError("AMD GPUs are not supported yet!");
    }

    // Check if we synchronized the non-async mem copies
    uint32_t flags = 0;
    HIP_CHECK(hipGetDeviceFlags(&flags));
    // TODO: This is not available on HIP yet
    // enable this later
    // if((flags & hipDeviceSyncMemops) == 0)
    // {
    //     throw MRayError("Unable to set \"hipDevice"
    //                     "SyncMemops\" flag on the device! ({:s})", props.name);
    // }
    if((flags & hipDeviceMapHost) == 0)
    {
        throw MRayError("The device does not  support \"hipDeviceMapHost\""
                        "flag! ({:s}:{})",
                        props.name, deviceId);
    }
    if(props.hostRegisterSupported == 0)
    {
        throw MRayError("The device does not support registering an "
                        "arbitrary host pointer! ({:s}:{})",
                        props.name, deviceId);
    }

    // Check VMM support (entire system requires this functionality)
    int vmmEnabled = 0;
    auto vmmAttib = hipDeviceAttributeVirtualMemoryManagementSupported;
    HIP_DRIVER_CHECK(hipDeviceGetAttribute(&vmmEnabled, vmmAttib, deviceId));
    if(vmmEnabled == 0)
    {
        throw MRayError("The device do not have virtual memory "
                        "management support!  ({:s}:{})",
                        props.name, deviceId);
    }

    // All Seems Fine Allocate Queues
    //
    HIP_CHECK(hipSetDevice(deviceId));
    for(uint32_t i = 0; i < ComputeQueuePerDevice; i++)
    {
        queues.emplace_back(props.multiProcessorCount, domain, this);
    }
    transferQueue = GPUQueueHIP(static_cast<uint32_t>(props.multiProcessorCount),
                                domain, this);
}

bool GPUDeviceHIP::operator==(const GPUDeviceHIP& other) const
{
    return deviceId == other.deviceId;
}

int GPUDeviceHIP::DeviceId() const
{
    return deviceId;
}

std::string GPUDeviceHIP::Name() const
{
    return std::string(props.name);
}

std::string GPUDeviceHIP::ComputeCapability() const
{
    return std::to_string(props.major) + std::to_string(props.minor);
}

size_t GPUDeviceHIP::TotalMemory() const
{
    return props.totalGlobalMem;
}

uint32_t GPUDeviceHIP::SMCount() const
{
    return static_cast<uint32_t>(props.multiProcessorCount);
}

uint32_t GPUDeviceHIP::MaxActiveBlockPerSM(uint32_t threadsPerBlock) const
{
    return static_cast<uint32_t>(props.maxThreadsPerMultiProcessor) / threadsPerBlock;
}

const GPUQueueHIP& GPUDeviceHIP::GetComputeQueue(uint32_t index) const
{
    assert(index < ComputeQueuePerDevice);
    return queues[index];
}

const GPUQueueHIP& GPUDeviceHIP::GetTransferQueue() const
{
    return transferQueue;
}

GPUSystemHIP::GPUSystemHIP()
    // TODO:
    : roctxDomain(0)
{
    if(globalGPUListPtr) throw MRayError("One process can only have "
                                         "a single GPUSystem object!");

    // Initialize the HIP
    int deviceCount;
    hipError_t err;

    err = hipGetDeviceCount(&deviceCount);
    if(err == hipErrorInsufficientDriver)
    {
        throw MRayError("Device has no drivers!");
    }
    else if(err == hipErrorNoDevice)
    {
        throw MRayError("No device is found!");
    }

    // All Fine Start Query Devices
    for(int i = 0; i < deviceCount; i++)
    {
        systemGPUs.emplace_back(i, roctxDomain);
        systemGPUPtrs.push_back(&systemGPUs.back());
    }
    // TODO: Do topology stuff here
    // handle selection etc. this is too
    // primitive currently

    // TODO: a design leak but what else you can do?
    globalGPUListPtr = &systemGPUs;
}

GPUSystemHIP::~GPUSystemHIP()
{
    // TODO:
    // nvtxDomainDestroy(static_cast<nvtxDomainHandle_t>(nvtxDomain));

    for(const auto& device : systemGPUs)
    {
        HIP_CHECK(hipSetDevice(device.DeviceId()));
        HIP_CHECK(hipDeviceSynchronize());
    }
    systemGPUs.clear();
    HIP_CHECK(hipDeviceReset());

    globalGPUListPtr = nullptr;
}

std::vector<size_t> GPUSystemHIP::SplitWorkToMultipleGPU(uint32_t workCount,
                                                         uint32_t threadCount,
                                                         uint32_t sharedMemSize,
                                                         void* kernelPtr) const
{
    std::vector<size_t> workPerGPU;
    // Split work into all GPUs
    uint32_t totalAvailBlocks = 0;
    for(const GPUDeviceHIP& g : systemGPUs)
    {
        uint32_t blockPerSM = GPUQueueHIP::RecommendedBlockCountSM(kernelPtr,
                                                                   threadCount,
                                                                   sharedMemSize);
        uint32_t blockGPU = blockPerSM * g.SMCount();
        workPerGPU.push_back(blockGPU);
        totalAvailBlocks += blockGPU;
    }

    // Total Threads
    uint32_t totalThreads = threadCount * totalAvailBlocks;
    uint32_t iterationPerThread = Math::DivideUp(workCount, totalThreads);

    size_t workDispatched = 0;
    for(size_t i = 0; i < systemGPUs.size(); i++)
    {
        // Send Data
        size_t workPerBlock = threadCount * iterationPerThread;
        size_t gpuWorkCount = workPerGPU[i] * workPerBlock;
        gpuWorkCount = std::min(gpuWorkCount, workCount - workDispatched);
        workDispatched += gpuWorkCount;
        workPerGPU[i] = gpuWorkCount;
    }
    return workPerGPU;
}

const GPUSystemHIP::GPUList& GPUSystemHIP::SystemDevices() const
{
    return systemGPUs;
}

const GPUSystemHIP::GPUPtrList& GPUSystemHIP::AllGPUs() const
{
    return systemGPUPtrs;
}

const GPUDeviceHIP& GPUSystemHIP::BestDevice() const
{
    // Return the largest memory GPU
    auto MemoryCompare = [](const GPUDeviceHIP& a, const GPUDeviceHIP& b)
    {
        return (a.TotalMemory() < b.TotalMemory());
    };
    auto element = std::max_element(systemGPUs.cbegin(), systemGPUs.cend(), MemoryCompare);
    return *element;
}

KernelAttributes GPUSystemHIP::GetKernelAttributes(const void* kernelPtr) const
{
    hipFuncAttributes result;
    HIP_CHECK(hipFuncGetAttributes(&result, kernelPtr));

    return KernelAttributes
    {
        .localMemoryPerThread = result.localSizeBytes,
        .constantMemorySize = result.constSizeBytes,
        .maxDynamicSharedMemorySize = result.maxDynamicSharedSizeBytes,
        .maxTBP = result.maxThreadsPerBlock,
        .registerCountPerThread = result.numRegs,
        .staticSharedMemorySize = result.sharedSizeBytes
    };
}

bool GPUSystemHIP::SetKernelShMemSize(const void* kernelPtr,
                                       int sharedMemConfigSize) const
{
    hipError_t error = hipFuncSetAttribute(kernelPtr,
                                             hipFuncAttributePreferredSharedMemoryCarveout,
                                             sharedMemConfigSize);
    return (error == hipSuccess);
}

size_t GPUSystemHIP::TotalMemory() const
{
    size_t memSize = 0;
    for(const auto& gpu : systemGPUs)
    {
        memSize += gpu.TotalMemory();
    }
    return memSize;
}

void GPUSystemHIP::SyncAll() const
{
    for(const auto& gpu : systemGPUs)
    {
        HIP_CHECK(hipSetDevice(gpu.DeviceId()));
        HIP_CHECK(hipDeviceSynchronize());
    }
}

typename GPUSystemHIP::GPUList*
GPUSystemHIP::globalGPUListPtr = nullptr;

void GPUSystemHIP::ThreadInitFunction()
{
    // Set all the devices on the thread, should be enough?
    for(const auto& device : (*globalGPUListPtr))
        HIP_CHECK(hipSetDevice(device.DeviceId()));
}

GPUAnnotationHIP GPUSystemHIP::CreateAnnotation(std::string_view name) const
{
    // TODO:
    //return GPUAnnotationCUDA(roctxDomain, name);
    return GPUAnnotationHIP(nullptr, name);
}

GPUThreadInitFunction GPUSystemHIP::GetThreadInitFunction() const
{
    return &GPUSystemHIP::ThreadInitFunction;
}

// Semaphore related namespace global functions
void TimelineSemAcquireInternal(void* params)
{
    GPUSemaphoreViewHIP* ts = static_cast<GPUSemaphoreViewHIP*>(params);
    // Device side acquision, we cant do much here,
    // because this is async, so we drop the result and on text iteration
    // GPU driving code may check the semaphore before sending an
    // acquisition code to GPU (a host launch)
    std::ignore = ts->HostAcquire();
}

void TimelineSemReleaseInternal(void* params)
{
    GPUSemaphoreViewHIP* ts = static_cast<GPUSemaphoreViewHIP*>(params);
    ts->HostRelease();
}

}