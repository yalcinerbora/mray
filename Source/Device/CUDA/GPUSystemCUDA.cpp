#include "GPUSystemCUDA.h"
#include "DeviceMemoryCUDA.h"

#include "Core/Error.hpp"
#include "Core/TimelineSemaphore.h"

#include <cuda.h>
#include <nvtx3/nvToolsExt.h>

#include "Core/Timer.h"

namespace mray::cuda
{

GPUSemaphoreViewCUDA::GPUSemaphoreViewCUDA(TimelineSemaphore* sem,
                                           uint64_t av)
    : externalSemaphore(sem)
    , acquireValue(av)
{}

void GPUSemaphoreViewCUDA::SkipAState()
{
    acquireValue += 1;
}

uint64_t GPUSemaphoreViewCUDA::ChangeToNextState()
{
    acquireValue += 2;
    return acquireValue - 1;
}

bool GPUSemaphoreViewCUDA::HostAcquire()
{
    bool acquired = externalSemaphore->Acquire(acquireValue);
    MRAY_LOG("[Tracer] Acquired Img {}", acquireValue);
    return acquired;
}

void GPUSemaphoreViewCUDA::HostRelease()
{
    MRAY_LOG("[Tracer] Released Img\n"
             "----------------------");
    externalSemaphore->Release();
}

GPUDeviceCUDA::GPUDeviceCUDA(int deviceId, AnnotationHandle domain)
    : deviceId(deviceId)
{
    // Enforce non-async functions to explicitly synchronize
    CUDA_CHECK(cudaInitDevice(deviceId,
                              cudaDeviceSyncMemops |
                              cudaDeviceScheduleAuto,
                              cudaInitDeviceFlagsAreValid));
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));

    // Check if we synchronized the non-async mem copies
    uint32_t flags = 0;
    CUDA_CHECK(cudaGetDeviceFlags(&flags));
    if((flags & cudaDeviceSyncMemops) == 0)
    {
        throw MRayError("Unable to set \"cudaDevice"
                        "SyncMemops\" flag on the device! ({:s})", props.name);
    }
    if((flags & cudaDeviceMapHost) == 0)
    {
        throw MRayError("The device does not  support \"cudaDeviceMapHost\""
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
    auto vmmAttib = CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED;
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&vmmEnabled, vmmAttib, deviceId));
    if(vmmEnabled == 0)
    {
        throw MRayError("The device do not have virtual memory "
                        "management support!  ({:s}:{})",
                        props.name, deviceId);
    }

    // All Seems Fine Allocate Queues
    //
    CUDA_CHECK(cudaSetDevice(deviceId));
    for(uint32_t i = 0; i < ComputeQueuePerDevice; i++)
    {
        queues.emplace_back(props.multiProcessorCount, domain, this);
    }
    transferQueue = GPUQueueCUDA(props.multiProcessorCount, domain, this);
}

bool GPUDeviceCUDA::operator==(const GPUDeviceCUDA& other) const
{
    return deviceId == other.deviceId;
}

int GPUDeviceCUDA::DeviceId() const
{
    return deviceId;
}

std::string GPUDeviceCUDA::Name() const
{
    return std::string(props.name);
}

std::string GPUDeviceCUDA::ComputeCapability() const
{
    return std::to_string(props.major) + std::to_string(props.minor);
}

size_t GPUDeviceCUDA::TotalMemory() const
{
    return props.totalGlobalMem;
}

uint32_t GPUDeviceCUDA::SMCount() const
{
    return static_cast<uint32_t>(props.multiProcessorCount);
}

uint32_t GPUDeviceCUDA::MaxActiveBlockPerSM(uint32_t threadsPerBlock) const
{
    return static_cast<uint32_t>(props.maxThreadsPerMultiProcessor) / threadsPerBlock;
}

const GPUQueueCUDA& GPUDeviceCUDA::GetComputeQueue(uint32_t index) const
{
    assert(index < ComputeQueuePerDevice);
    return queues[index];
}

const GPUQueueCUDA& GPUDeviceCUDA::GetTransferQueue() const
{
    return transferQueue;
}

GPUSystemCUDA::GPUSystemCUDA()
    : nvtxDomain(nvtxDomainCreateA("MRayCUDA"))
{
    if(globalGPUListPtr) throw MRayError("One process can only have "
                                         "a single GPUSystem object!");

    // Initialize the CUDA
    int deviceCount;
    cudaError err;

    err = cudaGetDeviceCount(&deviceCount);
    if(err == cudaErrorInsufficientDriver)
    {
        throw MRayError("Device has no drivers!");
    }
    else if(err == cudaErrorNoDevice)
    {
        throw MRayError("No device is found!");
    }

    // All Fine Start Query Devices
    for(int i = 0; i < deviceCount; i++)
    {
        systemGPUs.emplace_back(i, nvtxDomain);
        systemGPUPtrs.push_back(&systemGPUs.back());
    }
    // TODO: Do topology stuff here
    // handle selection etc. this is too
    // primitive currently


    // TODO: a design leak but what else you can do?
    globalGPUListPtr = &systemGPUs;
}

GPUSystemCUDA::~GPUSystemCUDA()
{
    nvtxDomainDestroy(static_cast<nvtxDomainHandle_t>(nvtxDomain));

    for(const auto& device : systemGPUs)
    {
        CUDA_CHECK(cudaSetDevice(device.DeviceId()));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    systemGPUs.clear();
    CUDA_CHECK(cudaDeviceReset());

    globalGPUListPtr = nullptr;
}

std::vector<size_t> GPUSystemCUDA::SplitWorkToMultipleGPU(uint32_t workCount,
                                                          uint32_t threadCount,
                                                          uint32_t sharedMemSize,
                                                          void* kernelPtr) const
{
    std::vector<size_t> workPerGPU;
    // Split work into all GPUs
    uint32_t totalAvailBlocks = 0;
    for(const GPUDeviceCUDA& g : systemGPUs)
    {
        uint32_t blockPerSM = GPUQueueCUDA::RecommendedBlockCountSM(kernelPtr,
                                                                    threadCount,
                                                                    sharedMemSize);
        uint32_t blockGPU = blockPerSM * g.SMCount();
        workPerGPU.push_back(blockGPU);
        totalAvailBlocks += blockGPU;
    }

    // Total Threads
    uint32_t totalThreads = threadCount * totalAvailBlocks;
    uint32_t iterationPerThread = MathFunctions::DivideUp(workCount, totalThreads);

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

const GPUSystemCUDA::GPUList& GPUSystemCUDA::SystemDevices() const
{
    return systemGPUs;
}

const GPUSystemCUDA::GPUPtrList& GPUSystemCUDA::AllGPUs() const
{
    return systemGPUPtrs;
}

const GPUDeviceCUDA& GPUSystemCUDA::BestDevice() const
{
    // Return the largest memory GPU
    auto MemoryCompare = [](const GPUDeviceCUDA& a, const GPUDeviceCUDA& b)
    {
        return (a.TotalMemory() < b.TotalMemory());
    };
    auto element = std::max_element(systemGPUs.cbegin(), systemGPUs.cend(), MemoryCompare);
    return *element;
}

KernelAttributes GPUSystemCUDA::GetKernelAttributes(const void* kernelPtr) const
{
    cudaFuncAttributes result;
    CUDA_CHECK(cudaFuncGetAttributes(&result, kernelPtr));

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

bool GPUSystemCUDA::SetKernelShMemSize(const void* kernelPtr,
                                       int sharedMemConfigSize) const
{
    cudaError_t error = cudaFuncSetAttribute(kernelPtr,
                                             cudaFuncAttributePreferredSharedMemoryCarveout,
                                             sharedMemConfigSize);
    return (error == cudaSuccess);
}

size_t GPUSystemCUDA::TotalMemory() const
{
    size_t memSize = 0;
    for(const auto& gpu : systemGPUs)
    {
        memSize += gpu.TotalMemory();
    }
    return memSize;
}

void GPUSystemCUDA::SyncAll() const
{
    for(const auto& gpu : systemGPUs)
    {
        CUDA_CHECK(cudaSetDevice(gpu.DeviceId()));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

typename GPUSystemCUDA::GPUList*
GPUSystemCUDA::globalGPUListPtr = nullptr;

void GPUSystemCUDA::ThreadInitFunction()
{
    // Set all the devices on the thread, should be enough?
    for(const auto& device : (*globalGPUListPtr))
        CUDA_CHECK(cudaSetDevice(device.DeviceId()));
}

GPUThreadInitFunction GPUSystemCUDA::GetThreadInitFunction() const
{
    return &GPUSystemCUDA::ThreadInitFunction;
}

// Semaphore related namespace global functions
void TimelineSemAcquireInternal(void* params)
{
    GPUSemaphoreViewCUDA* ts = static_cast<GPUSemaphoreViewCUDA*>(params);
    // Device side acquision, we cant do much here,
    // because this is async, so we drop the result and on text iteration
    // GPU driving code may check the semaphore before sending an
    // acquisition code to GPU (a host launch)
    std::ignore = ts->HostAcquire();
}

void TimelineSemReleaseInternal(void* params)
{
    GPUSemaphoreViewCUDA* ts = static_cast<GPUSemaphoreViewCUDA*>(params);
    ts->HostRelease();
}

}