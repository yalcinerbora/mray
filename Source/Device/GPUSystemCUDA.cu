#include "GPUSystem.h"
#include "DeviceMemoryCUDA.h"
#include <cuda.h>

namespace mray::cuda
{

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCUDA::GPUQueueCUDA(uint32_t multiprocessorCount,
                           DeviceQueueType t)
    : multiprocessorCount(multiprocessorCount)
{
    switch(t)
    {
        case DeviceQueueType::NORMAL:
            CUDA_CHECK(cudaStreamCreate(&stream));
            break;
        case DeviceQueueType::FIRE_AND_FORGET:
            stream = cudaStreamFireAndForget;
            break;
        case DeviceQueueType::TAIL_LAUNCH:
            stream = cudaStreamTailLaunch;
            break;
        default:
            assert(false);
            break;
    }
}

MRAY_HYBRID MRAY_CGPU_INLINE
GPUQueueCUDA::~GPUQueueCUDA()
{
    if(stream != cudaStreamTailLaunch ||
       stream != cudaStreamFireAndForget ||
       stream != (cudaStream_t)0)
    {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

GPUDeviceCUDA::GPUDeviceCUDA(int deviceId)
    : deviceId(deviceId)
{
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
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

const GPUQueue& GPUDeviceCUDA::GetQueue(uint32_t index) const
{
    return queues[index];
}

DeviceLocalMemoryCUDA GPUDeviceCUDA::GetAMemory() const
{
    return DeviceLocalMemoryCUDA(*this);
}

//template<uint32_t DIMS, class T>
//DeviceTextureCUDA<DIMS, T> GPUDeviceCUDA::GetTexture() const
//{
//
//}

GPUSystemCUDA::GPUSystemCUDA()
{
    // Initialize the CUDA
    CUDA_CHECK(cudaFree(nullptr));

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
        systemGPUs.emplace_back(i);
    }

    // TODO: Do topology stuff here
    // handle selection etc. this is too
    // primitive currently
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
        uint32_t blockPerSM = GPUQueueCUDA::RecommendedBlockCountPerSM(kernelPtr,
                                                                       threadCount,
                                                                       sharedMemSize);
        uint32_t blockGPU = blockPerSM * g.SMCount();
        workPerGPU.push_back(blockGPU);
        totalAvailBlocks += blockGPU;
    }

    // Total Threads
    uint32_t totalThreads = threadCount * totalAvailBlocks;
    uint32_t iterationPerThread = MathFunctions::AlignToMultiple(workCount, totalThreads);

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

DeviceMemoryCUDA GPUSystemCUDA::GetAMemory() const
{
    return DeviceMemory(*this);
}

}