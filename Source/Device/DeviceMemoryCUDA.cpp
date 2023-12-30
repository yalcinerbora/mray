#include "DeviceMemoryCUDA.h"
#include "DefinitionsCUDA.h"
#include "GPUSystemCUDA.h"

namespace mray::cuda
{

DeviceLocalMemoryCUDA::DeviceLocalMemoryCUDA(const GPUDeviceCUDA& device)
    : dPtr(nullptr)
    , size(0)
    , gpu(&device)
{}

DeviceLocalMemoryCUDA::DeviceLocalMemoryCUDA(const GPUDeviceCUDA& device, size_t sizeInBytes)
    : dPtr(nullptr)
    , size(sizeInBytes)
    , gpu(&device)
{
    CUDA_CHECK(cudaSetDevice(gpu->DeviceId()));
    CUDA_MEM_CHECK(cudaMalloc(&dPtr, size));
}

DeviceLocalMemoryCUDA::DeviceLocalMemoryCUDA(const DeviceLocalMemoryCUDA& other)
    : DeviceLocalMemoryCUDA(*(other.gpu), other.size)
{
    CUDA_CHECK(cudaSetDevice(other.gpu->DeviceId()));
    CUDA_CHECK(cudaMemcpy(dPtr, other.dPtr, size,
                          cudaMemcpyDeviceToDevice));
}

DeviceLocalMemoryCUDA::DeviceLocalMemoryCUDA(DeviceLocalMemoryCUDA&& other) noexcept
    : dPtr(other.dPtr)
    , size(other.size)
    , gpu(other.gpu)
{
    other.dPtr = nullptr;
}

DeviceLocalMemoryCUDA::~DeviceLocalMemoryCUDA()
{
    CUDA_CHECK(cudaSetDevice(gpu->DeviceId()));
    CUDA_CHECK(cudaFree(dPtr));
}

DeviceLocalMemoryCUDA& DeviceLocalMemoryCUDA::operator=(const DeviceLocalMemoryCUDA& other)
{
    assert(this != &other);

    // Determine Device
    CUDA_CHECK(cudaSetDevice(gpu->DeviceId()));
    // Realloc if necessary
    if(size != other.size)
    {
        CUDA_CHECK(cudaFree(dPtr));
        CUDA_MEM_CHECK(cudaMalloc(&dPtr, other.size));
    }

    // Copy Device Data
    // Check memory location
    if((*gpu) == (*other.gpu))
    {
        CUDA_CHECK(cudaMemcpy(dPtr, other.dPtr, other.size,
                              cudaMemcpyDeviceToDevice));
    }
    else
    {
        CUDA_CHECK(cudaMemcpyPeer(dPtr, gpu->DeviceId(), other.dPtr,
                                  other.gpu->DeviceId(), other.size));
    }
    size = other.size;
    return *this;
}

DeviceLocalMemoryCUDA& DeviceLocalMemoryCUDA::operator=(DeviceLocalMemoryCUDA&& other) noexcept
{
    assert(this != &other);
    size = other.size;

    // Determine Device
    CUDA_CHECK(cudaSetDevice(gpu->DeviceId()));
    CUDA_CHECK(cudaFree(dPtr));

    // Move Device Data
    // Check memory location
    if(gpu == other.gpu)
    {
        // Same Device you can assign ptrs
        dPtr = other.dPtr;
        other.dPtr = nullptr;
    }
    else
    {
        // Between devices fallback to copy and free
        CUDA_MEM_CHECK(cudaMalloc(&dPtr, size));
        CUDA_CHECK(cudaMemcpyPeer(dPtr, gpu->DeviceId(), other.dPtr,
                                  other.gpu->DeviceId(), other.size));

        // Remove memory from other device
        CUDA_CHECK(cudaSetDevice(other.gpu->DeviceId()));
        CUDA_CHECK(cudaFree(other.dPtr));
    }
    gpu = other.gpu;
    return *this;
}

void DeviceLocalMemoryCUDA::EnlargeBuffer(size_t newSize)
{
    // There is no good way to enlarge the buffer
    // Alloc -> Copy -> Free
    CUDA_CHECK(cudaSetDevice(gpu->DeviceId()));

    void* newPtr;
    CUDA_MEM_CHECK(cudaMalloc(&newPtr, newSize));
    CUDA_CHECK(cudaMemcpy(newPtr, dPtr, size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(dPtr));
    dPtr = newPtr;
}

size_t DeviceLocalMemoryCUDA::Size() const
{
    return size;
}

void DeviceLocalMemoryCUDA::MigrateToOtherDevice(const GPUDeviceCUDA& deviceTo)
{
    void* dNew = nullptr;
    CUDA_CHECK(cudaSetDevice(deviceTo.DeviceId()));
    CUDA_MEM_CHECK(cudaMalloc(&dNew, size));
    CUDA_CHECK(cudaMemcpyPeer(dNew, deviceTo.DeviceId(), dPtr,
                                   gpu->DeviceId(), size));
    CUDA_CHECK(cudaSetDevice(gpu->DeviceId()));
    CUDA_CHECK(cudaFree(dPtr));
    dPtr = dNew;
    gpu = &deviceTo;
}

DeviceMemoryCUDA::DeviceMemoryCUDA(const GPUSystemCUDA& sys)
    : mPtr(nullptr)
    , size(0)
    , system(sys)
{}

DeviceMemoryCUDA::DeviceMemoryCUDA(const GPUSystemCUDA& sys, size_t sizeInBytes)
    : size(sizeInBytes)
    , system(sys)
{
    CUDA_MEM_CHECK(cudaMallocManaged(&mPtr, size));
}

DeviceMemoryCUDA::DeviceMemoryCUDA(const DeviceMemoryCUDA& other)
    : DeviceMemoryCUDA(other.system, other.size)
{
    // TODO?????
    std::memcpy(mPtr, other.mPtr, size);
}

DeviceMemoryCUDA::DeviceMemoryCUDA(DeviceMemoryCUDA&& other) noexcept
    : mPtr(other.mPtr)
    , size(other.size)
    , system(other.system)
{
    other.mPtr = nullptr;
}

DeviceMemoryCUDA::~DeviceMemoryCUDA()
{
    CUDA_CHECK(cudaFree(mPtr));
}

DeviceMemoryCUDA& DeviceMemoryCUDA::operator=(const DeviceMemoryCUDA& other)
{
    if(size != other.size)
    {
        CUDA_CHECK(cudaFree(mPtr));
        CUDA_MEM_CHECK(cudaMallocManaged(&mPtr, other.size));
    }
    CUDA_CHECK(cudaMemcpy(mPtr, other.mPtr, other.size,
                          cudaMemcpyDeviceToDevice));
    size = other.size;
    return *this;
}

DeviceMemoryCUDA& DeviceMemoryCUDA::operator=(DeviceMemoryCUDA&& other) noexcept
{
    assert(this != &other);
    CUDA_CHECK(cudaFree(mPtr));
    mPtr = other.mPtr;
    size = other.size;
    other.mPtr = nullptr;
    return *this;
}

size_t DeviceMemoryCUDA::Size() const
{
    return size;
}

void DeviceMemoryCUDA::EnlargeBuffer(size_t newSize)
{

}

}