//#include "DeviceMemory.h"
//#include "RayLib/CudaCheck.h"
//#include "CudaSystem.h"
//
//#include <cuda_runtime.h>
//#include <cstring>
//#include <algorithm>
//#include <cstddef>
//
//DeviceLocalMemory::DeviceLocalMemory(const CudaGPU* device)
//    : DeviceLocalMemoryI(device)
//    , d_ptr(nullptr)
//    , size(0)
//{}
//
//DeviceLocalMemory::DeviceLocalMemory(const CudaGPU* device, size_t sizeInBytes)
//    : DeviceLocalMemoryI(device)
//    , size(sizeInBytes)
//{
//    CUDA_CHECK(cudaSetDevice(currentDevice->DeviceId()));
//    CUDA_MEMORY_CHECK(cudaMalloc(&d_ptr, size));
//}
//
//DeviceLocalMemory::DeviceLocalMemory(const DeviceLocalMemory& other)
//    : DeviceLocalMemory(other.currentDevice, other.size)
//{
//    CUDA_CHECK(cudaSetDevice(other.currentDevice->DeviceId()));
//    CUDA_CHECK(cudaMemcpy(d_ptr, other.d_ptr, size, cudaMemcpyDeviceToDevice));
//}
//
//DeviceLocalMemory::DeviceLocalMemory(DeviceLocalMemory&& other) noexcept
//    : DeviceLocalMemoryI(other.currentDevice)
//    , d_ptr(other.d_ptr)
//    , size(other.size)
//{
//    other.d_ptr = nullptr;
//}
//
//DeviceLocalMemory::~DeviceLocalMemory()
//{
//    CUDA_CHECK(cudaSetDevice(currentDevice->DeviceId()));
//    CUDA_CHECK(cudaFree(d_ptr));
//}
//
//DeviceLocalMemory& DeviceLocalMemory::operator=(const DeviceLocalMemory& other)
//{
//    assert(this != &other);
//
//    // Determine Device
//    CUDA_CHECK(cudaSetDevice(currentDevice->DeviceId()));
//    // Realloc if necessary
//    if(size != other.size)
//    {
//        CUDA_CHECK(cudaFree(d_ptr));
//        CUDA_MEMORY_CHECK(cudaMalloc(&d_ptr, other.size));
//    }
//
//    // Copy Device Data
//    // Check memory location
//    if(currentDevice == other.currentDevice)
//    {
//        CUDA_CHECK(cudaMemcpy(d_ptr, other.d_ptr, other.size, cudaMemcpyDeviceToDevice));
//    }
//    else
//    {
//        CUDA_CHECK(cudaMemcpyPeer(d_ptr, currentDevice->DeviceId(), other.d_ptr,
//                                  other.currentDevice->DeviceId(), other.size));
//    }
//    size = other.size;
//    return *this;
//}
//
//DeviceLocalMemory& DeviceLocalMemory::operator=(DeviceLocalMemory&& other) noexcept
//{
//    assert(this != &other);
//    size = other.size;
//
//    // Determine Device
//    CUDA_CHECK(cudaSetDevice(currentDevice->DeviceId()));
//    CUDA_CHECK(cudaFree(d_ptr));
//
//    // Move Device Data
//    // Check memory location
//    if(currentDevice == other.currentDevice)
//    {
//        // Same Device you can assign ptrs
//        d_ptr = other.d_ptr;
//        other.d_ptr = nullptr;
//    }
//    else
//    {
//        // Between devices fallback to copy and free
//        CUDA_MEMORY_CHECK(cudaMalloc(&d_ptr, size));
//        CUDA_CHECK(cudaMemcpyPeer(d_ptr, currentDevice->DeviceId(), other.d_ptr,
//                                  other.currentDevice->DeviceId(), other.size));
//
//        // Remove memory from other device
//        CUDA_CHECK(cudaSetDevice(other.currentDevice->DeviceId()));
//        CUDA_CHECK(cudaFree(other.d_ptr));
//    }
//    currentDevice = other.currentDevice;
//    return *this;
//}
//
//size_t DeviceLocalMemory::Size() const
//{
//    return size;
//}
//
//void DeviceLocalMemory::MigrateToOtherDevice(const CudaGPU* deviceTo, cudaStream_t stream)
//{
//    void* d_new = nullptr;
//    CUDA_CHECK(cudaSetDevice(deviceTo->DeviceId()));
//    CUDA_MEMORY_CHECK(cudaMalloc(&d_new, size));
//    CUDA_CHECK(cudaMemcpyPeerAsync(d_new, deviceTo->DeviceId(), d_ptr,
//                                   currentDevice->DeviceId(), size, stream));
//    CUDA_CHECK(cudaSetDevice(currentDevice->DeviceId()));
//    CUDA_CHECK(cudaFree(d_ptr));
//    d_ptr = d_new;
//    currentDevice = deviceTo;
//}
//
//DeviceMemoryCPUBacked::DeviceMemoryCPUBacked(const CudaGPU* device)
//    : DeviceLocalMemoryI(device)
//    , h_ptr(nullptr)
//    , d_ptr(nullptr)
//    , size(0)
//{}
//
//DeviceMemoryCPUBacked::DeviceMemoryCPUBacked(const CudaGPU* device, size_t sizeInBytes)
//    : DeviceLocalMemoryI(device)
//    , size(sizeInBytes)
//{
//    CUDA_CHECK(cudaSetDevice(currentDevice->DeviceId()));
//    CUDA_MEMORY_CHECK(cudaMalloc(&d_ptr, size));
//    h_ptr = malloc(size);
//}
//
//DeviceMemoryCPUBacked::DeviceMemoryCPUBacked(const DeviceMemoryCPUBacked& other)
//    : DeviceMemoryCPUBacked(other.currentDevice, other.size)
//{
//    CUDA_CHECK(cudaMemcpy(d_ptr, other.d_ptr, size, cudaMemcpyDeviceToDevice));
//    std::memcpy(h_ptr, other.h_ptr, size);
//}
//
//DeviceMemoryCPUBacked::DeviceMemoryCPUBacked(DeviceMemoryCPUBacked&& other)
//    : DeviceLocalMemoryI(other.currentDevice)
//    , h_ptr(other.h_ptr)
//    , d_ptr(other.d_ptr)
//    , size(other.size)
//{
//    other.h_ptr = nullptr;
//    other.d_ptr = nullptr;
//}
//
//DeviceMemoryCPUBacked::~DeviceMemoryCPUBacked()
//{
//    CUDA_CHECK(cudaSetDevice(currentDevice->DeviceId()));
//    CUDA_CHECK(cudaFree(d_ptr));
//    free(h_ptr);
//}
//
//DeviceMemoryCPUBacked& DeviceMemoryCPUBacked::operator=(const DeviceMemoryCPUBacked& other)
//{
//    assert(this != &other);
//
//    // Determine Device
//    CUDA_CHECK(cudaSetDevice(currentDevice->DeviceId()));
//
//    // Realloc if necessary
//    if(size != other.size)
//    {
//        CUDA_CHECK(cudaFree(d_ptr));
//        CUDA_MEMORY_CHECK(cudaMalloc(&d_ptr, other.size));
//
//        free(h_ptr);
//        h_ptr = malloc(size);
//    }
//
//    // Copy Host Data
//    std::memcpy(h_ptr, other.h_ptr, size);
//
//    // Copy Device Data
//    // Check memory location
//    if(currentDevice == other.currentDevice)
//    {
//        CUDA_CHECK(cudaMemcpy(d_ptr, other.d_ptr, other.size, cudaMemcpyDeviceToDevice));
//    }
//    else
//    {
//        CUDA_CHECK(cudaMemcpyPeer(d_ptr, currentDevice->DeviceId(),
//                                  other.d_ptr, other.currentDevice->DeviceId(),
//                                  other.size));
//    }
//    size = other.size;
//    return *this;
//}
//
//DeviceMemoryCPUBacked& DeviceMemoryCPUBacked::operator=(DeviceMemoryCPUBacked&& other)
//{
//    assert(this != &other);
//    size = other.size;
//
//    // Determine Device
//    CUDA_CHECK(cudaSetDevice(currentDevice->DeviceId()));
//    CUDA_CHECK(cudaFree(d_ptr));
//    free(h_ptr);
//
//    // Move host data
//    h_ptr = other.h_ptr;
//    other.h_ptr = nullptr;
//
//    // Move Device Data
//    // Check memory location
//    if(currentDevice == other.currentDevice)
//    {
//        // Same Device you can assign ptrs
//        d_ptr = other.d_ptr;
//        other.d_ptr = nullptr;
//    }
//    else
//    {
//        // Between devices fallback to copy and free
//        CUDA_MEMORY_CHECK(cudaMalloc(&d_ptr, size));
//        CUDA_CHECK(cudaMemcpyPeer(d_ptr, currentDevice->DeviceId(),
//                                  other.d_ptr, other.currentDevice->DeviceId(),
//                                  other.size));
//
//        // Remove memory from other device
//        CUDA_CHECK(cudaSetDevice(other.currentDevice->DeviceId()));
//        CUDA_CHECK(cudaFree(other.d_ptr));
//    }
//    return *this;
//}
//
//void DeviceMemoryCPUBacked::CopyToDevice(size_t offset, size_t copySize, cudaStream_t stream)
//{
//    copySize = std::max(size, copySize);
//    assert(offset + copySize <= size);
//
//    // Determine Device
//    CUDA_CHECK(cudaSetDevice(currentDevice->DeviceId()));
//    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char*>(d_ptr) + offset,
//                               reinterpret_cast<char*>(h_ptr) + offset,
//                               copySize, cudaMemcpyHostToDevice, stream));
//}
//
//void DeviceMemoryCPUBacked::CopyToHost(size_t offset, size_t copySize, cudaStream_t stream)
//{
//    copySize = std::max(size, copySize);
//    assert(offset + copySize <= size);
//
//    // Determine Device
//    CUDA_CHECK(cudaSetDevice(currentDevice->DeviceId()));
//    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char*>(h_ptr) + offset,
//                               reinterpret_cast<char*>(d_ptr) + offset,
//                               copySize, cudaMemcpyDeviceToHost, stream));
//}
//
//size_t DeviceMemoryCPUBacked::Size() const
//{
//    return size;
//}
//
//void DeviceMemoryCPUBacked::MigrateToOtherDevice(const CudaGPU* deviceTo, cudaStream_t stream)
//{
//    void* d_new = nullptr;
//    CUDA_CHECK(cudaSetDevice(deviceTo->DeviceId()));
//    CUDA_MEMORY_CHECK(cudaMalloc(&d_new, size));
//    CUDA_CHECK(cudaMemcpyPeerAsync(d_new, deviceTo->DeviceId(), d_ptr,
//                                   currentDevice->DeviceId(), size, stream));
//    CUDA_CHECK(cudaSetDevice(currentDevice->DeviceId()));
//    CUDA_CHECK(cudaFree(d_ptr));
//    d_ptr = d_new;
//    currentDevice = deviceTo;
//}
//
//DeviceMemory::DeviceMemory()
//    : m_ptr(nullptr)
//    , size(0)
//{}
//
//DeviceMemory::DeviceMemory(size_t sizeInBytes)
//    : size(sizeInBytes)
//{
//    CUDA_MEMORY_CHECK(cudaMallocManaged(&m_ptr, size));
//}
//
//DeviceMemory::DeviceMemory(const DeviceMemory& other)
//    : DeviceMemory(other.size)
//{
//    std::memcpy(m_ptr, other.m_ptr, size);
//}
//
//DeviceMemory::DeviceMemory(DeviceMemory&& other)
//    : m_ptr(other.m_ptr)
//    , size(other.size)
//{
//    other.m_ptr = nullptr;
//}
//
//DeviceMemory::~DeviceMemory()
//{
//    CUDA_CHECK(cudaFree(m_ptr));
//}
//
//DeviceMemory& DeviceMemory::operator=(const DeviceMemory& other)
//{
//    if(size != other.size)
//    {
//        CUDA_CHECK(cudaFree(m_ptr));
//        CUDA_MEMORY_CHECK(cudaMallocManaged(&m_ptr, other.size));
//    }
//    CUDA_CHECK(cudaMemcpy(m_ptr, other.m_ptr, other.size, cudaMemcpyDeviceToDevice));
//    size = other.size;
//    return *this;
//}
//
//DeviceMemory& DeviceMemory::operator=(DeviceMemory&& other)
//{
//    assert(this != &other);
//    CUDA_CHECK(cudaFree(m_ptr));
//    m_ptr = other.m_ptr;
//    size = other.size;
//    other.m_ptr = nullptr;
//    return *this;
//}
//
//size_t DeviceMemory::Size() const
//{
//    return size;
//}