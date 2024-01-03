#include "DeviceMemoryCUDA.h"
#include "DefinitionsCUDA.h"
#include "GPUSystemCUDA.h"
#include "Core/System.h"
#include <algorithm>

namespace mray::cuda
{

DeviceLocalMemoryCUDA::DeviceLocalMemoryCUDA(const GPUDeviceCUDA& device)
    : gpu(&device)
    , dPtr(nullptr)
    , size(0)
    , allocSize(0)
    , memHandle(0)
    , isTexMappable(false)
{}

DeviceLocalMemoryCUDA::DeviceLocalMemoryCUDA(const GPUDeviceCUDA& device,
                                             size_t sizeInBytes,
                                             bool isUsedForTexMapping)
    : DeviceLocalMemoryCUDA(device)
{
    assert(sizeInBytes != 0);
    size = sizeInBytes;
    isTexMappable = isUsedForTexMapping;

    CUmemAllocationProp props = {};
    props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    props.location.id = gpu->DeviceId();
    props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    props.allocFlags.usage = (isTexMappable) ? CU_MEM_CREATE_USAGE_TILE_POOL
                                             : 0;

    size_t granularity;
    CUDA_DRIVER_CHECK(cuMemGetAllocationGranularity(&granularity, &props,
                                                    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    allocSize = MathFunctions::NextMultiple(size, granularity);
    CUDA_DRIVER_CHECK(cuMemCreate(&memHandle, allocSize, &props, 0));

    // Map to address space
    CUdeviceptr driverPtr;
    CUDA_DRIVER_MEM_THROW(cuMemAddressReserve(&driverPtr, allocSize, 0, 0, 0));
    CUDA_DRIVER_MEM_THROW(cuMemMap(driverPtr, allocSize, 0, memHandle, 0));
    dPtr = std::bit_cast<void*>(driverPtr);

    // Set access (since this is device local memory,
    // only the variable "gpu" can access it)
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = gpu->DeviceId();
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUDA_DRIVER_CHECK(cuMemSetAccess(driverPtr, allocSize, &accessDesc, 1));
}

DeviceLocalMemoryCUDA::DeviceLocalMemoryCUDA(const DeviceLocalMemoryCUDA& other)
    : DeviceLocalMemoryCUDA(*(other.gpu), other.size, other.isTexMappable)
{

    CUDA_CHECK(cudaSetDevice(gpu->DeviceId()));
    CUDA_DRIVER_CHECK(cuMemcpy(std::bit_cast<CUdeviceptr>(dPtr),
                               std::bit_cast<CUdeviceptr>(other.dPtr), size));
}

DeviceLocalMemoryCUDA::DeviceLocalMemoryCUDA(DeviceLocalMemoryCUDA&& other) noexcept
    : gpu(other.gpu)
    , dPtr(other.dPtr)
    , size(other.size)
    , allocSize(other.allocSize)
    , memHandle(other.memHandle)
    , isTexMappable(other.isTexMappable)
{
    other.dPtr = nullptr;
    other.size = 0;
    other.allocSize = 0;
    other.memHandle = 0;

}

DeviceLocalMemoryCUDA::~DeviceLocalMemoryCUDA()
{
    CUdeviceptr driverPtr = std::bit_cast<CUdeviceptr>(dPtr);
    if(allocSize != 0)
    {
        CUDA_DRIVER_CHECK(cuMemUnmap(driverPtr, allocSize));
        CUDA_DRIVER_CHECK(cuMemRelease(memHandle));
        CUDA_DRIVER_CHECK(cuMemAddressFree(driverPtr, allocSize));
    }
}

DeviceLocalMemoryCUDA& DeviceLocalMemoryCUDA::operator=(const DeviceLocalMemoryCUDA& other)
{
    assert(this != &other);

    // Allocate fresh via move assignment operator
    (*this) = DeviceLocalMemoryCUDA(*other.gpu, other.size, other.isTexMappable);

    //
    CUDA_DRIVER_CHECK(cuMemcpy(std::bit_cast<CUdeviceptr>(dPtr),
                               std::bit_cast<CUdeviceptr>(other.dPtr), size));
    return *this;
}

DeviceLocalMemoryCUDA& DeviceLocalMemoryCUDA::operator=(DeviceLocalMemoryCUDA&& other) noexcept
{
    assert(this != &other);
    // Remove old memory
    CUdeviceptr driverPtr = std::bit_cast<CUdeviceptr>(dPtr);
    if(allocSize != 0)
    {
        CUDA_DRIVER_CHECK(cuMemUnmap(driverPtr, allocSize));
        CUDA_DRIVER_CHECK(cuMemRelease(memHandle));
        CUDA_DRIVER_CHECK(cuMemAddressFree(driverPtr, allocSize));
    }

    allocSize = other.allocSize;
    gpu = other.gpu;
    dPtr = other.dPtr;
    size = other.size;
    memHandle = other.memHandle;
    isTexMappable = other.isTexMappable;

    other.dPtr = nullptr;
    other.size = 0;
    other.allocSize = 0;
    other.memHandle = 0;


    return *this;
}

void DeviceLocalMemoryCUDA::ResizeBuffer(size_t newSize)
{
    // Do slow enlargement here, device local memory does not allocate
    // more than once. Device memory can be used for that instead
    DeviceLocalMemoryCUDA newMem(*gpu, newSize, isTexMappable);

    // Copy to new memory
    size_t copySize = std::min(newSize, size);
    //CUDA_DRIVER_CHECK(cuMemcpy(std::bit_cast<CUdeviceptr>(newMem.dPtr),
    //                           std::bit_cast<CUdeviceptr>(dPtr), copySize));
    //std::memcpy(newMem.dPtr, dPtr, copySize);
    CUDA_CHECK(cudaSetDevice(gpu->DeviceId()));
    CUDA_CHECK(cudaMemcpy(newMem.dPtr, dPtr, copySize, cudaMemcpyDefault));

    *this = std::move(newMem);
}

size_t DeviceLocalMemoryCUDA::Size() const
{
    return size;
}

void DeviceLocalMemoryCUDA::MigrateToOtherDevice(const GPUDeviceCUDA& deviceTo)
{
    // Allocate over the other device
    DeviceLocalMemoryCUDA newMem(deviceTo, size, isTexMappable);
    CUDA_CHECK(cudaMemcpyPeer(newMem.dPtr, newMem.gpu->DeviceId(),
                              dPtr, gpu->DeviceId(), size));

    *this = std::move(newMem);
}

HostLocalMemoryCUDA::HostLocalMemoryCUDA(const GPUSystemCUDA& system)
    : system(system)
    , hPtr(nullptr)
    , dPtr(nullptr)
    , size(0)
{}

HostLocalMemoryCUDA::HostLocalMemoryCUDA(const GPUSystemCUDA& system,
                                         size_t sizeInBytes)
    : HostLocalMemoryCUDA(system)
{
    assert(sizeInBytes != 0);
    // TODO: change this to virtual memory calls as well
    CUDA_MEM_THROW(cudaHostAlloc(&hPtr, sizeInBytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&dPtr, hPtr, 0));

}

HostLocalMemoryCUDA::HostLocalMemoryCUDA(const HostLocalMemoryCUDA& other)
    : HostLocalMemoryCUDA(other.system, other.size)
{
    std::memcpy(hPtr, other.hPtr, size);
}

HostLocalMemoryCUDA::HostLocalMemoryCUDA(HostLocalMemoryCUDA&& other) noexcept
    : system(other.system)
    , hPtr(other.hPtr)
    , dPtr(other.dPtr)
    , size(other.size)
{
    other.hPtr = nullptr;
    other.dPtr = nullptr;
    other.size = 0;
}

HostLocalMemoryCUDA::~HostLocalMemoryCUDA()
{
    CUDA_CHECK(cudaFreeHost(hPtr));
}

HostLocalMemoryCUDA& HostLocalMemoryCUDA::operator=(const HostLocalMemoryCUDA& other)
{
    assert(this != &other);

    size = other.size;
    CUDA_CHECK(cudaFreeHost(hPtr));
    CUDA_MEM_THROW(cudaHostAlloc(&hPtr, size, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&dPtr, hPtr, 0));
    std::memcpy(hPtr, other.hPtr, size);
    return *this;
}

HostLocalMemoryCUDA& HostLocalMemoryCUDA::operator=(HostLocalMemoryCUDA&& other) noexcept
{
    assert(this != &other);
    size = other.size;
    hPtr = other.hPtr;
    dPtr = other.dPtr;

    other.size = 0;
    other.hPtr = 0;
    other.dPtr = 0;
    return *this;
}

Byte* HostLocalMemoryCUDA::DevicePtr()
{
    return reinterpret_cast<Byte*>(dPtr);
}

const Byte* HostLocalMemoryCUDA::DevicePtr() const
{
    return reinterpret_cast<const Byte*>(dPtr);
}

void HostLocalMemoryCUDA::ResizeBuffer(size_t newSize)
{
    HostLocalMemoryCUDA newMem(system, newSize);
    std::memcpy(newMem.hPtr, hPtr, size);
    *this = std::move(newMem);
}

size_t HostLocalMemoryCUDA::Size() const
{
    return size;
}

size_t DeviceMemoryCUDA::FindCommonGranularity() const
{
    // Determine a device common granularity
    size_t commonGranularity = 1;
    std::for_each(deviceIds.cbegin(), deviceIds.cend(),
                  [&](int deviceId)
    {
        CUmemAllocationProp props = {};
        props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = deviceId;
        props.type = CU_MEM_ALLOCATION_TYPE_PINNED;

        size_t devGranularity;
        CUDA_DRIVER_CHECK(cuMemGetAllocationGranularity(&devGranularity, &props,
                                                        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        // This is technically not correct
        commonGranularity = std::max(commonGranularity, devGranularity);
    });
    return commonGranularity;
}

size_t DeviceMemoryCUDA::NextDeviceIndex()
{
    curDeviceIndex = (curDeviceIndex + 1) % static_cast<int>(deviceIds.size());
    return curDeviceIndex;
}

DeviceMemoryCUDA::DeviceMemoryCUDA(const std::vector<const GPUDeviceCUDA*>& devices,
                                   size_t allocGranularity,
                                   size_t resGranularity)
    : allocSize(0)
{
    assert(resGranularity != 0);
    assert(allocGranularity != 0);
    assert(!devices.empty());

    deviceIds.reserve(devices.size());
    for(const GPUDeviceCUDA* dPtr : devices)
        deviceIds.push_back(dPtr->DeviceId());

    size_t commonGranularity = FindCommonGranularity();
    allocationGranularity = MathFunctions::NextMultiple(allocGranularity, commonGranularity);
    reserveGranularity = MathFunctions::NextMultiple(resGranularity, commonGranularity);

    reservedSize = reserveGranularity;
    CUDA_DRIVER_MEM_THROW(cuMemAddressReserve(&mPtr, reservedSize, 0, 0, 0));
    vaRanges.emplace_back(mPtr, reservedSize);
}

DeviceMemoryCUDA::~DeviceMemoryCUDA()
{
    CUDA_DRIVER_CHECK(cuMemUnmap(mPtr, allocSize));
    for(const Allocations& a : allocs)
    {
        CUDA_DRIVER_CHECK(cuMemRelease(a.handle));
    }
    for(const VARanges& vaRange : vaRanges)
        CUDA_DRIVER_CHECK(cuMemAddressFree(vaRange.first, vaRange.second));
}

void DeviceMemoryCUDA::ResizeBuffer(size_t newSize)
{
    // Align the newSize
    newSize = MathFunctions::NextMultiple(newSize, allocationGranularity);

    if(newSize > reservedSize)
    {
        size_t extraReserve = MathFunctions::NextMultiple(newSize, reserveGranularity);
        extraReserve -= reservedSize;

        // Try to allocate adjacent chunk
        CUdeviceptr newPtr = 0ULL;
        CUresult status = cuMemAddressReserve(&newPtr, extraReserve, 0,
                                              mPtr + reservedSize, 0);

        // Failed to allocate
        if((status != CUDA_SUCCESS) ||
           (newPtr != mPtr + reservedSize))
        {
            //MRAY_LOG("Complete Remap");
            size_t offset = 0;
            // Realloc
            if(allocSize != 0) CUDA_DRIVER_CHECK(cuMemUnmap(mPtr, allocSize));
            for(const VARanges& vaRange : vaRanges)
                CUDA_DRIVER_CHECK(cuMemAddressFree(vaRange.first, vaRange.second));
            // Get a green slate
            reservedSize = extraReserve + reservedSize;
            CUDA_DRIVER_MEM_THROW(cuMemAddressReserve(&newPtr, reservedSize, 0, 0, 0));
            // Remap the old memory
            for(const Allocations& a : allocs)
            {
                CUDA_DRIVER_MEM_THROW(cuMemMap(newPtr + offset, a.allocSize, 0, a.handle, 0));
                offset += a.allocSize;
            }
            mPtr = newPtr;
            vaRanges.clear();
            vaRanges.emplace_back(mPtr, reservedSize);
        }
        // We do manage to increase the reservation
        else
        {
            reservedSize = extraReserve + reservedSize;
            vaRanges.emplace_back(newPtr, extraReserve);
        }
    }
    // Shrink the memory
    if(newSize < allocSize)
    {
        //MRAY_LOG("Shrink");
        size_t offset = allocSize;
        auto it = allocs.crbegin();
        for(; it != allocs.crend(); it++)
        {
            offset -= allocationGranularity;
            if(newSize > offset)
                break;
            // Relase the memory
            CUDA_DRIVER_CHECK(cuMemUnmap(mPtr + offset, allocationGranularity));
            CUDA_DRIVER_CHECK(cuMemRelease(it->handle));
        }
        allocs.resize(std::distance(it, allocs.crend()));
        allocSize = offset + allocationGranularity;
    }
    // Enlarge the memory
    else if(newSize > allocSize)
    {
        size_t offset = allocSize;
        // Now calculate extra allocation
        size_t extraSize = newSize - allocSize;
        extraSize = MathFunctions::NextMultiple(extraSize, allocationGranularity);

        assert((extraSize % allocationGranularity) == 0);
        size_t newAllocCount = extraSize / allocationGranularity;

        // Add extra allocations
        for(size_t i = 0; i < newAllocCount; i++)
        {
            int deviceId = deviceIds[curDeviceIndex];
            CUmemAllocationProp props = {};
            props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            props.location.id = deviceId;
            props.type = CU_MEM_ALLOCATION_TYPE_PINNED;

            Allocations newAlloc =
            {
                .deviceId = deviceId,
                .handle = 0,
                .allocSize = allocationGranularity,
            };
            CUDA_DRIVER_CHECK(cuMemCreate(&newAlloc.handle, allocationGranularity, &props, 0));
            CUDA_DRIVER_MEM_THROW(cuMemMap(mPtr + offset, allocationGranularity, 0, newAlloc.handle, 0));
            offset += allocationGranularity;

            allocs.push_back(newAlloc);
            NextDeviceIndex();
        }
        assert(offset == newSize);
        allocSize = newSize;
    }
    // Set the newly mapped range accessor
    std::vector<CUmemAccessDesc> accessDescriptions(deviceIds.size(),
                                                    CUmemAccessDesc{});
    for(size_t i = 0; i < accessDescriptions.size(); i++)
    {
        CUmemAccessDesc& ad = accessDescriptions[i];
        ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        ad.location.id = deviceIds[i];
        ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    CUDA_DRIVER_CHECK(cuMemSetAccess(mPtr, allocSize,
                                     accessDescriptions.data(),
                                     accessDescriptions.size()));
}

size_t DeviceMemoryCUDA::Size() const
{
    return allocSize;
}

}