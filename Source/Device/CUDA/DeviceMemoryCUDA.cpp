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
{}

DeviceLocalMemoryCUDA::DeviceLocalMemoryCUDA(const GPUDeviceCUDA& device, size_t sizeInBytes)
    : DeviceLocalMemoryCUDA(device)
{
    assert(sizeInBytes != 0);
    size = sizeInBytes;

    CUmemAllocationProp props = {};
    props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    props.location.id = gpu->DeviceId();
    props.type = CU_MEM_ALLOCATION_TYPE_PINNED;

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
    : DeviceLocalMemoryCUDA(*(other.gpu), other.size)
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
{
    other.dPtr = nullptr;
    other.size = 0;
    other.allocSize = 0;
    other.memHandle = 0;
}

DeviceLocalMemoryCUDA& DeviceLocalMemoryCUDA::operator=(const DeviceLocalMemoryCUDA& other)
{
    assert(this != &other);

    // Allocate fresh via move assignment operator
    (*this) = DeviceLocalMemoryCUDA(*other.gpu, other.size);
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

    other.dPtr = nullptr;
    other.size = 0;
    other.allocSize = 0;
    other.memHandle = 0;


    return *this;
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

void DeviceLocalMemoryCUDA::ResizeBuffer(size_t newSize)
{
    // Do slow enlargement here, device local memory does not allocate
    // more than once. Device memory can be used for that instead
    DeviceLocalMemoryCUDA newMem(*gpu, newSize);

    // Copy to new memory
    size_t copySize = std::min(newSize, size);
    CUDA_DRIVER_CHECK(cuMemcpy(std::bit_cast<CUdeviceptr>(newMem.dPtr),
                               std::bit_cast<CUdeviceptr>(dPtr), copySize));
    *this = std::move(newMem);
}

size_t DeviceLocalMemoryCUDA::Size() const
{
    return size;
}

void DeviceLocalMemoryCUDA::MigrateToOtherDevice(const GPUDeviceCUDA& deviceTo)
{
    // Allocate over the other device
    DeviceLocalMemoryCUDA newMem(deviceTo, size);
    CUDA_CHECK(cudaMemcpyPeer(newMem.dPtr, newMem.gpu->DeviceId(),
                              dPtr, gpu->DeviceId(), size));

    *this = std::move(newMem);
}

HostLocalMemoryCUDA::HostLocalMemoryCUDA(const GPUSystemCUDA& system,
                                         bool neverDecrease)
    : system(&system)
    , hPtr(nullptr)
    , dPtr(nullptr)
    , size(0)
    , neverDecrease(neverDecrease)
{}

HostLocalMemoryCUDA::HostLocalMemoryCUDA(const GPUSystemCUDA& system,
                                         size_t sizeInBytes,
                                         bool neverDecrease)
    : HostLocalMemoryCUDA(system, neverDecrease)
{
    assert(sizeInBytes != 0);
    // TODO: change this to virtual memory calls as well
    CUDA_MEM_THROW(cudaHostAlloc(&hPtr, sizeInBytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&dPtr, hPtr, 0));
    size = sizeInBytes;
}

HostLocalMemoryCUDA::HostLocalMemoryCUDA(const HostLocalMemoryCUDA& other)
    : HostLocalMemoryCUDA(*other.system, other.size, other.neverDecrease)
{
    std::memcpy(hPtr, other.hPtr, size);
}

HostLocalMemoryCUDA::HostLocalMemoryCUDA(HostLocalMemoryCUDA&& other) noexcept
    : system(other.system)
    , hPtr(other.hPtr)
    , dPtr(other.dPtr)
    , size(other.size)
    , neverDecrease(other.neverDecrease)
{
    CUDA_CHECK(cudaFreeHost(hPtr));
    other.hPtr = nullptr;
    other.dPtr = nullptr;
    other.size = 0;
}

HostLocalMemoryCUDA& HostLocalMemoryCUDA::operator=(const HostLocalMemoryCUDA& other)
{
    assert(this != &other);

    size = other.size;
    neverDecrease = other.neverDecrease;
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
    neverDecrease = other.neverDecrease;

    other.size = 0;
    other.hPtr = nullptr;
    other.dPtr = nullptr;
    return *this;
}

HostLocalMemoryCUDA::~HostLocalMemoryCUDA()
{
    CUDA_CHECK(cudaFreeHost(hPtr));
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
    if(neverDecrease && newSize <= size) return;

    size_t copySize = std::min(newSize, size);
    HostLocalMemoryCUDA newMem(*system, newSize, neverDecrease);
    std::memcpy(newMem.hPtr, hPtr, copySize);
    *this = std::move(newMem);
}

size_t HostLocalMemoryCUDA::Size() const
{
    return size;
}

// Constructors & Destructor
HostLocalAlignedMemoryCUDA::HostLocalAlignedMemoryCUDA(const GPUSystemCUDA& systemIn,
                                                       size_t alignIn, bool ndIn)
    : system(&systemIn)
    , hPtr(nullptr)
    , dPtr(nullptr)
    , size(0)
    , allocSize(0)
    , alignment(alignIn)
    , neverDecrease(ndIn)
{}

HostLocalAlignedMemoryCUDA::HostLocalAlignedMemoryCUDA(const GPUSystemCUDA& systemIn,
                                                       size_t sizeInBytes, size_t alignIn,
                                                       bool ndIn)
    : HostLocalAlignedMemoryCUDA(systemIn, alignIn, ndIn)
{
    size = sizeInBytes;
    allocSize = MathFunctions::NextMultiple(sizeInBytes, alignment);
    alignment = alignIn;

    // Windows is hipster as always
    // does not have "std::aligned_alloc"
    // but have its own "_aligned_malloc" so using it.
    // To confuse it is also has its parameters swapped :)
    #ifdef MRAY_WINDOWS
        hPtr = _aligned_malloc(allocSize, alignment);
    #elif defined MRAY_LINUX
        hPtr = std::aligned_alloc(alignment, allocSize);
    #endif

    CUDA_CHECK(cudaHostRegister(hPtr, size, cudaHostRegisterMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&dPtr, hPtr, 0));
}

HostLocalAlignedMemoryCUDA::HostLocalAlignedMemoryCUDA(const HostLocalAlignedMemoryCUDA& other)
    : HostLocalAlignedMemoryCUDA(*other.system,
                                 other.size, other.alignment,
                                 other.neverDecrease)
{
    std::memcpy(hPtr, other.hPtr, size);
}

HostLocalAlignedMemoryCUDA::HostLocalAlignedMemoryCUDA(HostLocalAlignedMemoryCUDA&& other) noexcept
    : system(other.system)
    , hPtr(std::exchange(other.hPtr, nullptr))
    , dPtr(std::exchange(other.dPtr, nullptr))
    , size(other.size)
    , allocSize(other.allocSize)
    , alignment(other.alignment)
    , neverDecrease(other.neverDecrease)
{}

HostLocalAlignedMemoryCUDA& HostLocalAlignedMemoryCUDA::operator=(const HostLocalAlignedMemoryCUDA& other)
{
    // Utilize copy constructor + move assignment operator
    *this = HostLocalAlignedMemoryCUDA(other);
    return *this;
}

HostLocalAlignedMemoryCUDA& HostLocalAlignedMemoryCUDA::operator=(HostLocalAlignedMemoryCUDA&& other) noexcept
{
    assert(this != &other);
    if(size != 0)
    {
        CUDA_CHECK(cudaHostUnregister(hPtr));
        #ifdef MRAY_WINDOWS
            _aligned_free(hPtr);
        #elif defined MRAY_LINUX
            std::free(hPtr);
        #endif
    }

    system = other.system;
    hPtr = std::exchange(other.hPtr, nullptr);
    dPtr = std::exchange(other.dPtr, nullptr);
    size = other.size;
    allocSize = other.allocSize;
    alignment = other.alignment;
    neverDecrease = other.neverDecrease;
    return *this;
}

HostLocalAlignedMemoryCUDA::~HostLocalAlignedMemoryCUDA()
{
    if(hPtr != 0) CUDA_CHECK(cudaHostUnregister(hPtr));
    #ifdef MRAY_WINDOWS
        _aligned_free(hPtr);
    #elif defined MRAY_LINUX
        std::free(hPtr);
    #endif
}

Byte* HostLocalAlignedMemoryCUDA::DevicePtr()
{
    return static_cast<Byte*>(dPtr);
}

const Byte* HostLocalAlignedMemoryCUDA::DevicePtr() const
{
    return static_cast<const Byte*>(dPtr);
}

void HostLocalAlignedMemoryCUDA::ResizeBuffer(size_t newSize)
{
    if(neverDecrease && newSize <= size) return;

    size_t copySize = std::min(newSize, size);
    HostLocalAlignedMemoryCUDA newMem(*system, newSize, alignment, neverDecrease);
    std::memcpy(newMem.hPtr, hPtr, copySize);
    *this = std::move(newMem);
}

size_t HostLocalAlignedMemoryCUDA::Size() const
{
    return size;
}

size_t HostLocalAlignedMemoryCUDA::AllocSize() const
{
    return allocSize;
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
    curDeviceIndex = (curDeviceIndex + 1) % deviceIds.size();
    return curDeviceIndex;
}

DeviceMemoryCUDA::DeviceMemoryCUDA(const std::vector<const GPUDeviceCUDA*>& devices,
                                   size_t allocGranularity,
                                   size_t resGranularity,
                                   bool neverDecrease)
    : allocSize(0)
    , neverDecrease(neverDecrease)
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
    if(allocSize != 0) CUDA_DRIVER_CHECK(cuMemUnmap(mPtr, allocSize));
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
    if(!neverDecrease && newSize < allocSize)
    {
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
        assert(std::distance(it, allocs.crend()) >= 0);
        allocs.resize(static_cast<size_t>(std::distance(it, allocs.crend())));
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

    // Nothing to set access to
    if(allocSize == 0) return;

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