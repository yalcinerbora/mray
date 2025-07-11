#include "DeviceMemoryHIP.h"
#include "DefinitionsHIP.h"
#include "GPUSystemHIP.h"
#include "Core/System.h"
#include <algorithm>

hipDeviceptr_t AdvanceHIPPtr(hipDeviceptr_t dPtr, size_t offset)
{
    return reinterpret_cast<Byte*>(dPtr) + offset;
}

namespace mray::hip
{

DeviceLocalMemoryHIP::DeviceLocalMemoryHIP(const GPUDeviceHIP& device)
    : gpu(&device)
    , dPtr(nullptr)
    , size(0)
    , allocSize(0)
    , memHandle(0)
{}

DeviceLocalMemoryHIP::DeviceLocalMemoryHIP(const GPUDeviceHIP& device, size_t sizeInBytes)
    : DeviceLocalMemoryHIP(device)
{
    assert(sizeInBytes != 0);
    size = sizeInBytes;

    hipMemAllocationProp props = {};
    props.location.type = hipMemLocationTypeDevice;
    props.location.id = gpu->DeviceId();
    props.type = hipMemAllocationTypePinned;

    size_t granularity;
    HIP_DRIVER_CHECK(hipMemGetAllocationGranularity(&granularity, &props,
                                                    hipMemAllocationGranularityRecommended));
    allocSize = Math::NextMultiple(size, granularity);
    HIP_DRIVER_CHECK(hipMemCreate(&memHandle, allocSize, &props, 0));

    // Map to address space
    hipDeviceptr_t driverPtr;
    HIP_DRIVER_MEM_THROW(hipMemAddressReserve(&driverPtr, allocSize, 0, 0, 0));
    HIP_DRIVER_MEM_THROW(hipMemMap(driverPtr, allocSize, 0, memHandle, 0));
    dPtr = std::bit_cast<void*>(driverPtr);

    // Set access (since this is device local memory,
    // only the variable "gpu" can access it)
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = gpu->DeviceId();
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_DRIVER_CHECK(hipMemSetAccess(driverPtr, allocSize, &accessDesc, 1));
}

DeviceLocalMemoryHIP::DeviceLocalMemoryHIP(const DeviceLocalMemoryHIP& other)
    : DeviceLocalMemoryHIP(*(other.gpu), other.size)
{

    HIP_CHECK(hipSetDevice(gpu->DeviceId()));
    HIP_DRIVER_CHECK(hipMemcpy(std::bit_cast<hipDeviceptr_t>(dPtr),
                               std::bit_cast<hipDeviceptr_t>(other.dPtr), size,
                               hipMemcpyDefault));
}

DeviceLocalMemoryHIP::DeviceLocalMemoryHIP(DeviceLocalMemoryHIP&& other) noexcept
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

DeviceLocalMemoryHIP& DeviceLocalMemoryHIP::operator=(const DeviceLocalMemoryHIP& other)
{
    assert(this != &other);

    // Allocate fresh via move assignment operator
    (*this) = DeviceLocalMemoryHIP(*other.gpu, other.size);
    HIP_DRIVER_CHECK(hipMemcpy(std::bit_cast<hipDeviceptr_t>(dPtr),
                               std::bit_cast<hipDeviceptr_t>(other.dPtr), size,
                               hipMemcpyDefault));
    return *this;
}

DeviceLocalMemoryHIP& DeviceLocalMemoryHIP::operator=(DeviceLocalMemoryHIP&& other) noexcept
{
    assert(this != &other);
    // Remove old memory
    hipDeviceptr_t driverPtr = std::bit_cast<hipDeviceptr_t>(dPtr);
    if(allocSize != 0)
    {
        HIP_DRIVER_CHECK(hipMemUnmap(driverPtr, allocSize));
        HIP_DRIVER_CHECK(hipMemRelease(memHandle));
        HIP_DRIVER_CHECK(hipMemAddressFree(driverPtr, allocSize));
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

DeviceLocalMemoryHIP::~DeviceLocalMemoryHIP()
{
    hipDeviceptr_t driverPtr = std::bit_cast<hipDeviceptr_t>(dPtr);
    if(allocSize != 0)
    {
        HIP_DRIVER_CHECK(hipMemUnmap(driverPtr, allocSize));
        HIP_DRIVER_CHECK(hipMemRelease(memHandle));
        HIP_DRIVER_CHECK(hipMemAddressFree(driverPtr, allocSize));
    }
}

void DeviceLocalMemoryHIP::ResizeBuffer(size_t newSize)
{
    // Do slow enlargement here, device local memory does not allocate
    // more than once. Device memory can be used for that instead
    DeviceLocalMemoryHIP newMem(*gpu, newSize);

    // Copy to new memory
    size_t copySize = std::min(newSize, size);
    HIP_DRIVER_CHECK(hipMemcpy(std::bit_cast<hipDeviceptr_t>(newMem.dPtr),
                               std::bit_cast<hipDeviceptr_t>(dPtr), copySize,
                               hipMemcpyDefault));
    *this = std::move(newMem);
}

size_t DeviceLocalMemoryHIP::Size() const
{
    return size;
}

void DeviceLocalMemoryHIP::MigrateToOtherDevice(const GPUDeviceHIP& deviceTo)
{
    // Allocate over the other device
    DeviceLocalMemoryHIP newMem(deviceTo, size);
    HIP_CHECK(hipMemcpyPeer(newMem.dPtr, newMem.gpu->DeviceId(),
                            dPtr, gpu->DeviceId(), size));

    *this = std::move(newMem);
}

HostLocalMemoryHIP::HostLocalMemoryHIP(const GPUSystemHIP& system,
                                       bool neverDecrease)
    : system(&system)
    , hPtr(nullptr)
    , dPtr(nullptr)
    , size(0)
    , neverDecrease(neverDecrease)
{}

HostLocalMemoryHIP::HostLocalMemoryHIP(const GPUSystemHIP& system,
                                       size_t sizeInBytes,
                                       bool neverDecrease)
    : HostLocalMemoryHIP(system, neverDecrease)
{
    assert(sizeInBytes != 0);
    // TODO: change this to virtual memory calls as well
    HIP_MEM_THROW(hipHostMalloc(&hPtr, sizeInBytes, hipHostMallocMapped));
    HIP_CHECK(hipHostGetDevicePointer(&dPtr, hPtr, 0));
    size = sizeInBytes;
}

HostLocalMemoryHIP::HostLocalMemoryHIP(const HostLocalMemoryHIP& other)
    : HostLocalMemoryHIP(*other.system, other.size, other.neverDecrease)
{
    std::memcpy(hPtr, other.hPtr, size);
}

HostLocalMemoryHIP::HostLocalMemoryHIP(HostLocalMemoryHIP&& other) noexcept
    : system(other.system)
    , hPtr(other.hPtr)
    , dPtr(other.dPtr)
    , size(other.size)
    , neverDecrease(other.neverDecrease)
{
    HIP_CHECK(hipHostFree(hPtr));
    other.hPtr = nullptr;
    other.dPtr = nullptr;
    other.size = 0;
}

HostLocalMemoryHIP& HostLocalMemoryHIP::operator=(const HostLocalMemoryHIP& other)
{
    assert(this != &other);

    size = other.size;
    neverDecrease = other.neverDecrease;
    HIP_CHECK(hipHostFree(hPtr));
    HIP_MEM_THROW(hipHostMalloc(&hPtr, size, hipHostMallocMapped));
    HIP_CHECK(hipHostGetDevicePointer(&dPtr, hPtr, 0));
    std::memcpy(hPtr, other.hPtr, size);
    return *this;
}

HostLocalMemoryHIP& HostLocalMemoryHIP::operator=(HostLocalMemoryHIP&& other) noexcept
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

HostLocalMemoryHIP::~HostLocalMemoryHIP()
{
    HIP_CHECK(hipHostFree(hPtr));
}

Byte* HostLocalMemoryHIP::DevicePtr()
{
    return reinterpret_cast<Byte*>(dPtr);
}

const Byte* HostLocalMemoryHIP::DevicePtr() const
{
    return reinterpret_cast<const Byte*>(dPtr);
}

void HostLocalMemoryHIP::ResizeBuffer(size_t newSize)
{
    if(neverDecrease && newSize <= size) return;

    size_t copySize = std::min(newSize, size);
    HostLocalMemoryHIP newMem(*system, newSize, neverDecrease);
    std::memcpy(newMem.hPtr, hPtr, copySize);
    *this = std::move(newMem);
}

size_t HostLocalMemoryHIP::Size() const
{
    return size;
}

// Constructors & Destructor
HostLocalAlignedMemoryHIP::HostLocalAlignedMemoryHIP(const GPUSystemHIP& systemIn,
                                                     size_t alignIn, bool ndIn)
    : system(&systemIn)
    , hPtr(nullptr)
    , dPtr(nullptr)
    , size(0)
    , allocSize(0)
    , alignment(alignIn)
    , neverDecrease(ndIn)
{}

HostLocalAlignedMemoryHIP::HostLocalAlignedMemoryHIP(const GPUSystemHIP& systemIn,
                                                     size_t sizeInBytes, size_t alignIn,
                                                     bool ndIn)
    : HostLocalAlignedMemoryHIP(systemIn, alignIn, ndIn)
{
    size = sizeInBytes;
    allocSize = Math::NextMultiple(sizeInBytes, alignment);
    alignment = alignIn;
    hPtr = AlignedAlloc(alignment, allocSize);
    HIP_CHECK(hipHostRegister(hPtr, size, hipHostRegisterMapped));
    HIP_CHECK(hipHostGetDevicePointer(&dPtr, hPtr, 0));
}

HostLocalAlignedMemoryHIP::HostLocalAlignedMemoryHIP(const HostLocalAlignedMemoryHIP& other)
    : HostLocalAlignedMemoryHIP(*other.system,
                                 other.size, other.alignment,
                                 other.neverDecrease)
{
    std::memcpy(hPtr, other.hPtr, size);
}

HostLocalAlignedMemoryHIP::HostLocalAlignedMemoryHIP(HostLocalAlignedMemoryHIP&& other) noexcept
    : system(other.system)
    , hPtr(std::exchange(other.hPtr, nullptr))
    , dPtr(std::exchange(other.dPtr, nullptr))
    , size(other.size)
    , allocSize(other.allocSize)
    , alignment(other.alignment)
    , neverDecrease(other.neverDecrease)
{}

HostLocalAlignedMemoryHIP& HostLocalAlignedMemoryHIP::operator=(const HostLocalAlignedMemoryHIP& other)
{
    // Utilize copy constructor + move assignment operator
    *this = HostLocalAlignedMemoryHIP(other);
    return *this;
}

HostLocalAlignedMemoryHIP& HostLocalAlignedMemoryHIP::operator=(HostLocalAlignedMemoryHIP&& other) noexcept
{
    assert(this != &other);
    if(size != 0)
    {
        HIP_CHECK(hipHostUnregister(hPtr));
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

HostLocalAlignedMemoryHIP::~HostLocalAlignedMemoryHIP()
{
    if(hPtr != 0) HIP_CHECK(hipHostUnregister(hPtr));
    #ifdef MRAY_WINDOWS
        _aligned_free(hPtr);
    #elif defined MRAY_LINUX
        std::free(hPtr);
    #endif
}

Byte* HostLocalAlignedMemoryHIP::DevicePtr()
{
    return static_cast<Byte*>(dPtr);
}

const Byte* HostLocalAlignedMemoryHIP::DevicePtr() const
{
    return static_cast<const Byte*>(dPtr);
}

void HostLocalAlignedMemoryHIP::ResizeBuffer(size_t newSize)
{
    if(neverDecrease && newSize <= size) return;

    size_t copySize = std::min(newSize, size);
    HostLocalAlignedMemoryHIP newMem(*system, newSize, alignment, neverDecrease);
    std::memcpy(newMem.hPtr, hPtr, copySize);
    *this = std::move(newMem);
}

size_t HostLocalAlignedMemoryHIP::Size() const
{
    return size;
}

size_t HostLocalAlignedMemoryHIP::AllocSize() const
{
    return allocSize;
}

size_t DeviceMemoryHIP::FindCommonGranularity() const
{
    // Determine a device common granularity
    size_t commonGranularity = 1;
    std::for_each(deviceIds.cbegin(), deviceIds.cend(),
                  [&](int deviceId)
    {
        hipMemAllocationProp props = {};
        props.location.type = hipMemLocationTypeDevice;
        props.location.id = deviceId;
        props.type = hipMemAllocationTypePinned;

        size_t devGranularity;
        HIP_DRIVER_CHECK(hipMemGetAllocationGranularity(&devGranularity, &props,
                                                        hipMemAllocationGranularityRecommended));
        // This is technically not correct
        commonGranularity = std::max(commonGranularity, devGranularity);
    });
    return commonGranularity;
}

size_t DeviceMemoryHIP::NextDeviceIndex()
{
    curDeviceIndex = (curDeviceIndex + 1) % deviceIds.size();
    return curDeviceIndex;
}

DeviceMemoryHIP::DeviceMemoryHIP(const std::vector<const GPUDeviceHIP*>& devices,
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
    for(const GPUDeviceHIP* dPtr : devices)
        deviceIds.push_back(dPtr->DeviceId());

    size_t commonGranularity = FindCommonGranularity();
    allocationGranularity = Math::NextMultiple(allocGranularity, commonGranularity);
    reserveGranularity = Math::NextMultiple(resGranularity, commonGranularity);

    reservedSize = reserveGranularity;
    HIP_DRIVER_MEM_THROW(hipMemAddressReserve(&mPtr, reservedSize, 0, 0, 0));
    vaRanges.emplace_back(mPtr, reservedSize);
}

DeviceMemoryHIP& DeviceMemoryHIP::operator=(DeviceMemoryHIP&& other) noexcept
{
    assert(this != &other);
    // Dealloc Memory
    if(allocSize != 0) HIP_DRIVER_CHECK(hipMemUnmap(mPtr, allocSize));
    for(const Allocations& a : allocs)
    {
        HIP_DRIVER_CHECK(hipMemRelease(a.handle));
    }
    for(const VARanges& vaRange : vaRanges)
        HIP_DRIVER_CHECK(hipMemAddressFree(vaRange.first, vaRange.second));

    //
    deviceIds = std::move(other.deviceIds);
    allocs = std::move(other.allocs);
    vaRanges = std::move(other.vaRanges);
    curDeviceIndex = other.curDeviceIndex;
    allocationGranularity = other.allocationGranularity;
    reserveGranularity = other.reserveGranularity;
    reservedSize = other.reservedSize;
    mPtr = other.mPtr;
    allocSize = other.allocSize;
    neverDecrease = other.neverDecrease;

    return *this;
}

DeviceMemoryHIP::~DeviceMemoryHIP()
{
    if(allocSize != 0) HIP_DRIVER_CHECK(hipMemUnmap(mPtr, allocSize));
    for(const Allocations& a : allocs)
    {
        HIP_DRIVER_CHECK(hipMemRelease(a.handle));
    }
    for(const VARanges& vaRange : vaRanges)
        HIP_DRIVER_CHECK(hipMemAddressFree(vaRange.first, vaRange.second));
}

void DeviceMemoryHIP::ResizeBuffer(size_t newSize)
{
    auto HaltVisibleDevices = [this]()
    {
        int curDevice;
        HIP_CHECK(hipGetDevice(&curDevice));
        for(int deviceId : deviceIds)
        {
            HIP_CHECK(hipSetDevice(deviceId));
            HIP_CHECK(hipDeviceSynchronize());
        }
        HIP_CHECK(hipSetDevice(curDevice));
    };

    // Align the newSize
    newSize = Math::NextMultiple(newSize, allocationGranularity);
    bool allocSizeChanged = false;
    bool reservationRelocated = false;

    if(newSize > reservedSize)
    {
        size_t extraReserve = Math::NextMultiple(newSize, reserveGranularity);
        extraReserve -= reservedSize;

        // Try to allocate adjacent chunk
        hipDeviceptr_t newPtr = 0ULL;
        hipError_t status = hipMemAddressReserve(&newPtr, extraReserve, 0,
                                                 AdvanceHIPPtr(mPtr, reservedSize), 0);

        // Failed to allocate
        if((status != hipSuccess) ||
           (newPtr != AdvanceHIPPtr(mPtr, reservedSize)))
        {
            // Do a complete halt of the GPU(s) usage
            // we need to reallocate.
            // TODO: should we restate the device?
            HaltVisibleDevices();

            size_t offset = 0;
            // Realloc
            if(allocSize != 0) HIP_DRIVER_CHECK(hipMemUnmap(mPtr, allocSize));
            for(const VARanges& vaRange : vaRanges)
                HIP_DRIVER_CHECK(hipMemAddressFree(vaRange.first, vaRange.second));
            // Get a green slate
            reservedSize = extraReserve + reservedSize;
            HIP_DRIVER_MEM_THROW(hipMemAddressReserve(&newPtr, reservedSize, 0, 0, 0));
            // Remap the old memory
            for(const Allocations& a : allocs)
            {
                HIP_DRIVER_MEM_THROW(hipMemMap(AdvanceHIPPtr(newPtr, offset), a.allocSize, 0, a.handle, 0));
                offset += a.allocSize;
            }
            mPtr = newPtr;
            vaRanges.clear();
            vaRanges.emplace_back(mPtr, reservedSize);
            reservationRelocated = true;
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
        // We need to shrink the memory, meaning we will
        // do an unmap, so halt all operations on the device
        // TODO: If check is unecessary?
        if(!reservationRelocated)
            HaltVisibleDevices();

        size_t offset = allocSize;
        auto it = allocs.crbegin();
        for(; it != allocs.crend(); it++)
        {
            offset -= allocationGranularity;
            if(newSize > offset)
                break;
            // Release the memory
            HIP_DRIVER_CHECK(hipMemUnmap(AdvanceHIPPtr(mPtr, offset), allocationGranularity));
            HIP_DRIVER_CHECK(hipMemRelease(it->handle));
        }
        assert(std::distance(it, allocs.crend()) >= 0);
        allocs.resize(static_cast<size_t>(std::distance(it, allocs.crend())));
        allocSize = offset + allocationGranularity;
        allocSizeChanged = true;
    }
    // Enlarge the memory
    else if(newSize > allocSize)
    {
        size_t offset = allocSize;
        // Now calculate extra allocation
        size_t extraSize = newSize - allocSize;
        extraSize = Math::NextMultiple(extraSize, allocationGranularity);

        assert((extraSize % allocationGranularity) == 0);
        size_t newAllocCount = extraSize / allocationGranularity;

        // Add extra allocations
        for(size_t i = 0; i < newAllocCount; i++)
        {
            int deviceId = deviceIds[curDeviceIndex];
            hipMemAllocationProp props = {};
            props.location.type = hipMemLocationTypeDevice;
            props.location.id = deviceId;
            props.type = hipMemAllocationTypePinned;

            Allocations newAlloc =
            {
                .deviceId = deviceId,
                .handle = 0,
                .allocSize = allocationGranularity,
            };
            HIP_DRIVER_CHECK(hipMemCreate(&newAlloc.handle, allocationGranularity, &props, 0));
            HIP_DRIVER_MEM_THROW(hipMemMap(AdvanceHIPPtr(mPtr, offset), allocationGranularity, 0, newAlloc.handle, 0));
            offset += allocationGranularity;

            allocs.push_back(newAlloc);
            NextDeviceIndex();
        }
        assert(offset == newSize);
        allocSize = newSize;
        allocSizeChanged = true;
    }

    // Nothing to set access to
    if(allocSize == 0 || !allocSizeChanged) return;

    // Set the newly mapped range accessor
    std::vector<hipMemAccessDesc> accessDescriptions(deviceIds.size(),
                                                     hipMemAccessDesc{});
    for(size_t i = 0; i < accessDescriptions.size(); i++)
    {
        hipMemAccessDesc& ad = accessDescriptions[i];
        ad.location.type = hipMemLocationTypeDevice;
        ad.location.id = deviceIds[i];
        ad.flags = hipMemAccessFlagsProtReadWrite;
    }
    HIP_DRIVER_CHECK(hipMemSetAccess(mPtr, allocSize,
                                     accessDescriptions.data(),
                                     accessDescriptions.size()));
}

size_t DeviceMemoryHIP::Size() const
{
    return allocSize;
}

}