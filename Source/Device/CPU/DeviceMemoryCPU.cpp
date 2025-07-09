#include "DeviceMemoryCPU.h"
#include "DefinitionsCPU.h"
#include "GPUSystemCPU.h"

#include "Core/System.h"

#include <algorithm>

namespace mray::host
{

DeviceLocalMemoryCPU::DeviceLocalMemoryCPU(const GPUDeviceCPU& device)
    : gpu(&device)
    , dPtr(nullptr)
    , size(0)
    , allocSize(0)
{}

DeviceLocalMemoryCPU::DeviceLocalMemoryCPU(const GPUDeviceCPU& device, size_t sizeInBytes)
    : DeviceLocalMemoryCPU(device)
{
    assert(sizeInBytes != 0);
    allocSize = Math::NextMultiple(sizeInBytes, MemAlloc::DefaultSystemAlignment());
    size = sizeInBytes;
    dPtr = AlignedAlloc(allocSize, MemAlloc::DefaultSystemAlignment());
}

DeviceLocalMemoryCPU::DeviceLocalMemoryCPU(const DeviceLocalMemoryCPU& other)
    : DeviceLocalMemoryCPU(*(other.gpu), other.size)
{
    std::memcpy(dPtr, other.dPtr, size);
}

DeviceLocalMemoryCPU::DeviceLocalMemoryCPU(DeviceLocalMemoryCPU&& other) noexcept
    : gpu(other.gpu)
    , dPtr(other.dPtr)
    , size(other.size)
    , allocSize(other.allocSize)
{
    other.dPtr = nullptr;
}

DeviceLocalMemoryCPU& DeviceLocalMemoryCPU::operator=(const DeviceLocalMemoryCPU& other)
{
    assert(this != &other);
    if(dPtr)
    {
        AlignedFree(dPtr, allocSize, MemAlloc::DefaultSystemAlignment());
        dPtr = nullptr;
    }
    dPtr = AlignedAlloc(other.allocSize, MemAlloc::DefaultSystemAlignment());
    gpu = other.gpu;
    size = other.size;
    allocSize = other.allocSize;
    std::memcpy(dPtr, other.dPtr, size);
    return *this;
}

DeviceLocalMemoryCPU& DeviceLocalMemoryCPU::operator=(DeviceLocalMemoryCPU&& other) noexcept
{
    assert(this != &other);
    if(dPtr)
    {
        AlignedFree(dPtr, allocSize, MemAlloc::DefaultSystemAlignment());
        dPtr = nullptr;
    }
    gpu = other.gpu;
    dPtr = other.dPtr;
    size = other.size;
    allocSize = other.allocSize;
    //
    other.dPtr = nullptr;
    return *this;
}

DeviceLocalMemoryCPU::~DeviceLocalMemoryCPU()
{
    // TODO: Checl if aligned versions accept nullptr
    if(dPtr)
    {
        AlignedFree(dPtr, allocSize, MemAlloc::DefaultSystemAlignment());
        dPtr = nullptr;
    }
}

void DeviceLocalMemoryCPU::ResizeBuffer(size_t newSize)
{
    DeviceLocalMemoryCPU newMem(*gpu, newSize);
    size_t copySize = std::min(newSize, size);
    if(dPtr) std::memcpy(newMem.dPtr, dPtr, copySize);
    *this = std::move(newMem);
}

size_t DeviceLocalMemoryCPU::Size() const
{
    return size;
}

void DeviceLocalMemoryCPU::MigrateToOtherDevice(const GPUDeviceCPU& deviceTo)
{
    // TODO: This needs to be changed after NUMA support
    gpu = &deviceTo;
}

HostLocalMemoryCPU::HostLocalMemoryCPU(const GPUSystemCPU& system,
                                       bool neverDecrease)
    : system(&system)
    , hPtr(nullptr)
    , dPtr(nullptr)
    , size(0)
    , neverDecrease(neverDecrease)
{}

HostLocalMemoryCPU::HostLocalMemoryCPU(const GPUSystemCPU& system,
                                       size_t sizeInBytes,
                                       bool neverDecrease)
    : HostLocalMemoryCPU(system, neverDecrease)
{
    assert(sizeInBytes != 0);
    // TODO: change this to virtual memory calls as well
    hPtr = AlignedAlloc(sizeInBytes, MemAlloc::DefaultSystemAlignment());
    dPtr = hPtr;
    size = sizeInBytes;
}

HostLocalMemoryCPU::HostLocalMemoryCPU(const HostLocalMemoryCPU& other)
    : HostLocalMemoryCPU(*other.system, other.size, other.neverDecrease)
{
    std::memcpy(hPtr, other.hPtr, size);
}

HostLocalMemoryCPU::HostLocalMemoryCPU(HostLocalMemoryCPU&& other) noexcept
    : system(other.system)
    , hPtr(other.hPtr)
    , dPtr(other.dPtr)
    , size(other.size)
    , neverDecrease(other.neverDecrease)
{
    if(hPtr) AlignedFree(hPtr, size, MemAlloc::DefaultSystemAlignment());
    other.hPtr = nullptr;
    other.dPtr = nullptr;
    other.size = 0;
}

HostLocalMemoryCPU& HostLocalMemoryCPU::operator=(const HostLocalMemoryCPU& other)
{
    assert(this != &other);

    if(hPtr) AlignedFree(hPtr, size, MemAlloc::DefaultSystemAlignment());
    size = other.size;
    neverDecrease = other.neverDecrease;
    hPtr = AlignedAlloc(size, MemAlloc::DefaultSystemAlignment());
    dPtr = hPtr;
    std::memcpy(hPtr, other.hPtr, size);
    return *this;
}

HostLocalMemoryCPU& HostLocalMemoryCPU::operator=(HostLocalMemoryCPU&& other) noexcept
{
    assert(this != &other);
    if(hPtr) AlignedFree(hPtr, size, MemAlloc::DefaultSystemAlignment());
    size = other.size;
    hPtr = other.hPtr;
    dPtr = other.dPtr;
    neverDecrease = other.neverDecrease;

    other.size = 0;
    other.hPtr = nullptr;
    other.dPtr = nullptr;
    return *this;
}

HostLocalMemoryCPU::~HostLocalMemoryCPU()
{
    if(hPtr) AlignedFree(hPtr, size, MemAlloc::DefaultSystemAlignment());
}

Byte* HostLocalMemoryCPU::DevicePtr()
{
    return reinterpret_cast<Byte*>(dPtr);
}

const Byte* HostLocalMemoryCPU::DevicePtr() const
{
    return reinterpret_cast<const Byte*>(dPtr);
}

void HostLocalMemoryCPU::ResizeBuffer(size_t newSize)
{
    if(neverDecrease && newSize <= size) return;

    size_t copySize = std::min(newSize, size);
    HostLocalMemoryCPU newMem(*system, newSize, neverDecrease);
    std::memcpy(newMem.hPtr, hPtr, copySize);
    *this = std::move(newMem);
}

size_t HostLocalMemoryCPU::Size() const
{
    return size;
}

// Constructors & Destructor
HostLocalAlignedMemoryCPU::HostLocalAlignedMemoryCPU(const GPUSystemCPU& systemIn,
                                                     size_t alignIn, bool ndIn)
    : system(&systemIn)
    , hPtr(nullptr)
    , dPtr(nullptr)
    , size(0)
    , allocSize(0)
    , alignment(std::max(alignIn, MemAlloc::DefaultSystemAlignment()))
    , neverDecrease(ndIn)
{}

HostLocalAlignedMemoryCPU::HostLocalAlignedMemoryCPU(const GPUSystemCPU& systemIn,
                                                     size_t sizeInBytes, size_t alignIn,
                                                     bool ndIn)
    : HostLocalAlignedMemoryCPU(systemIn, alignIn, ndIn)
{
    size = sizeInBytes;
    allocSize = Math::NextMultiple(sizeInBytes, alignment);
    hPtr = AlignedAlloc(allocSize, alignment);
    dPtr = hPtr;
}

HostLocalAlignedMemoryCPU::HostLocalAlignedMemoryCPU(const HostLocalAlignedMemoryCPU& other)
    : HostLocalAlignedMemoryCPU(*other.system,
                                 other.size, other.alignment,
                                 other.neverDecrease)
{
    std::memcpy(hPtr, other.hPtr, size);
}

HostLocalAlignedMemoryCPU::HostLocalAlignedMemoryCPU(HostLocalAlignedMemoryCPU&& other) noexcept
    : system(other.system)
    , hPtr(std::exchange(other.hPtr, nullptr))
    , dPtr(std::exchange(other.dPtr, nullptr))
    , size(other.size)
    , allocSize(other.allocSize)
    , alignment(other.alignment)
    , neverDecrease(other.neverDecrease)
{}

HostLocalAlignedMemoryCPU& HostLocalAlignedMemoryCPU::operator=(const HostLocalAlignedMemoryCPU& other)
{
    // Utilize copy constructor + move assignment operator
    *this = HostLocalAlignedMemoryCPU(other);
    return *this;
}

HostLocalAlignedMemoryCPU& HostLocalAlignedMemoryCPU::operator=(HostLocalAlignedMemoryCPU&& other) noexcept
{
    assert(this != &other);
    if(hPtr) AlignedFree(hPtr, allocSize, alignment);

    system = other.system;
    hPtr = std::exchange(other.hPtr, nullptr);
    dPtr = std::exchange(other.dPtr, nullptr);
    size = other.size;
    allocSize = other.allocSize;
    alignment = other.alignment;
    neverDecrease = other.neverDecrease;
    return *this;
}

HostLocalAlignedMemoryCPU::~HostLocalAlignedMemoryCPU()
{
    if(hPtr) AlignedFree(hPtr, allocSize, alignment);
}

Byte* HostLocalAlignedMemoryCPU::DevicePtr()
{
    return static_cast<Byte*>(dPtr);
}

const Byte* HostLocalAlignedMemoryCPU::DevicePtr() const
{
    return static_cast<const Byte*>(dPtr);
}

void HostLocalAlignedMemoryCPU::ResizeBuffer(size_t newSize)
{
    if(neverDecrease && newSize <= size) return;

    size_t copySize = std::min(newSize, size);
    HostLocalAlignedMemoryCPU newMem(*system, newSize, alignment, neverDecrease);
    if(hPtr) std::memcpy(newMem.hPtr, hPtr, copySize);
    *this = std::move(newMem);
}

size_t HostLocalAlignedMemoryCPU::Size() const
{
    return size;
}

size_t HostLocalAlignedMemoryCPU::AllocSize() const
{
    return allocSize;
}

DeviceMemoryCPU::DeviceMemoryCPU(const std::vector<const GPUDeviceCPU*>& devicesIn,
                                 [[maybe_unused]] size_t allocGranularityIn,
                                 [[maybe_unused]] size_t reserveGranularityIn,
                                 bool neverDecrease)
    : devices(devicesIn)
    , mPtr(nullptr)
    , allocationGranularity(MemAlloc::DefaultSystemAlignment())
    , reserveGranularity(MemAlloc::DefaultSystemAlignment())
    , size(0)
    , allocSize(0)
    , neverDecrease(neverDecrease)
{
    assert(allocGranularityIn != 0);
    assert(reserveGranularityIn != 0);
    assert(!devices.empty());
}

DeviceMemoryCPU& DeviceMemoryCPU::operator=(DeviceMemoryCPU&& other) noexcept
{
    assert(this != &other);
    if(mPtr) AlignedFree(mPtr, allocSize, allocationGranularity);
    //
    devices = std::move(other.devices);
    allocationGranularity = other.allocationGranularity;
    reserveGranularity = other.reserveGranularity;
    mPtr = other.mPtr;
    size = other.size;
    allocSize = other.allocSize;
    neverDecrease = other.neverDecrease;
    return *this;
}

DeviceMemoryCPU::~DeviceMemoryCPU()
{
    if(mPtr) AlignedFree(mPtr, allocSize, allocationGranularity);
}

void DeviceMemoryCPU::ResizeBuffer(size_t newSize)
{
    if(neverDecrease && newSize < size)
    {
        size = newSize;
        return;
    }
    //
    auto HaltVisibleDevices = [this]()
    {
        for(const auto* device : devices)
        {
            for(uint32_t i = 0; i < ComputeQueuePerDevice; i++)
                device->GetComputeQueue(0).Barrier().Wait();
            device->GetTransferQueue().Barrier().Wait();
        }
    };
    //
    HaltVisibleDevices();
    size_t newAllocSize = Math::NextMultiple(newSize, allocationGranularity);
    void* newPtr = AlignedAlloc(newAllocSize, allocationGranularity);
    if(mPtr)
    {
        std::memcpy(newPtr, mPtr, std::min(newSize, size));
        AlignedFree(mPtr, allocSize, allocationGranularity);
    }
    mPtr = newPtr;
    allocSize = newAllocSize;
    size = newSize;
}

size_t DeviceMemoryCPU::Size() const
{
    return size;
}

}