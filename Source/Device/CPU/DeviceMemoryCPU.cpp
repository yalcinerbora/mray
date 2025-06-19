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
}

DeviceLocalMemoryCPU::DeviceLocalMemoryCPU(const DeviceLocalMemoryCPU& other)
    : DeviceLocalMemoryCPU(*(other.gpu), other.size)
{
}

DeviceLocalMemoryCPU::DeviceLocalMemoryCPU(DeviceLocalMemoryCPU&& other) noexcept
    : gpu(other.gpu)
    , dPtr(other.dPtr)
    , size(other.size)
    , allocSize(other.allocSize)
{
}

DeviceLocalMemoryCPU& DeviceLocalMemoryCPU::operator=(const DeviceLocalMemoryCPU& other)
{
    assert(this != &other);
    return *this;
}

DeviceLocalMemoryCPU& DeviceLocalMemoryCPU::operator=(DeviceLocalMemoryCPU&& other) noexcept
{
    assert(this != &other);
    return *this;
}

DeviceLocalMemoryCPU::~DeviceLocalMemoryCPU()
{
}

void DeviceLocalMemoryCPU::ResizeBuffer(size_t newSize)
{
}

size_t DeviceLocalMemoryCPU::Size() const
{
    return size;
}

void DeviceLocalMemoryCPU::MigrateToOtherDevice(const GPUDeviceCPU& deviceTo)
{
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
}

HostLocalMemoryCPU::HostLocalMemoryCPU(const HostLocalMemoryCPU& other)
    : HostLocalMemoryCPU(*other.system, other.size, other.neverDecrease)
{
}

HostLocalMemoryCPU::HostLocalMemoryCPU(HostLocalMemoryCPU&& other) noexcept
    : system(other.system)
    , hPtr(other.hPtr)
    , dPtr(other.dPtr)
    , size(other.size)
    , neverDecrease(other.neverDecrease)
{
}

HostLocalMemoryCPU& HostLocalMemoryCPU::operator=(const HostLocalMemoryCPU& other)
{
    assert(this != &other);
    return *this;
}

HostLocalMemoryCPU& HostLocalMemoryCPU::operator=(HostLocalMemoryCPU&& other) noexcept
{
    assert(this != &other);
    return *this;
}

HostLocalMemoryCPU::~HostLocalMemoryCPU()
{
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
    , alignment(alignIn)
    , neverDecrease(ndIn)
{}

HostLocalAlignedMemoryCPU::HostLocalAlignedMemoryCPU(const GPUSystemCPU& systemIn,
                                                     size_t sizeInBytes, size_t alignIn,
                                                     bool ndIn)
    : HostLocalAlignedMemoryCPU(systemIn, alignIn, ndIn)
{
    size = sizeInBytes;
    allocSize = Math::NextMultiple(sizeInBytes, alignment);
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
    if(size != 0)
    {
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

HostLocalAlignedMemoryCPU::~HostLocalAlignedMemoryCPU()
{
    #ifdef MRAY_WINDOWS
        _aligned_free(hPtr);
    #elif defined MRAY_LINUX
        std::free(hPtr);
    #endif
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
    std::memcpy(newMem.hPtr, hPtr, copySize);
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

size_t DeviceMemoryCPU::FindCommonGranularity() const
{
    return 4_KiB;
}

size_t DeviceMemoryCPU::NextDeviceIndex()
{
    curDeviceIndex = (curDeviceIndex + 1) % deviceIds.size();
    return curDeviceIndex;
}

DeviceMemoryCPU::DeviceMemoryCPU(const std::vector<const GPUDeviceCPU*>& devices,
                                 size_t allocGranularity,
                                 size_t resGranularity,
                                 bool neverDecrease)
    : allocSize(0)
    , neverDecrease(neverDecrease)
{
    assert(resGranularity != 0);
    assert(allocGranularity != 0);
    assert(!devices.empty());
}

DeviceMemoryCPU& DeviceMemoryCPU::operator=(DeviceMemoryCPU&& other) noexcept
{
    assert(this != &other);
    return *this;
}

DeviceMemoryCPU::~DeviceMemoryCPU()
{
}

void DeviceMemoryCPU::ResizeBuffer(size_t newSize)
{
}

size_t DeviceMemoryCPU::Size() const
{
    return allocSize;
}

}