#pragma once

/**

CPU Device Memory RAII principle classes
These are mirror of CUDA types. Some of the allocation
types does not make sense on CPU only context.

Still we implement these not with new/malloc etc. But
with virtual memory system of the OS
(mmap() VirtualAlloc() etc.) if applicable.

All of the operations (except allocation) are asynchronous.

TODO: should we interface these?

*/

#include <cassert>
#include <vector>

#include "Core/Definitions.h"
#include "Core/MemAlloc.h"

namespace mray::host
{

void*   AlignedAllocate(size_t allocSize, size_t alignment);
void    AlignedFree(void* ptr, size_t allocSize, size_t alignment);

class GPUDeviceCPU;
class GPUSystemCPU;
class GPUQueueCPU;

class DeviceLocalMemoryCPU
{
    private:
    const GPUDeviceCPU*             gpu;
    void*                           dPtr;
    size_t                          size;
    size_t                          allocSize;

    protected:
    public:
    // Constructors & Destructor
                            DeviceLocalMemoryCPU(const GPUDeviceCPU& gpu);
                            DeviceLocalMemoryCPU(const GPUDeviceCPU& gpu, size_t sizeInBytes);
                            DeviceLocalMemoryCPU(const DeviceLocalMemoryCPU&);
                            DeviceLocalMemoryCPU(DeviceLocalMemoryCPU&&) noexcept;
    DeviceLocalMemoryCPU&   operator=(const DeviceLocalMemoryCPU&);
    DeviceLocalMemoryCPU&   operator=(DeviceLocalMemoryCPU&&) noexcept;
                            ~DeviceLocalMemoryCPU();

    // Access
    explicit                operator Byte* ();
    explicit                operator const Byte* () const;

    // Misc
    void                    ResizeBuffer(size_t newSize);
    const GPUDeviceCPU&     Device() const;
    size_t                  Size() const;
    void                    MigrateToOtherDevice(const GPUDeviceCPU& deviceTo);
};

// Host local memory but it is visible to the device
// used for fast communication
class HostLocalMemoryCPU
{
    private:
    const GPUSystemCPU*     system;
    void*                   hPtr;
    void*                   dPtr;
    size_t                  size;
    bool                    neverDecrease = false;

    public:
    // Constructors & Destructor
                            HostLocalMemoryCPU(const GPUSystemCPU& system,
                                               bool neverDecrease = false);
                            HostLocalMemoryCPU(const GPUSystemCPU& system,
                                               size_t sizeInBytes,
                                               bool neverDecrease = false);
                            HostLocalMemoryCPU(const HostLocalMemoryCPU&);
                            HostLocalMemoryCPU(HostLocalMemoryCPU&&) noexcept;
    HostLocalMemoryCPU&     operator=(const HostLocalMemoryCPU&);
    HostLocalMemoryCPU&     operator=(HostLocalMemoryCPU&&) noexcept;
                            ~HostLocalMemoryCPU();
    // Access
    explicit                operator Byte* ();
    explicit                operator const Byte* () const;
    Byte*                   DevicePtr();
    const Byte*             DevicePtr() const;

    // Misc
    void                    ResizeBuffer(size_t newSize);
    size_t                  Size() const;
};

// ============================================== //
// This comment is copied from the CUDA version
// ============================================== //
//
// Vulkan foreign memory requires the mem to be
// page-aligned (Well it returns 4096, so thats what
// I inferred). On CUDA, cudaMallocHost returns 256 byte-aligned
// memory so we cannot use that
// So we utilize C's "aligned malloc" function and register that
// memory to CUDA.
class HostLocalAlignedMemoryCPU
{
    private:
    const GPUSystemCPU*     system;
    void*                   hPtr;
    void*                   dPtr;
    size_t                  size;
    size_t                  allocSize;
    size_t                  alignment;
    bool                    neverDecrease = false;

    public:
    // Constructors & Destructor
                                HostLocalAlignedMemoryCPU(const GPUSystemCPU& system,
                                                           size_t alignment,
                                                           bool neverDecrease = false);
                                HostLocalAlignedMemoryCPU(const GPUSystemCPU& system,
                                                           size_t sizeInBytes, size_t alignment,
                                                           bool neverDecrease = false);
                                HostLocalAlignedMemoryCPU(const HostLocalAlignedMemoryCPU&);
                                HostLocalAlignedMemoryCPU(HostLocalAlignedMemoryCPU&&) noexcept;
    HostLocalAlignedMemoryCPU&  operator=(const HostLocalAlignedMemoryCPU&);
    HostLocalAlignedMemoryCPU&  operator=(HostLocalAlignedMemoryCPU&&) noexcept;
                                ~HostLocalAlignedMemoryCPU();

    // Access
    explicit                operator Byte* ();
    explicit                operator const Byte* () const;
    Byte*                   DevicePtr();
    const Byte*             DevicePtr() const;

    // Misc
    void                    ResizeBuffer(size_t newSize);
    size_t                  Size() const;
    size_t                  AllocSize() const;
};

// Generic Device Memory (most of the cases this should be used)
// Automatic multi-device seperation (round-robin style) etc.
class DeviceMemoryCPU
{
    private:
    std::vector<const GPUDeviceCPU*>  devices;

    void*                       mPtr;
    size_t                      allocationGranularity;
    size_t                      reserveGranularity;
    size_t                      size;
    size_t                      allocSize;
    bool                        neverDecrease = false;

    public:
    // Constructors & Destructor
                            DeviceMemoryCPU(const std::vector<const GPUDeviceCPU*>& devices,
                                            size_t allocationGranularity,
                                            size_t preReserveSize,
                                            bool neverDecrease = false);
                            DeviceMemoryCPU(const DeviceMemoryCPU&) = delete;
                            DeviceMemoryCPU(DeviceMemoryCPU&&) noexcept = default;
    DeviceMemoryCPU&        operator=(const DeviceMemoryCPU&) = delete;
    DeviceMemoryCPU&        operator=(DeviceMemoryCPU&&) noexcept;
                            ~DeviceMemoryCPU();

    // Access
    explicit                operator Byte*();
    explicit                operator const Byte*() const;
    // Misc
    void                    ResizeBuffer(size_t newSize);
    size_t                  Size() const;
};

inline const GPUDeviceCPU& DeviceLocalMemoryCPU::Device() const
{
    return *gpu;
}

inline DeviceLocalMemoryCPU::operator Byte*()
{
    return reinterpret_cast<Byte*>(dPtr);
}

inline DeviceLocalMemoryCPU::operator const Byte*() const
{
    return reinterpret_cast<const Byte*>(dPtr);
}

inline HostLocalMemoryCPU::operator Byte*()
{
    return reinterpret_cast<Byte*>(hPtr);
}

inline HostLocalMemoryCPU::operator const Byte*() const
{
    return reinterpret_cast<const Byte*>(hPtr);
}

inline HostLocalAlignedMemoryCPU::operator Byte* ()
{
    return reinterpret_cast<Byte*>(hPtr);
}

inline HostLocalAlignedMemoryCPU::operator const Byte* () const
{
    return reinterpret_cast<const Byte*>(hPtr);
}

inline DeviceMemoryCPU::operator Byte*()
{
    return std::bit_cast<Byte*>(mPtr);
}

inline DeviceMemoryCPU::operator const Byte*() const
{
    return std::bit_cast<const Byte*>(mPtr);
}

static_assert(MemoryC<DeviceLocalMemoryCPU>,
              "\"DeviceLocalMemoryCPU\" does not satisfy memory concept.");
static_assert(MemoryC<HostLocalMemoryCPU>,
              "\"HostLocalMemoryCPU\" does not satisfy memory concept.");
static_assert(MemoryC<HostLocalAlignedMemoryCPU>,
              "\"HostLocalAlignedMemoryCPU\" does not satisfy memory concept.");
static_assert(MemoryC<DeviceMemoryCPU>,
              "\"DeviceMemoryCPU\" does not satisfy memory concept.");

}