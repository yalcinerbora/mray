#pragma once

/**

HIP Device Memory RAII principle classes

New unified memory classes are used where applicable
These are wrapper of hip functions and their most important responsibility is
to delete allocated memory

All of the operations (except allocation) are asynchronous.

TODO: should we interface these?

*/

#include <hip/hip_runtime.h>
#include <cassert>
#include <vector>

#include "Core/Definitions.h"
#include "Core/MemAlloc.h"

namespace mray::hip
{

class GPUDeviceHIP;
class GPUSystemHIP;
class GPUQueueHIP;

class DeviceLocalMemoryHIP
{
    private:
    const GPUDeviceHIP*             gpu;
    void*                           dPtr;
    size_t                          size;
    size_t                          allocSize;
    hipMemGenericAllocationHandle_t memHandle;

    protected:
    public:
    // Constructors & Destructor
                            DeviceLocalMemoryHIP(const GPUDeviceHIP& gpu);
                            DeviceLocalMemoryHIP(const GPUDeviceHIP& gpu, size_t sizeInBytes);
                            DeviceLocalMemoryHIP(const DeviceLocalMemoryHIP&);
                            DeviceLocalMemoryHIP(DeviceLocalMemoryHIP&&) noexcept;
    DeviceLocalMemoryHIP&   operator=(const DeviceLocalMemoryHIP&);
    DeviceLocalMemoryHIP&   operator=(DeviceLocalMemoryHIP&&) noexcept;
                            ~DeviceLocalMemoryHIP();

    // Access
    explicit                operator Byte* ();
    explicit                operator const Byte* () const;

    // Misc
    void                    ResizeBuffer(size_t newSize);
    const GPUDeviceHIP&     Device() const;
    size_t                  Size() const;
    void                    MigrateToOtherDevice(const GPUDeviceHIP& deviceTo);
};

// Host local memory but it is visible to the device
// used for fast communication
class HostLocalMemoryHIP
{
    private:
    const GPUSystemHIP*     system;
    void*                   hPtr;
    void*                   dPtr;
    size_t                  size;
    bool                    neverDecrease = false;

    public:
    // Constructors & Destructor
                            HostLocalMemoryHIP(const GPUSystemHIP& system,
                                               bool neverDecrease = false);
                            HostLocalMemoryHIP(const GPUSystemHIP& system,
                                               size_t sizeInBytes,
                                               bool neverDecrease = false);
                            HostLocalMemoryHIP(const HostLocalMemoryHIP&);
                            HostLocalMemoryHIP(HostLocalMemoryHIP&&) noexcept;
    HostLocalMemoryHIP&     operator=(const HostLocalMemoryHIP&);
    HostLocalMemoryHIP&     operator=(HostLocalMemoryHIP&&) noexcept;
                            ~HostLocalMemoryHIP();
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
class HostLocalAlignedMemoryHIP
{
    private:
    const GPUSystemHIP*     system;
    void*                   hPtr;
    void*                   dPtr;
    size_t                  size;
    size_t                  allocSize;
    size_t                  alignment;
    bool                    neverDecrease = false;

    public:
    // Constructors & Destructor
                                HostLocalAlignedMemoryHIP(const GPUSystemHIP& system,
                                                           size_t alignment,
                                                           bool neverDecrease = false);
                                HostLocalAlignedMemoryHIP(const GPUSystemHIP& system,
                                                           size_t sizeInBytes, size_t alignment,
                                                           bool neverDecrease = false);
                                HostLocalAlignedMemoryHIP(const HostLocalAlignedMemoryHIP&);
                                HostLocalAlignedMemoryHIP(HostLocalAlignedMemoryHIP&&) noexcept;
    HostLocalAlignedMemoryHIP&  operator=(const HostLocalAlignedMemoryHIP&);
    HostLocalAlignedMemoryHIP&  operator=(HostLocalAlignedMemoryHIP&&) noexcept;
                                ~HostLocalAlignedMemoryHIP();

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
class DeviceMemoryHIP
{
    struct Allocations
    {
        int                             deviceId;
        hipMemGenericAllocationHandle_t handle;
        size_t                          allocSize;
    };
    using VARanges = Pair<hipDeviceptr_t, size_t>;


    private:
    std::vector<int>            deviceIds;
    std::vector<Allocations>    allocs;
    std::vector<VARanges>       vaRanges;

    size_t                      curDeviceIndex = 0;
    size_t                      allocationGranularity;
    size_t                      reserveGranularity;
    size_t                      reservedSize;

    hipDeviceptr_t              mPtr;
    size_t                      allocSize;
    bool                        neverDecrease = false;

    size_t                      FindCommonGranularity() const;
    size_t                      NextDeviceIndex();

    public:
    // Constructors & Destructor
                            DeviceMemoryHIP(const std::vector<const GPUDeviceHIP*>& devices,
                                            size_t allocationGranularity,
                                            size_t preReserveSize,
                                            bool neverDecrease = false);
                            DeviceMemoryHIP(const DeviceMemoryHIP&) = delete;
                            DeviceMemoryHIP(DeviceMemoryHIP&&) noexcept = default;
    DeviceMemoryHIP&        operator=(const DeviceMemoryHIP&) = delete;
    DeviceMemoryHIP&        operator=(DeviceMemoryHIP&&) noexcept;
                            ~DeviceMemoryHIP();

    // Access
    explicit                operator Byte*();
    explicit                operator const Byte*() const;
    // Misc
    void                    ResizeBuffer(size_t newSize);
    size_t                  Size() const;
};

inline const GPUDeviceHIP& DeviceLocalMemoryHIP::Device() const
{
    return *gpu;
}

inline DeviceLocalMemoryHIP::operator Byte*()
{
    return reinterpret_cast<Byte*>(dPtr);
}

inline DeviceLocalMemoryHIP::operator const Byte*() const
{
    return reinterpret_cast<const Byte*>(dPtr);
}

inline HostLocalMemoryHIP::operator Byte*()
{
    return reinterpret_cast<Byte*>(hPtr);
}

inline HostLocalMemoryHIP::operator const Byte*() const
{
    return reinterpret_cast<const Byte*>(hPtr);
}

inline HostLocalAlignedMemoryHIP::operator Byte* ()
{
    return reinterpret_cast<Byte*>(hPtr);
}

inline HostLocalAlignedMemoryHIP::operator const Byte* () const
{
    return reinterpret_cast<const Byte*>(hPtr);
}

inline DeviceMemoryHIP::operator Byte*()
{
    return std::bit_cast<Byte*>(mPtr);
}

inline DeviceMemoryHIP::operator const Byte*() const
{
    return std::bit_cast<const Byte*>(mPtr);
}

static_assert(MemoryC<DeviceLocalMemoryHIP>,
              "\"DeviceLocalMemoryHIP\" does not satisfy memory concept.");
static_assert(MemoryC<HostLocalMemoryHIP>,
              "\"HostLocalMemoryHIP\" does not satisfy memory concept.");
static_assert(MemoryC<HostLocalAlignedMemoryHIP>,
              "\"HostLocalAlignedMemoryHIP\" does not satisfy memory concept.");
static_assert(MemoryC<DeviceMemoryHIP>,
              "\"DeviceMemoryHIP\" does not satisfy memory concept.");

}