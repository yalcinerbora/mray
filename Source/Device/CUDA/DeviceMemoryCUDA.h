#pragma once

/**

CUDA Device Memory RAII principle classes

New unified memory classes are used where applicable
These are wrapper of cuda functions and their most important responsibility is
to delete allocated memory

All of the operations (except allocation) are asynchronous.

TODO: should we interface these?

*/
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include <cuda.h>

#include "Core/Definitions.h"
#include "Core/MemAlloc.h"

namespace mray::cuda
{

class GPUDeviceCUDA;
class GPUSystemCUDA;
class GPUQueueCUDA;

class DeviceLocalMemoryCUDA
{
    private:
    const GPUDeviceCUDA*            gpu;
    void*                           dPtr;
    size_t                          size;
    size_t                          allocSize;
    CUmemGenericAllocationHandle    memHandle;

    protected:
    public:
    // Constructors & Destructor
                            DeviceLocalMemoryCUDA(const GPUDeviceCUDA& gpu);
                            DeviceLocalMemoryCUDA(const GPUDeviceCUDA& gpu, size_t sizeInBytes);
                            DeviceLocalMemoryCUDA(const DeviceLocalMemoryCUDA&);
                            DeviceLocalMemoryCUDA(DeviceLocalMemoryCUDA&&) noexcept;
    DeviceLocalMemoryCUDA&  operator=(const DeviceLocalMemoryCUDA&);
    DeviceLocalMemoryCUDA&  operator=(DeviceLocalMemoryCUDA&&) noexcept;
                            ~DeviceLocalMemoryCUDA();

    // Access
    explicit                operator Byte* ();
    explicit                operator const Byte* () const;

    // Misc
    void                    ResizeBuffer(size_t newSize);
    const GPUDeviceCUDA&    Device() const;
    size_t                  Size() const;
    void                    MigrateToOtherDevice(const GPUDeviceCUDA& deviceTo);
};

// Host local memory but it is visible to the device
// used for fast communication
class HostLocalMemoryCUDA
{
    private:
    const GPUSystemCUDA*    system;
    void*                   hPtr;
    void*                   dPtr;
    size_t                  size;
    bool                    neverDecrease = false;

    public:
    // Constructors & Destructor
                            HostLocalMemoryCUDA(const GPUSystemCUDA& system,
                                                bool neverDecrease = false);
                            HostLocalMemoryCUDA(const GPUSystemCUDA& system,
                                                size_t sizeInBytes,
                                                bool neverDecrease = false);
                            HostLocalMemoryCUDA(const HostLocalMemoryCUDA&);
                            HostLocalMemoryCUDA(HostLocalMemoryCUDA&&) noexcept;
    HostLocalMemoryCUDA&    operator=(const HostLocalMemoryCUDA&);
    HostLocalMemoryCUDA&    operator=(HostLocalMemoryCUDA&&) noexcept;
                            ~HostLocalMemoryCUDA();
    // Access
    explicit                operator Byte* ();
    explicit                operator const Byte* () const;
    Byte*                   DevicePtr();
    const Byte*             DevicePtr() const;

    // Misc
    void                    ResizeBuffer(size_t newSize);
    size_t                  Size() const;
};

// Vulkan foreign memory requires the mem to be
// page-aligned (Well it returns 4096, so thats what
// I inferred). On CUDA, cudaMallocHost returns 256 byte-aligned
// memory so we cannot use that
// So we utilize C's "aligned malloc" function and register that
// memory to CUDA.
class HostLocalAlignedMemoryCUDA
{
    private:
    const GPUSystemCUDA*    system;
    void*                   hPtr;
    void*                   dPtr;
    size_t                  size;
    size_t                  allocSize;
    size_t                  alignment;
    bool                    neverDecrease = false;

    public:
    // Constructors & Destructor
                                HostLocalAlignedMemoryCUDA(const GPUSystemCUDA& system,
                                                           size_t alignment,
                                                           bool neverDecrease = false);
                                HostLocalAlignedMemoryCUDA(const GPUSystemCUDA& system,
                                                           size_t sizeInBytes, size_t alignment,
                                                           bool neverDecrease = false);
                                HostLocalAlignedMemoryCUDA(const HostLocalAlignedMemoryCUDA&);
                                HostLocalAlignedMemoryCUDA(HostLocalAlignedMemoryCUDA&&) noexcept;
    HostLocalAlignedMemoryCUDA& operator=(const HostLocalAlignedMemoryCUDA&);
    HostLocalAlignedMemoryCUDA& operator=(HostLocalAlignedMemoryCUDA&&) noexcept;
                                ~HostLocalAlignedMemoryCUDA();

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
class DeviceMemoryCUDA
{
    struct Allocations
    {
        int                             deviceId;
        CUmemGenericAllocationHandle    handle;
        size_t                          allocSize;
    };
    using VARanges = Pair<CUdeviceptr, size_t>;


    private:
    std::vector<int>            deviceIds;
    std::vector<Allocations>    allocs;
    std::vector<VARanges>       vaRanges;

    size_t                      curDeviceIndex = 0;
    size_t                      allocationGranularity;
    size_t                      reserveGranularity;
    size_t                      reservedSize;

    CUdeviceptr                 mPtr;
    size_t                      allocSize;
    bool                        neverDecrease = false;

    size_t                      FindCommonGranularity() const;
    size_t                      NextDeviceIndex();

    public:
    // Constructors & Destructor
                            DeviceMemoryCUDA(const std::vector<const GPUDeviceCUDA*>& devices,
                                                size_t allocationGranularity,
                                                size_t preReserveSize,
                                                bool neverDecrease = false);
                            DeviceMemoryCUDA(const DeviceMemoryCUDA&) = delete;
                            DeviceMemoryCUDA(DeviceMemoryCUDA&&) noexcept = default;
    DeviceMemoryCUDA&       operator=(const DeviceMemoryCUDA&) = delete;
    DeviceMemoryCUDA&       operator=(DeviceMemoryCUDA&&) noexcept = default;
                            ~DeviceMemoryCUDA();

    // Access
    explicit                operator Byte*();
    explicit                operator const Byte*() const;
    // Misc
    void                    ResizeBuffer(size_t newSize);
    size_t                  Size() const;
};

inline const GPUDeviceCUDA& DeviceLocalMemoryCUDA::Device() const
{
    return *gpu;
}

inline DeviceLocalMemoryCUDA::operator Byte*()
{
    return reinterpret_cast<Byte*>(dPtr);
}

inline DeviceLocalMemoryCUDA::operator const Byte*() const
{
    return reinterpret_cast<const Byte*>(dPtr);
}

inline HostLocalMemoryCUDA::operator Byte*()
{
    return reinterpret_cast<Byte*>(hPtr);
}

inline HostLocalMemoryCUDA::operator const Byte*() const
{
    return reinterpret_cast<const Byte*>(hPtr);
}

inline HostLocalAlignedMemoryCUDA::operator Byte* ()
{
    return reinterpret_cast<Byte*>(hPtr);
}

inline HostLocalAlignedMemoryCUDA::operator const Byte* () const
{
    return reinterpret_cast<const Byte*>(hPtr);
}

inline DeviceMemoryCUDA::operator Byte*()
{
    return std::bit_cast<Byte*>(mPtr);
}

inline DeviceMemoryCUDA::operator const Byte*() const
{
    return std::bit_cast<const Byte*>(mPtr);
}

static_assert(MemoryC<DeviceLocalMemoryCUDA>,
              "\"DeviceLocalMemoryCUDA\" does not satisfy memory concept.");
static_assert(MemoryC<HostLocalMemoryCUDA>,
              "\"HostLocalMemoryCUDA\" does not satisfy memory concept.");
static_assert(MemoryC<HostLocalAlignedMemoryCUDA>,
              "\"HostLocalAlignedMemoryCUDA\" does not satisfy memory concept.");
static_assert(MemoryC<DeviceMemoryCUDA>,
              "\"DeviceMemoryCUDA\" does not satisfy memory concept.");

}