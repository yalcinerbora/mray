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

#include <limits>
#include <fstream>
#include <iostream>
#include <tuple>
#include <cassert>
#include <vector>

#include "Core/Definitions.h"
#include <cuda.h>

namespace mray::cuda
{

class GPUDeviceCUDA;
class GPUSystemCUDA;
class GPUQueueCUDA;

class DeviceLocalMemoryCUDA
{
    friend CUmemGenericAllocationHandle ToHandleCUDA(const DeviceLocalMemoryCUDA& mem);

    private:
    const GPUDeviceCUDA*            gpu;
    void*                           dPtr;
    size_t                          size;
    size_t                          allocSize;
    CUmemGenericAllocationHandle    memHandle;
    bool                            isTexMappable;

    protected:
    public:
    // Constructors & Destructor
                            DeviceLocalMemoryCUDA(const GPUDeviceCUDA& gpu);
                            DeviceLocalMemoryCUDA(const GPUDeviceCUDA& gpu, size_t sizeInBytes,
                                                  bool isUsedForTexMapping = false);
                            DeviceLocalMemoryCUDA(const DeviceLocalMemoryCUDA&);
                            DeviceLocalMemoryCUDA(DeviceLocalMemoryCUDA&&) noexcept;
                            ~DeviceLocalMemoryCUDA();
    DeviceLocalMemoryCUDA&  operator=(const DeviceLocalMemoryCUDA&);
    DeviceLocalMemoryCUDA&  operator=(DeviceLocalMemoryCUDA&&) noexcept;

    // Access
    explicit                operator void* ();
    explicit                operator const void* () const;

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
    const GPUSystemCUDA&    system;
    void*                   hPtr;
    void*                   dPtr;
    size_t                  size;

    public:
                            HostLocalMemoryCUDA(const GPUSystemCUDA& system);
                            HostLocalMemoryCUDA(const GPUSystemCUDA& system,
                                                size_t sizeInBytes);
                            HostLocalMemoryCUDA(const HostLocalMemoryCUDA&);
                            HostLocalMemoryCUDA(HostLocalMemoryCUDA&&) noexcept;
                            ~HostLocalMemoryCUDA();
    HostLocalMemoryCUDA&    operator=(const HostLocalMemoryCUDA&);
    HostLocalMemoryCUDA&    operator=(HostLocalMemoryCUDA&&) noexcept;

    // Access
    void*                   HostPtr();
    const void*             HostPtr() const;
    void*                   DevicePtr();
    const void*             DevicePtr() const;

    // Misc
    void                    ResizeBuffer(size_t newSize);
    size_t                  Size() const;
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
    using VARanges = std::pair<CUdeviceptr, size_t>;


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

    size_t                      FindCommonGranularity() const;
    size_t                      NextDeviceIndex();

    public:
        // Constructors & Destructor
                                DeviceMemoryCUDA(const std::vector<const GPUDeviceCUDA*>& devices,
                                                 size_t allocationGranularity,
                                                 size_t preReserveSize);
                                DeviceMemoryCUDA(const DeviceMemoryCUDA&) = delete;
                                DeviceMemoryCUDA(DeviceMemoryCUDA&&) noexcept = default;
                                ~DeviceMemoryCUDA();
        DeviceMemoryCUDA&       operator=(const DeviceMemoryCUDA&) = delete;
        DeviceMemoryCUDA&       operator=(DeviceMemoryCUDA&&) noexcept = default;

        // Access
        explicit                operator void*();
        explicit                operator const void*() const;
        // Misc
        void                    ResizeBuffer(size_t newSize);
        size_t                  Size() const;
};

inline const GPUDeviceCUDA& DeviceLocalMemoryCUDA::Device() const
{
    return *gpu;
}

inline DeviceLocalMemoryCUDA::operator void* ()
{
    return dPtr;
}

inline DeviceLocalMemoryCUDA::operator const void* () const
{
    return dPtr;
}

inline DeviceMemoryCUDA::operator void*()
{
    return std::bit_cast<void*>(mPtr);
}

inline DeviceMemoryCUDA::operator const void*() const
{
    return std::bit_cast<const void*>(mPtr);
}

inline CUmemGenericAllocationHandle ToHandleCUDA(const DeviceLocalMemoryCUDA& mem)
{
    return mem.memHandle;
}

}