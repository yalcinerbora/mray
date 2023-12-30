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

#include "Core/Definitions.h"

namespace mray::cuda
{

class GPUDeviceCUDA;
class GPUSystemCUDA;
class GPUQueueCUDA;

class DeviceLocalMemoryCUDA
{
    private:
        void*                       dPtr;
        size_t                      size;
        const GPUDeviceCUDA*        gpu;

    protected:
    public:
        // Constructors & Destructor
                                    DeviceLocalMemoryCUDA(const GPUDeviceCUDA& gpu);
                                    DeviceLocalMemoryCUDA(const GPUDeviceCUDA& gpu, size_t sizeInBytes);
                                    DeviceLocalMemoryCUDA(const DeviceLocalMemoryCUDA&);
                                    DeviceLocalMemoryCUDA(DeviceLocalMemoryCUDA&&) noexcept;
                                    ~DeviceLocalMemoryCUDA();
        DeviceLocalMemoryCUDA&      operator=(const DeviceLocalMemoryCUDA&);
        DeviceLocalMemoryCUDA&      operator=(DeviceLocalMemoryCUDA&&) noexcept;

        // Access
        constexpr explicit          operator void* ();
        constexpr explicit          operator const void* () const;

        // Misc
        void                        EnlargeBuffer(size_t newSize);
        const GPUDeviceCUDA&        Device() const;
        size_t                      Size() const;
        void                        MigrateToOtherDevice(const GPUDeviceCUDA& deviceTo);
};

// Generic Device Memory (most of the cases this should be used)
// Automatic multi-device seperation etc.
class DeviceMemoryCUDA
{
    private:
        void*                       mPtr;
        size_t                      size;
        const GPUSystemCUDA&        system;

    protected:
    public:
        // Constructors & Destructor
                                    DeviceMemoryCUDA(const GPUSystemCUDA&);
                                    DeviceMemoryCUDA(const GPUSystemCUDA&, size_t sizeInBytes);
                                    DeviceMemoryCUDA(const DeviceMemoryCUDA&);
                                    DeviceMemoryCUDA(DeviceMemoryCUDA&&) noexcept;
                                    ~DeviceMemoryCUDA();
        DeviceMemoryCUDA&           operator=(const DeviceMemoryCUDA&);
        DeviceMemoryCUDA&           operator=(DeviceMemoryCUDA&&) noexcept;

        // Access
        constexpr explicit          operator void*();
        constexpr explicit          operator const void*() const;

        // Misc
        void                        EnlargeBuffer(size_t newSize);
        size_t                      Size() const;
};

inline const GPUDeviceCUDA& DeviceLocalMemoryCUDA::Device() const
{
    return *gpu;
}

inline constexpr DeviceLocalMemoryCUDA::operator void* ()
{
    return dPtr;
}

inline constexpr DeviceLocalMemoryCUDA::operator const void* () const
{
    return dPtr;
}

inline constexpr DeviceMemoryCUDA::operator void*()
{
    return mPtr;
}

inline constexpr DeviceMemoryCUDA::operator const void*() const
{
    return mPtr;
}

}