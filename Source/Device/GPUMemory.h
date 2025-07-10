#pragma once

#include "GPUSystemForward.h"

#ifdef MRAY_GPU_BACKEND_CUDA
    #include "CUDA/DeviceMemoryCUDA.h"

    // Alias the types
    using DeviceMemory              = mray::cuda::DeviceMemoryCUDA;
    using DeviceLocalMemory         = mray::cuda::DeviceLocalMemoryCUDA;
    using HostLocalMemory           = mray::cuda::HostLocalMemoryCUDA;
    using HostLocalAlignedMemory    = mray::cuda::HostLocalAlignedMemoryCUDA;

#elif defined MRAY_GPU_BACKEND_HIP
    #include "HIP/DeviceMemoryHIP.h"

    // Alias the types
    using DeviceMemory              = mray::hip::DeviceMemoryHIP;
    using DeviceLocalMemory         = mray::hip::DeviceLocalMemoryHIP;
    using HostLocalMemory           = mray::hip::HostLocalMemoryHIP;
    using HostLocalAlignedMemory    = mray::hip::HostLocalAlignedMemoryHIP;

#elif defined MRAY_GPU_BACKEND_CPU
    #include "CPU/DeviceMemoryCPU.h" // IWYU pragma: export

    // Alias the types
    using DeviceMemory              = mray::host::DeviceMemoryCPU;
    using DeviceLocalMemory         = mray::host::DeviceLocalMemoryCPU;
    using HostLocalMemory           = mray::host::HostLocalMemoryCPU;
    using HostLocalAlignedMemory    = mray::host::HostLocalAlignedMemoryCPU;

#else
    #error Please define a GPU Backend!
#endif

// Concept Checks
template<class MemType>
concept DeviceMemBaseC = requires(MemType m)
{
    {m.ResizeBuffer(size_t{})} -> std::same_as<void>;
    {m.Size()} -> std::same_as<size_t>;
};

template<class MemType>
concept DeviceLocalMemC = requires(MemType m,
                                   const GPUDevice& dev,
                                   const GPUQueue& queue)
{
    requires DeviceMemBaseC<MemType>;
    {m.MigrateToOtherDevice(dev)} -> std::same_as<void>;
    {m.Device()} -> std::same_as<const GPUDevice&>;
};

template<class MemType>
concept DeviceMemC = requires(MemType m)
{
    requires DeviceMemBaseC<MemType>;
};

static_assert(DeviceMemC<DeviceMemory>,
              "Device memory does not satisfy its concept!");

static_assert(DeviceLocalMemC<DeviceLocalMemory>,
              "Device local memory does not satisfy its concept!");