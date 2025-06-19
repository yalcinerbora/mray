#pragma once


#include "Core/Types.h"
#include <string_view>

#ifdef MRAY_GPU_BACKEND_CUDA

    namespace mray::cuda
    {
        struct KernelCallParamsCUDA;
        class GPUSemaphoreViewCUDA;
        class GPUDeviceCUDA;
        class GPUQueueCUDA;
        class GPUFenceCUDA;
        class GPUSystemCUDA;
        class DeviceMemoryCUDA;
        class DeviceLocalMemoryCUDA;
        class HostLocalMemoryCUDA;
        class TextureBackingMemoryCUDA;

        template<uint32_t DIM, class T>
        class TextureViewCUDA;

        template<uint32_t DIM, class T>
        class RWTextureViewCUDA;

        template<uint32_t DIM, class T>
        class RWTextureRefCUDA;

        template<uint32_t DIM, class T>
        class TextureCUDA;
    }

    // Alias the types
    using KernelCallParams      = mray::cuda::KernelCallParamsCUDA;
    using GPUSemaphore          = mray::cuda::GPUSemaphoreViewCUDA;
    using GPUDevice             = mray::cuda::GPUDeviceCUDA;
    using GPUQueue              = mray::cuda::GPUQueueCUDA;
    using GPUFence              = mray::cuda::GPUFenceCUDA;
    using GPUSystem             = mray::cuda::GPUSystemCUDA;
    using DeviceMemory          = mray::cuda::DeviceMemoryCUDA;
    using DeviceLocalMemory     = mray::cuda::DeviceLocalMemoryCUDA;
    using HostLocalMemory       = mray::cuda::HostLocalMemoryCUDA;
    using TextureBackingMemory  = mray::cuda::TextureBackingMemoryCUDA;

    template<uint32_t D, class T>
    using TextureView = mray::cuda::TextureViewCUDA<D, T>;
    template<uint32_t D, class T>
    using RWTextureView = mray::cuda::RWTextureViewCUDA<D, T>;
    template<uint32_t D, class T>
    using RWTextureRef = mray::cuda::RWTextureRefCUDA<D, T>;

    template<uint32_t D, class T>
    using Texture = mray::cuda::TextureCUDA<D, T>;

#elif defined(MRAY_GPU_BACKEND_HIP)

    namespace mray::hip
    {
        struct KernelCallParamsHIP;
        class GPUSemaphoreViewHIP;
        class GPUDeviceHIP;
        class GPUQueueHIP;
        class GPUFenceHIP;
        class GPUSystemHIP;
        class DeviceMemoryHIP;
        class DeviceLocalMemoryHIP;
        class HostLocalMemoryHIP;
        class TextureBackingMemoryHIP;

        template<uint32_t DIM, class T>
        class TextureViewHIP;

        template<uint32_t DIM, class T>
        class RWTextureViewHIP;

        template<uint32_t DIM, class T>
        class RWTextureRefHIP;

        template<uint32_t DIM, class T>
        class TextureHIP;
    }

    // Alias the types
    using KernelCallParams      = mray::hip::KernelCallParamsHIP;
    using GPUSemaphore          = mray::hip::GPUSemaphoreViewHIP;
    using GPUDevice             = mray::hip::GPUDeviceHIP;
    using GPUQueue              = mray::hip::GPUQueueHIP;
    using GPUFence              = mray::hip::GPUFenceHIP;
    using GPUSystem             = mray::hip::GPUSystemHIP;
    using DeviceMemory          = mray::hip::DeviceMemoryHIP;
    using DeviceLocalMemory     = mray::hip::DeviceLocalMemoryHIP;
    using HostLocalMemory       = mray::hip::HostLocalMemoryHIP;
    using TextureBackingMemory  = mray::hip::TextureBackingMemoryHIP;

    template<uint32_t D, class T>
    using TextureView = mray::hip::TextureViewHIP<D, T>;
    template<uint32_t D, class T>
    using RWTextureView = mray::hip::RWTextureViewHIP<D, T>;
    template<uint32_t D, class T>
    using RWTextureRef = mray::hip::RWTextureRefHIP<D, T>;

    template<uint32_t D, class T>
    using Texture = mray::hip::TextureHIP<D, T>;

#elif defined MRAY_GPU_BACKEND_CPU

    namespace mray::host
    {
        struct KernelCallParamsCPU;
        class GPUSemaphoreViewCPU;
        class GPUDeviceCPU;
        class GPUQueueCPU;
        class GPUFenceCPU;
        class GPUSystemCPU;
        class DeviceMemoryCPU;
        class DeviceLocalMemoryCPU;
        class HostLocalMemoryCPU;
        class TextureBackingMemoryCPU;

        template<uint32_t DIM, class T>
        class TextureViewCPU;

        template<uint32_t DIM, class T>
        class RWTextureViewCPU;

        template<uint32_t DIM, class T>
        class RWTextureRefCPU;

        template<uint32_t DIM, class T>
        class TextureCPU;
    }

    // Alias the types
    using KernelCallParams = mray::host::KernelCallParamsCPU;
    using GPUSemaphore = mray::host::GPUSemaphoreViewCPU;
    using GPUDevice = mray::host::GPUDeviceCPU;
    using GPUQueue = mray::host::GPUQueueCPU;
    using GPUFence = mray::host::GPUFenceCPU;
    using GPUSystem = mray::host::GPUSystemCPU;
    using DeviceMemory = mray::host::DeviceMemoryCPU;
    using DeviceLocalMemory = mray::host::DeviceLocalMemoryCPU;
    using HostLocalMemory = mray::host::HostLocalMemoryCPU;
    using TextureBackingMemory = mray::host::TextureBackingMemoryCPU;

    template<uint32_t D, class T>
    using TextureView = mray::host::TextureViewCPU<D, T>;
    template<uint32_t D, class T>
    using RWTextureView = mray::host::RWTextureViewCPU<D, T>;
    template<uint32_t D, class T>
    using RWTextureRef = mray::host::RWTextureRefCPU<D, T>;

    template<uint32_t D, class T>
    using Texture = mray::host::TextureCPU<D, T>;

#else
    #error Please define a GPU Backend!
#endif

class GPUQueueIteratorRoundRobin;
