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

#else
    #error Please define a GPU Backend!
#endif

class GPUQueueIteratorRoundRobin;
