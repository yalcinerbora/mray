#pragma once

#include "GPUTypes.h"

#ifdef MRAY_GPU_BACKEND_CUDA
    #include "CUDA/GPUSystemCUDA.h"

    // Alias the types
    using KernelCallParams          = mray::cuda::KernelCallParamsCUDA;
    using GPUSemaphoreView          = mray::cuda::GPUSemaphoreViewCUDA;
    using GPUDevice                 = mray::cuda::GPUDeviceCUDA;
    using GPUQueue                  = mray::cuda::GPUQueueCUDA;
    using GPUFence                  = mray::cuda::GPUFenceCUDA;
    using GPUSystem                 = mray::cuda::GPUSystemCUDA;

#elif defined GPU_BACKEND_HOST
    #error Not yet!
#else
    #error Please define a GPU Backend!
#endif

// After the inclusion of device specific implementation
// these should be defined
#ifndef MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
    #error "MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT" is not defined!
#endif

#ifndef MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM
    #error "MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM" is not defined!
#endif

#ifndef MRAY_GRID_CONSTANT
    #error "MRAY_GRID_CONSTANT" is not defined!
#endif

#ifndef MRAY_SHARED_MEMORY
    #error "MRAY_SHARED_MEMORY" is not defined!
#endif


class GPUQueueIteratorRoundRobin
{
    private:
    const GPUSystem&    gpuSystem;
    Vector2ui           indices = Vector2ui::Zero();

    public:
                        GPUQueueIteratorRoundRobin(const GPUSystem& s);
    const GPUQueue&     Queue() const;
    void                Next();
};

inline GPUQueueIteratorRoundRobin::GPUQueueIteratorRoundRobin(const GPUSystem& s)
    : gpuSystem(s)
{}

inline const GPUQueue& GPUQueueIteratorRoundRobin::Queue() const
{
    return gpuSystem.AllGPUs()[indices[0]]->GetComputeQueue(indices[1]);
}

inline void GPUQueueIteratorRoundRobin::Next()
{
    indices[1]++;
    if(indices[1] == ComputeQueuePerDevice)
    {
        indices[1] = 0;
        indices[0]++;
        if(indices[0] == gpuSystem.AllGPUs().size())
        {
            indices[0] = 0;
        }
    }
}