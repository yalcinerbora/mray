#pragma once


#include "Core/Definitions.h"
#include "Core/Error.h"

#include "GPUTypes.h"
#include "GPUSystemForward.h"

#ifdef MRAY_GPU_BACKEND_CUDA
    #include "CUDA/GPUSystemCUDA.h"
    #include "CUDA/TextureCUDA.h"
    #include "CUDA/DeviceMemoryCUDA.h"

    // Alias the types
    using KernelCallParams      = mray::cuda::KernelCallParamsCUDA;
    using GPUSemaphore          = mray::cuda::GPUSemaphoreCUDA;
    using GPUDevice             = mray::cuda::GPUDeviceCUDA;
    using GPUQueue              = mray::cuda::GPUQueueCUDA;
    using GPUFence              = mray::cuda::GPUFenceCUDA;
    using GPUSystem             = mray::cuda::GPUSystemCUDA;
    using DeviceMemory          = mray::cuda::DeviceMemoryCUDA;
    using DeviceLocalMemory     = mray::cuda::DeviceLocalMemoryCUDA;
    using HostLocalMemory       = mray::cuda::HostLocalMemoryCUDA;
    using TextureBackingMemory  = mray::cuda::TextureBackingMemoryCUDA;

    template<uint32_t DIM, class T>
    using TextureView = mray::cuda::TextureViewCUDA<DIM, T>;

    template<uint32_t DIM, class T>
    using Texture = mray::cuda::TextureCUDA<DIM, T>;

//#elif defined MRAY_GPU_BACKEND_SYCL
//    // TODO:
//    //#include "GPUSystemSycl.h"
#elif defined GPU_BACKEND_HOST

    //DeviceVisit()

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

#ifndef MRAY_DEVICE_BLOCK_SYNC
    #error "MRAY_DEVICE_BLOCK_SYNC" is not defined!
#endif

#ifndef MRAY_SHARED_MEMORY
    #error "MRAY_SHARED_MEMORY" is not defined!
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
    return gpuSystem.AllGPUs()[indices[0]]->GetQueue(indices[1]);
}

inline void GPUQueueIteratorRoundRobin::Next()
{
    indices[1]++;
    if(indices[1] == QueuePerDevice)
    {
        indices[1] = 0;
        indices[0]++;
        if(indices[0] == gpuSystem.AllGPUs().size())
        {
            indices[0] = 0;
        }
    }
}

namespace DeviceDebug
{

using namespace std::string_view_literals;

template<class T>
static void DumpGPUMemToStream(std::ostream& s,
                               Span<const T> data,
                               const GPUQueue& queue,
                               std::string_view seperator)
{
    std::vector<T> hostBuffer(data.size());
    queue.MemcpyAsync(Span<T>(hostBuffer), data);
    queue.Barrier().Wait();

    for(const T& d : hostBuffer)
    {
        s << MRAY_FORMAT("{}{:s}", d, seperator);
    }
}

template<class T>
static void DumpGPUMemToFile(const std::string& fName,
                             Span<const T> data,
                             const GPUQueue& queue,
                             std::string_view seperator)
{
    std::ofstream file(fName);
    DumpGPUMemToStream(file, data, queue, seperator);
}

template<class T>
static void DumpGPUMemToStdOut(std::string_view header,
                               Span<const T> data,
                               const GPUQueue& queue,
                               std::string_view seperator)
{
    if(!header.empty()) MRAY_DEBUG_LOG("{}", header);
    DumpGPUMemToStream(std::cout, data, queue, seperator);
    if(!header.empty()) MRAY_DEBUG_LOG("-------------");
}

}