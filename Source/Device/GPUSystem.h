#pragma once

#include <cstdint>
#include <string>
#include <array>
#include <vector>
#include "Core/Definitions.h"

struct DeviceError
{
    public:
    enum Type
    {
        OK,
        OLD_DRIVER,
        NO_DEVICE,
        OUT_OF_MEMORY,
        // End
        END
    };

    private:
    Type        type;

    public:
    // Constructors & Destructor
    DeviceError(Type = Type::OK);

    operator Type() const;
    operator std::string_view() const;
};

inline DeviceError::DeviceError(DeviceError::Type t)
    : type(t)
{}

#include "Core/Log.h"

inline DeviceError::operator Type() const
{
    std::string gg("gg");
    std::string fmt22 = fmt::format(fg(fmt::color::green),
                                  std::string("CUDA Failure"));

    MRAY_ERROR_LOG("{:s} {:s}", gg, gg);

    return type;
}

inline DeviceError::operator std::string_view() const
{
    using ErrorStrings = std::array<std::string_view, Type::END>;

    static constexpr const ErrorStrings Errors =
    {
        "OK",
        "Driver is not up-to-date",
        "No cuda capable device is found",
        "GPU is out of memory"
    };

    return Errors[static_cast<int>(type)];
}

static constexpr uint32_t WarpSize();
static constexpr uint32_t StaticThreadPerBlock1D();
static constexpr uint32_t TotalQueuePerDevice();

enum class DeviceQueueType
{
    NORMAL,
    FIRE_AND_FORGET,
    TAIL_LAUNCH
};

struct KernelAttributes
{
    size_t  localMemoryPerThread;
    size_t  constantMemorySize;
    int     maxDynamicSharedMemorySize;
    int     maxTBP;
    int     registerCountPerThread;
    size_t  staticSharedMemorySize;
};

// Generic Call Parameters
struct KernelCallParameters1D
{
    uint32_t gridSize;
    uint32_t blockSize;
    uint32_t blockId;
    uint32_t threadId;
};

#ifdef MRAY_GPU_BACKEND_CUDA
    #include "GPUSystemCUDA.hpp"

    // Alias the types
    using GPUDevice     = mray::cuda::GPUDeviceCUDA;
    using GPUQueue      = mray::cuda::GPUQueueCUDA;
    using GPUSystem     = mray::cuda::GPUSystemCUDA;


//#elif defined MRAY_GPU_BACKEND_SYCL
//    // TODO:
//    //#include "GPUSystemSycl.hpp"
#else
    #error Please define a GPU Backend!
#endif

// After the inclusion of device specific implementation
// these should be defined
#ifndef MRAY_DEVICE_LAUNCH_BOUNDS_1D
    #error "MRAY_DEVICE_LAUNCH_BOUNDS_1D" is not defined!
#endif

#ifndef MRAY_DEVICE_LAUNCH_BOUNDS
    #error "MRAY_DEVICE_LAUNCH_BOUNDS" is not defined!
#endif

#ifndef MRAY_GRID_CONSTANT
    #error "MRAY_GRID_CONSTANT" is not defined!
#endif


// Define an emulator type as well



