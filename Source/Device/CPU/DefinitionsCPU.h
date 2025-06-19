#pragma once

#include "Core/Log.h"
#include "Core/Error.h"

namespace mray::host
{
    inline constexpr void DeviceAssert(void* code, const char* file, int line)
    {
    }

    inline constexpr void GPUMemThrow(void* code, const char* file, int line)
    {
    }

    inline constexpr void GPUDriverAssert(void* code, const char* file, int line)
    {

    }

    inline constexpr void GPUDriverMemThrow(void* code, const char* file, int line)
    {
    }

}

#ifdef MRAY_DEBUG
    #define HOST_CHECK(func)
    #define HOST_DRIVER_CHECK(func)
    #define HOST_KERNEL_CHECK()
#else
    #define HOST_CHECK(func)
    #define HOST_DRIVER_CHECK(func)
    #define HOST_KERNEL_CHECK()
#endif

#define HOST_MEM_THROW(func)
#define HOST_DRIVER_MEM_THROW(func)
