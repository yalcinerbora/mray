#pragma once

#include "Core/Log.h"
#include "Core/Error.h"

namespace mray::host
{
    inline constexpr void DeviceAssert(void*, const char*, int)
    {
    }

    inline constexpr void GPUMemThrow(void*, const char*, int)
    {
    }

    inline constexpr void GPUDriverAssert(void*, const char*, int)
    {

    }

    inline constexpr void GPUDriverMemThrow(void*, const char*, int)
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
