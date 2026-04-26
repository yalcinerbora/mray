#pragma once

#include "Core/Log.h"
#include "Core/Error.h"

namespace mray::hip
{

    MRAY_HOST void GPUAssertHost(hipError_t code, const char* file, int line);
    MRAY_HOST void GPUMemThrow(hipError_t code, const char* file, int line);
    MRAY_HOST void GPUDriverAssert(hipError_t code, const char* file, int line);
    MRAY_HOST void GPUDriverMemThrow(hipError_t code, const char* file, int line);

    inline constexpr void GPUAssert(hipError_t code, const char* file, int line)
    {
        #ifndef __HIP_DEVICE_COMPILE__
            if(code == hipSuccess) return;
            GPUAssertHost(code, file, line);
        #else
            if(code == hipSuccess) return;

            printf("%s: %s %s:%d", "HIP Failure",
                   hipGetErrorString(code), file, line);
            // TODO: hip does not have breakpoint or trap
            // directly aborting
            abort();
        #endif
    }
}


#ifdef MRAY_DEBUG
    #define HIP_CHECK(func) mray::hip::GPUAssert((func), __FILE__, __LINE__)
    #define HIP_DRIVER_CHECK(func) mray::hip::GPUDriverAssert((func), __FILE__, __LINE__)

    #ifdef __HIP_DEVICE_COMPILE__
        #define HIP_KERNEL_CHECK()
    #else
        #define HIP_KERNEL_CHECK() \
                    HIP_CHECK(hipDeviceSynchronize()); \
                    HIP_CHECK(hipGetLastError())
    #endif

#else
    #define HIP_CHECK(func) (void)func
    #define HIP_DRIVER_CHECK(func) (void)func
    #define HIP_KERNEL_CHECK()
#endif

#define HIP_MEM_THROW(func) mray::hip::GPUMemThrow((func), __FILE__, __LINE__)
#define HIP_DRIVER_MEM_THROW(func) mray::hip::GPUDriverMemThrow((func), __FILE__, __LINE__)
