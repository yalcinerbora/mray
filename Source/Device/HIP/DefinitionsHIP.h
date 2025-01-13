#pragma once

#include "Core/Log.h"
#include "Core/Error.h"

namespace mray::hip
{
    inline constexpr void GPUAssert(hipError_t code, const char* file, int line)
    {
        #ifndef __HIP_DEVICE_COMPILE__
            if(code == hipSuccess) return;

            MRAY_ERROR_LOG("{:s}: {:s} {:s}:{:d}",
                            fmt::format(fg(fmt::color::green),
                                        std::string("HIP Failure")),
                            hipGetErrorString(code), file, line);
            assert(false);
        #else
            if(code == hipSuccess) return;

            printf("%s: %s %s:%d", "HIP Failure",
                   hipGetErrorString(code), file, line);
            // TODO: hip does not have breakpoint or trap
            // directly aborting
            abort();
        #endif
    }

    inline constexpr void GPUMemThrow(hipError_t code, const char* file, int line)
    {
        if(code == hipErrorMemoryAllocation)
        {
            MRAY_ERROR_LOG("{:s}: {:s} {:s}:{:d}",
                           fmt::format(fg(fmt::color::green),
                                       std::string("HIP Failure")),
                           hipGetErrorString(code),
                           file,
                           line);

            throw MRayError("GPU Device is out of memory!");
        }
    }

    inline constexpr void GPUDriverAssert(hipError_t code, const char* file, int line)
    {
        if(code != hipSuccess)
        {
            std::string greenErrorCode = fmt::format(fg(fmt::color::green),
                                                     std::string("HIP Failure"));
            const char* errStr = hipGetErrorString(code);

            MRAY_ERROR_LOG("{:s}: {:s} {:s}:{:d}",
                           fmt::format(fg(fmt::color::green),
                                       std::string("HIP Failure")),
                           errStr, file, line);
            assert(false);
        }
    }

    inline constexpr void GPUDriverMemThrow(hipError_t code, const char* file, int line)
    {
        if(code == hipErrorMemoryAllocation)
        {
            std::string greenErrorCode = fmt::format(fg(fmt::color::green),
                                                     std::string("HIP Failure"));
            const char* errStr = hipGetErrorString(code);

            MRAY_ERROR_LOG("{:s}: {:s} {:s}:{:d}",
                           fmt::format(fg(fmt::color::green),
                                       std::string("HIP Failure")),
                           errStr, file, line);

            throw MRayError("GPU Device is out of memory!");
        }
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
