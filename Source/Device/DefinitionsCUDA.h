#pragma once

#include "Core/Log.h"
#include "Core/Error.h"

namespace mray::cuda
{
    inline constexpr void GPUAssert(cudaError_t code, const char* file, int line)
    {
        #ifndef __CUDA_ARCH__
            if(code == cudaSuccess) return;

            MRAY_ERROR_LOG("{:s}: {:s} {:s}:{:d}",
                            fmt::format(fg(fmt::color::green),
                                        std::string("CUDA Failure")),
                            cudaGetErrorString(code), file, line);
            assert(false);
            __debugbreak();
        #else
            if(code == cudaSuccess) return;

            printf("%s: %s %s:%d", "CUDA Failure",
                   cudaGetErrorString(code), file, line);
            __brkpt();
        #endif
    }

    inline constexpr void GPUMemThrow(cudaError_t code, const char* file, int line)
    {
        if(code == cudaErrorMemoryAllocation)
        {
            MRAY_ERROR_LOG("{:s}: {:s} {:s}:{:d}",
                           fmt::format(fg(fmt::color::green),
                                       std::string("CUDA Failure")),
                           cudaGetErrorString(code),
                           file,
                           line);

            throw MRayError("GPU Device is out of memory!");
        }
    }

    inline constexpr void GPUDriverAssert(CUresult code, const char* file, int line)
    {
        if(code != CUDA_SUCCESS)
        {
            std::string greenErrorCode = fmt::format(fg(fmt::color::green),
                                                     std::string("CUDA Failure"));
            const char* errStr;
            cuGetErrorString(code, &errStr);

            MRAY_ERROR_LOG("{:s}: {:s} {:s}:{:d}",
                           fmt::format(fg(fmt::color::green),
                                       std::string("CUDA Failure")),
                           errStr, file, line);
            assert(false);
        }
    }

    inline constexpr void GPUDriverMemThrow(CUresult code, const char* file, int line)
    {
        if(code == CUDA_ERROR_OUT_OF_MEMORY)
        {
            std::string greenErrorCode = fmt::format(fg(fmt::color::green),
                                                     std::string("CUDA Failure"));
            const char* errStr;
            cuGetErrorString(code, &errStr);

            MRAY_ERROR_LOG("{:s}: {:s} {:s}:{:d}",
                           fmt::format(fg(fmt::color::green),
                                       std::string("CUDA Failure")),
                           errStr, file, line);

            throw MRayError("GPU Device is out of memory!");
        }
    }

}


#ifdef MRAY_DEBUG
    #define CUDA_CHECK(func) mray::cuda::GPUAssert((func), __FILE__, __LINE__)
    #define CUDA_DRIVER_CHECK(func) mray::cuda::GPUDriverAssert((func), __FILE__, __LINE__)

    #ifdef __CUDA_ARCH__
        #define CUDA_KERNEL_CHECK()
    #else
        #define CUDA_KERNEL_CHECK() \
                    CUDA_CHECK(cudaDeviceSynchronize()); \
                    CUDA_CHECK(cudaGetLastError())
    #endif

#else
    #define CUDA_CHECK(func) func
    #define CUDA_DRIVER_CHECK(func) func
    #define CUDA_KERNEL_CHECK()
#endif

#define CUDA_MEM_THROW(func) mray::cuda::GPUMemThrow((func), __FILE__, __LINE__)
#define CUDA_DRIVER_MEM_THROW(func) mray::cuda::GPUDriverMemThrow((func), __FILE__, __LINE__)
