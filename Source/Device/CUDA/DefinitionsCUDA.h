#pragma once

#include "Core/Log.h"
#include "Core/Error.h"

namespace mray::cuda
{
    MRAY_HOST void GPUAssertHost(cudaError_t code, const char* file, int line);
    MRAY_HOST void GPUMemThrow(cudaError_t code, const char* file, int line);
    MRAY_HOST void GPUDriverAssert(CUresult code, const char* file, int line);
    MRAY_HOST void GPUDriverMemThrow(CUresult code, const char* file, int line);

    inline constexpr void GPUAssert(cudaError_t code, const char* file, int line)
    {
        #ifndef __CUDA_ARCH__
            if(code == cudaSuccess) return;
            GPUAssertHost(code, file, line);
        #else
            if(code == cudaSuccess) return;

            printf("%s: %s %s:%d", "CUDA Failure",
                   cudaGetErrorString(code), file, line);
            __brkpt();
        #endif
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
