#pragma once

#include "Core/Log.h"

namespace mray::cuda
{
    inline static constexpr void GPUAssert(cudaError_t code, const char* file, int line)
    {
        if(code != cudaSuccess)
        {
            std::string greenErrorCode = fmt::format(fg(fmt::color::green),
                                                     std::string("CUDA Failure"));
            MRAY_ERROR_LOG("{:s}: {:s} {:s}:{:d}",
                           fmt::format(fg(fmt::color::green),
                                       std::string("CUDA Failure")),
                           cudaGetErrorString(code),
                           file,
                           line);
            assert(false);
        }
    }

    inline static constexpr void GPUMemoryCheck(cudaError_t code, const char* file, int line)
    {
        if(code == cudaErrorMemoryAllocation)
        {
            MRAY_ERROR_LOG("{:s}: {:s} {:s}:{:d}",
                           fmt::format(fg(fmt::color::green),
                                       std::string("CUDA Failure")),
                           cudaGetErrorString(code),
                           file,
                           line);

            throw DeviceError(DeviceError::OUT_OF_MEMORY);
        }
    }
}


#ifdef MRAY_DEBUG
    #define CUDA_CHECK(func) mray::cuda::GPUAssert((func), __FILE__, __LINE__)
    #define CUDA_CHECK_ERROR(err) mray::cuda::GPUAssert(err, __FILE__, __LINE__)
    #define CUDA_MEM_CHECK(func) mray::cuda::GPUMemoryCheck((func), __FILE__, __LINE__)
    #define CUDA_KERNEL_CHECK() \
                    CUDA_CHECK(cudaDeviceSynchronize()); \
                    CUDA_CHECK(cudaGetLastError())
#else
    #define CUDA_CHECK_ERROR(err)
    #define CUDA_KERNEL_PRINTF()
    #define CUDA_CHECK(func) func
    #define CUDA_MEM_CHECK(func) func
    #define CUDA_KERNEL_CHECK()
#endif

