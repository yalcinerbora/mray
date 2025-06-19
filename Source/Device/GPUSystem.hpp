#pragma once

// This header hides the triple chevron calls
// from non "*.cu" files
#ifdef MRAY_GPU_BACKEND_CUDA
    #include "CUDA/GPUSystemCUDA.hpp"
#elif defined MRAY_GPU_BACKEND_HIP
    #include "HIP/GPUSystemHIP.hpp"
#elif defined MRAY_GPU_BACKEND_CPU
    #include "CPU/GPUSystemCPU.hpp"
#else
    #error Please define a GPU Backend!
#endif
