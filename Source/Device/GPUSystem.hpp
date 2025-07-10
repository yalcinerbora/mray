#pragma once

// This header hides the triple chevron calls
// from non "*.cu" files
#ifdef MRAY_GPU_BACKEND_CUDA
    #include "CUDA/GPUSystemCUDA.hpp"   // IWYU pragma: export
#elif defined MRAY_GPU_BACKEND_HIP
    #include "HIP/GPUSystemHIP.hpp"     // IWYU pragma: export
#elif defined MRAY_GPU_BACKEND_CPU
    #include "CPU/GPUSystemCPU.hpp"     // IWYU pragma: export
#else
    #error Please define a GPU Backend!
#endif
