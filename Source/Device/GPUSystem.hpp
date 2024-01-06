#pragma once

// This header hides the triple chevron calls
// from non "*.cu" files
#ifdef MRAY_GPU_BACKEND_CUDA
    #include "GPUSystemCUDA.hpp"
    #include "AlgorithmsCUDA.h"

    namespace DeviceAlgorithms = mray::cuda::algorithms;

//#elif defined MRAY_GPU_BACKEND_SYCL
//    // TODO:
//    //#include "GPUSystemSycl.hpp"
#else
    #error Please define a GPU Backend!
#endif
