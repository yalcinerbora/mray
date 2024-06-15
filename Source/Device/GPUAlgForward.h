#pragma once

#ifdef MRAY_GPU_BACKEND_CUDA

    #include "CUDA/AlgForwardCUDA.h"

    namespace DeviceAlgorithms
    {
        inline namespace DeviceSpecific{ using namespace ::mray::cuda::algorithms; }
    }

#else
    #error Please define a GPU Backend!
#endif