#pragma once

#ifdef MRAY_GPU_BACKEND_CUDA

    #include "CUDA/AlgBinaryPartitionCUDA.h"

    namespace DeviceAlgorithms
    {
        inline namespace DeviceSpecific{ using namespace ::mray::cuda::algorithms; }
    }

#elif defined MRAY_GPU_BACKEND_HIP

    #include "HIP/AlgBinaryPartitionHIP.h"

    namespace DeviceAlgorithms
    {
        inline namespace DeviceSpecific{ using namespace ::mray::hip::algorithms; }
    }

#else
    #error Please define a GPU Backend!
#endif