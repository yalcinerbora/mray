#pragma once

#ifdef MRAY_GPU_BACKEND_CUDA

    #include "CUDA/AlgScanCUDA.h" // IWYU pragma: export

    namespace DeviceAlgorithms
    {
        inline namespace DeviceSpecific{ using namespace ::mray::cuda::algorithms; }
    }

#elif defined MRAY_GPU_BACKEND_HIP

    #include "HIP/AlgScanHIP.h" // IWYU pragma: export

    namespace DeviceAlgorithms
    {
        inline namespace DeviceSpecific{ using namespace ::mray::hip::algorithms; }
    }

#elif defined MRAY_GPU_BACKEND_CPU
    #include "CPU/AlgScanCPU.h" // IWYU pragma: export

    namespace DeviceAlgorithms
    {
        inline namespace DeviceSpecific{ using namespace ::mray::host::algorithms; }
    }

#else
    #error Please define a GPU Backend!
#endif