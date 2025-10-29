#pragma once

#ifdef MRAY_GPU_BACKEND_CUDA

    #include "CUDA/GPUWarpCUDA.h" // IWYU pragma: export

    namespace DeviceWarp
    {
        using namespace ::mray::cuda::warp;
    }

#elif defined MRAY_GPU_BACKEND_HIP

    #include "HIP/GPUWarpHIP.h" // IWYU pragma: export

    namespace DeviceWarp
    {
        using namespace ::mray::hip::warp;
    }

#elif defined MRAY_GPU_BACKEND_CPU

    #include "CPU/GPUWarpCPU.h" // IWYU pragma: export

    namespace DeviceWarp
    {
        using namespace ::mray::host::warp;
    }

#else
    #error Please define a GPU Backend!
#endif