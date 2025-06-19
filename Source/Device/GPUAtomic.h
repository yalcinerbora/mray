#pragma once

#ifdef MRAY_GPU_BACKEND_CUDA

    #include "CUDA/GPUAtomicCUDA.h"

    namespace DeviceAtomic
    {
        using namespace ::mray::cuda::atomic;
    }

#elif defined MRAY_GPU_BACKEND_HIP

    #include "HIP/GPUAtomicHIP.h"

    namespace DeviceAtomic
    {
        using namespace ::mray::hip::atomic;
    }
#elif defined MRAY_GPU_BACKEND_CPU
    #include "CPU/GPUAtomicCPU.h"

    namespace DeviceAtomic
    {
        using namespace ::mray::host::atomic;
    }

#else
    #error Please define a GPU Backend!
#endif