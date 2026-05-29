#pragma once

#ifdef MRAY_GPU_BACKEND_CUDA

    #include "CUDA/GPUPrintCUDA.h" // IWYU pragma: export

    namespace Device
    {
        using namespace ::mray::cuda::print;
    }

#elif defined MRAY_GPU_BACKEND_HIP

    #include "HIP/GPUPrintHIP.h" // IWYU pragma: export

    namespace Device
    {
        using namespace ::mray::hip::print;
    }

#elif defined MRAY_GPU_BACKEND_CPU

    #include "CPU/GPUPrintCPU.h" // IWYU pragma: export

    namespace Device
    {
        using namespace ::mray::host::print;
    }

#else
    #error Please define a GPU Backend!
#endif