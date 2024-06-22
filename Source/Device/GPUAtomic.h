#pragma once

#ifdef MRAY_GPU_BACKEND_CUDA

#include "CUDA/GPUAtomicCUDA.h"

namespace DeviceAtomic
{
    using namespace ::mray::cuda::atomic;
}

#else

#error Please define a GPU Backend!

#endif