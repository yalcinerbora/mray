#pragma once

#include <cassert>

#ifdef MRAY_GPU_BACKEND_CUDA

    #include <cuda_runtime.h>

    #define MRAY_HYBRID __host__ __device__
    #define MRAY_GPU __device__
    #define MRAY_HOST __host__
    #define MRAY_KERNEL __global__

    #define MRAY_GPU_INLINE __forceinline__

#else

    #define MRAY_GPU_INLINE inline

    #define MRAY_HYBRID
    #define MRAY_GPU
    #define MRAY_HOST
    #define MRAY_KERNEL

#endif

// Change to constexpr directly from this
#ifdef MRAY_DEBUG
    constexpr bool MRAY_IS_DEBUG = true;
#else
    constexpr bool MRAY_IS_DEBUG = false;
#endif // MRAY_


// We are on Device Compiler
#ifdef __CUDA_ARCH__
    #define UNROLL_LOOP _Pragma("unroll")
    #define UNROLL_LOOP_COUNT(count) _Pragma("unroll")(count)

#else  // We are on Host Compiler
    #define UNROLL_LOOP
    #define UNROLL_LOOP_COUNT(count)
#endif

// Hybrid function inline
#define MRAY_CGPU_INLINE inline
#define NO_DISCARD [[nodiscard]]

// Comes from build system
#define MRAY_SPECTRA_PER_SPECTRUM 4

static constexpr int SpectraPerSpectrum = MRAY_SPECTRA_PER_SPECTRUM;
static_assert(SpectraPerSpectrum <= 4,
              "Spectra per spectrum can at most be 4"
              " (Due to Vector template at most hold 4 floats).");

using Float = float;

using Byte = unsigned char;

class EmptyType{};

