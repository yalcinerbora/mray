#pragma once

#include <cassert>
#include <cstddef>

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

// TODO: This should come from CMake
using Float = float;

using Byte = std::byte;

class EmptyType{};

// Main data types that the system accepts
// Type erasure stuff
//namespace MRayDataDetail
//{

enum class MRayDataEnum : uint32_t
{
    MR_CHAR,
    MR_VECTOR_2C,
    MR_VECTOR_3C,
    MR_VECTOR_4C,

    MR_SHORT,
    MR_VECTOR_2S,
    MR_VECTOR_3S,
    MR_VECTOR_4S,

    MR_INT,
    MR_VECTOR_2I,
    MR_VECTOR_3I,
    MR_VECTOR_4I,

    MR_UCHAR,
    MR_VECTOR_2UC,
    MR_VECTOR_3UC,
    MR_VECTOR_4UC,

    MR_USHORT,
    MR_VECTOR_2US,
    MR_VECTOR_3US,
    MR_VECTOR_4US,

    MR_UINT,
    MR_VECTOR_2UI,
    MR_VECTOR_3UI,
    MR_VECTOR_4UI,

    // TODO: Add half here (find a good lib)

    MR_FLOAT,
    MR_VECTOR_2F,
    MR_VECTOR_3F,
    MR_VECTOR_4F,

    MR_DOUBLE,
    MR_VECTOR_2D,
    MR_VECTOR_3D,
    MR_VECTOR_4D,

    // Default floating point related types
    // of the system
    MR_DEFAULT_FLT,
    MR_VECTOR_2,
    MR_VECTOR_3,
    MR_VECTOR_4,

    MR_QUATERNION,
    MR_MATRIX_4x4,
    MR_MATRIX_3x3,
    MR_AABB3_ENUM, // Clashes with the typename
    MR_RAY,

    // Normalized Types
    MR_UNORM_4x8,
    MR_UNORM_2x16,
    MR_SNORM_4x8,
    MR_SNORM_2x16,

    MR_UNORM_8x8,
    MR_UNORM_4x16,
    MR_UNORM_2x32,
    MR_SNORM_8x8,
    MR_SNORM_4x16,
    MR_SNORM_2x32,

    MR_UNORM_16x8,
    MR_UNORM_8x16,
    MR_UNORM_4x32,
    MR_UNORM_2x64,
    MR_SNORM_16x8,
    MR_SNORM_8x16,
    MR_SNORM_4x32,
    MR_SNORM_2x64,

    MR_UNORM_32x8,
    MR_UNORM_16x16,
    MR_UNORM_8x32,
    MR_UNORM_4x64,
    MR_SNORM_32x8,
    MR_SNORM_16x16,
    MR_SNORM_8x32,
    MR_SNORM_4x64,

    MR_STRING
};
