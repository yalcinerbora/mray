#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <array>
#include <string_view>

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

// Untill c++23, we custom define this
// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2674r0.pdf
// Directly from the above paper
template <class T>
concept ImplicitLifetimeC = requires()
{
    std::disjunction
    <
        std::is_scalar<T>,
        std::is_array<T>,
        std::is_aggregate<T>,
        std::conjunction
        <
            std::is_trivially_destructible<T>,
            std::disjunction
            <
                std::is_trivially_default_constructible<T>,
                std::is_trivially_copy_constructible<T>,
                std::is_trivially_move_constructible<T>
            >
        >
    >::value;
};

// TODO: This should come from CMake
using Float = float;

using Byte = std::byte;

class EmptyType{};

// Main data types that the system accepts
// Type erasure stuff
//namespace MRayDataDetail
//{

enum class MRayDataEnum : uint16_t
{
    MR_INT8,
    MR_VECTOR_2C,
    MR_VECTOR_3C,
    MR_VECTOR_4C,

    MR_INT16,
    MR_VECTOR_2S,
    MR_VECTOR_3S,
    MR_VECTOR_4S,

    MR_INT32,
    MR_VECTOR_2I,
    MR_VECTOR_3I,
    MR_VECTOR_4I,

    MR_INT64,
    MR_VECTOR_2L,
    MR_VECTOR_3L,
    MR_VECTOR_4L,

    MR_UINT8,
    MR_VECTOR_2UC,
    MR_VECTOR_3UC,
    MR_VECTOR_4UC,

    MR_UINT16,
    MR_VECTOR_2US,
    MR_VECTOR_3US,
    MR_VECTOR_4US,

    MR_UINT32,
    MR_VECTOR_2UI,
    MR_VECTOR_3UI,
    MR_VECTOR_4UI,

    MR_UINT64,
    MR_VECTOR_2UL,
    MR_VECTOR_3UL,
    MR_VECTOR_4UL,

    // Default floating point related types
    // of the system
    // TODO: Add other type of floats as well
    MR_FLOAT,
    MR_VECTOR_2,
    MR_VECTOR_3,
    MR_VECTOR_4,

    MR_QUATERNION,
    MR_MATRIX_4x4,
    MR_MATRIX_3x3,
    MR_AABB3,
    MR_RAY,

    // Normalized Types
    MR_UNORM_4x8,
    MR_UNORM_2x16,
    MR_SNORM_4x8,
    MR_SNORM_2x16,

    MR_STRING,
    MR_BOOL,

    MR_END
};

enum class MRayPixelEnum : uint16_t
{
    // UNORMS
    MR_R8_UNORM,
    MR_RG8_UNORM,
    MR_RGB8_UNORM,
    MR_RGBA8_UNORM,

    MR_R16_UNORM,
    MR_RG16_UNORM,
    MR_RGB16_UNORM,
    MR_RGBA16_UNORM,

    // SNORMS
    MR_R8_SNORM,
    MR_RG8_SNORM,
    MR_RGB8_SNORM,
    MR_RGBA8_SNORM,

    MR_R16_SNORM,
    MR_RG16_SNORM,
    MR_RGB16_SNORM,
    MR_RGBA16_SNORM,

    // FLOAT
    MR_R_HALF,
    MR_RG_HALF,
    MR_RGB_HALF,
    MR_RGBA_HALF,

    MR_R_FLOAT,
    MR_RG_FLOAT,
    MR_RGB_FLOAT,
    MR_RGBA_FLOAT,

    // Graphics Related Compressed Images
    // https://docs.microsoft.com/en-us/windows/win32/direct3d11/texture-block-compression-in-direct3d-11
    // CUDA also support these but it is pain in the ass to properly load into
    // textureObjects
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g6b3a50368a0aa592f65e928adca9b929
    MR_BC1_UNORM,
    MR_BC2_UNORM,
    MR_BC3_UNORM,
    MR_BC4_UNORM,
    MR_BC4_SNORM,
    MR_BC5_UNORM,
    MR_BC5_SNORM,
    MR_BC6H_UFLOAT,
    MR_BC6H_SFLOAT,
    MR_BC7_UNORM,

    MR_END
};

enum class MRayColorSpaceEnum : uint32_t
{
    // These are more or less mapped from OIIO/OCIO
    // https://opencolorio.readthedocs.io/en/latest/configurations/aces_1.0.3.html#colorspaces
    //
    // https://openimageio.readthedocs.io/en/latest/stdmetadata.html#color-information
    //
    // TODO: check D60 variants
    // TODO: Color spaces are overwheling... Reiterate over these
    // Luminance linearity are not covered by this enum, a seperate enum (or float)
    // will define if the color space is log or linear
    MR_ACES2065_1,
    MR_ACES_CG,
    MR_REC_709,
    MR_REC_2020,
    MR_DCI_P3,
    MR_DEFAULT, // Use it "as is" disregard everything about color spaces

    MR_END
};

struct MRayDataTypeStringifier
{
    using enum MRayDataEnum;
    static constexpr std::array<std::string_view, static_cast<size_t>(MR_END)> Names =
    {
        "INT8",
        "VECTOR_2C",
        "VECTOR_3C",
        "VECTOR_4C",
        "INT16",
        "VECTOR_2S",
        "VECTOR_3S",
        "VECTOR_4S",
        "INT32",
        "VECTOR_2I",
        "VECTOR_3I",
        "VECTOR_4I",
        "INT64",
        "VECTOR_2L",
        "VECTOR_3L",
        "VECTOR_4L",
        "UINT8",
        "VECTOR_2UC",
        "VECTOR_3UC",
        "VECTOR_4UC",
        "UINT16",
        "VECTOR_2US",
        "VECTOR_3US",
        "VECTOR_4US",
        "UINT32",
        "VECTOR_2UI",
        "VECTOR_3UI",
        "VECTOR_4UI",
        "UINT64",
        "VECTOR_2UL",
        "VECTOR_3UL",
        "VECTOR_4UL",
        "FLOAT",
        "VECTOR_2",
        "VECTOR_3",
        "VECTOR_4",
        "QUATERNION",
        "MATRIX_4x4",
        "MATRIX_3x3",
        "AABB3",
        "RAY",
        "UNORM_4x8",
        "UNORM_2x16",
        "SNORM_4x8",
        "SNORM_2x16",
        "STRING",
        "BOOL"
    };
    static constexpr std::string_view ToString(MRayDataEnum e);
};

struct MRayPixelTypeStringifier
{
    using enum MRayPixelEnum;
    static constexpr std::array<std::string_view, static_cast<size_t>(MR_END)> Names =
    {
        "MR_R8_UNORM",
        "MR_RG8_UNORM",
        "MR_RGB8_UNORM",
        "MR_RGBA8_UNORM",
        "MR_R16_UNORM",
        "MR_RG16_UNORM",
        "MR_RGB16_UNORM",
        "MR_RGBA16_UNORM",
        "MR_R8_SNORM",
        "MR_RG8_SNORM",
        "MR_RGB8_SNORM",
        "MR_RGBA8_SNORM",
        "MR_R16_SNORM",
        "MR_RG16_SNORM",
        "MR_RGB16_SNORM",
        "MR_RGBA16_SNORM",
        "MR_R_HALF",
        "MR_RG_HALF",
        "MR_RGB_HALF",
        "MR_RGBA_HALF",
        "MR_R_FLOAT",
        "MR_RG_FLOAT",
        "MR_RGB_FLOAT",
        "MR_RGBA_FLOAT",
        "MR_BC1_UNORM",
        "MR_BC2_UNORM",
        "MR_BC3_UNORM",
        "MR_BC4_UNORM",
        "MR_BC4_SNORM",
        "MR_BC5_UNORM",
        "MR_BC5_SNORM",
        "MR_BC6H_UFLOAT",
        "MR_BC6H_SFLOAT",
        "MR_BC7_UNORM"
    };
    static constexpr std::string_view ToString(MRayPixelEnum e);
};

struct MRayColorSpaceStringifier
{
    using enum MRayColorSpaceEnum;
    static constexpr std::array<std::string_view, static_cast<size_t>(MR_END)> Names =
    {
        "ACES2065_1",
        "ACES_CG",
        "REC_709",
        "REC_2020",
        "DCI_P3",
        "DEFAULT",
    };
    static constexpr std::string_view ToString(MRayColorSpaceEnum e);
};

constexpr std::string_view MRayDataTypeStringifier::ToString(MRayDataEnum e)
{
    return Names[static_cast<uint32_t>(e)];
}

constexpr std::string_view MRayPixelTypeStringifier::ToString(MRayPixelEnum e)
{
    return Names[static_cast<uint32_t>(e)];
}

constexpr std::string_view MRayColorSpaceStringifier::ToString(MRayColorSpaceEnum e)
{
    return Names[static_cast<uint32_t>(e)];
}