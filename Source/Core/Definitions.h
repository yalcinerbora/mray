#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <array>
#include <string_view>
#include <algorithm>

#ifdef MRAY_GPU_BACKEND_CUDA

    #include <cuda_runtime.h>

    #define MRAY_HYBRID __host__ __device__
    #define MRAY_GPU __device__
    #define MRAY_HOST __host__
    #define MRAY_KERNEL __global__

    #ifdef MRAY_WINDOWS
        #define MRAY_GPU_INLINE __forceinline__
    #else
        #define MRAY_GPU_INLINE inline
    #endif

    #ifdef __CUDA_ARCH__
        #define MRAY_DEVICE_CODE_PATH
        #define MRAY_DEVICE_CODE_PATH_CUDA
    #endif

    #define NO_DISCARD [[nodiscard]]

#elif defined MRAY_GPU_BACKEND_HIP

    #include <hip/hip_runtime.h>

    #define MRAY_HYBRID __host__ __device__
    #define MRAY_GPU __device__
    #define MRAY_HOST __host__
    #define MRAY_KERNEL __global__

    #ifdef MRAY_WINDOWS
        #define MRAY_GPU_INLINE __forceinline__
    #else
        #define MRAY_GPU_INLINE inline
    #endif

    #ifdef __HIP_DEVICE_COMPILE__
        #define MRAY_DEVICE_CODE_PATH
        #define MRAY_DEVICE_CODE_PATH_HIP
    #endif

    // TODO: Hip does not like the order.
    // "__host__ [[nodiscard]]", but NVCC does not like
    // the order "[[nodiscard]] __host__", if I remember correctly
    // So, we will not use NO_DISCARD for now
    #define NO_DISCARD

#else

    #define MRAY_GPU_INLINE inline

    #define MRAY_HYBRID
    #define MRAY_GPU
    #define MRAY_HOST
    #define MRAY_KERNEL

    #define NO_DISCARD [[nodiscard]]

#endif

// Change to constexpr directly from this
#ifdef MRAY_DEBUG
    constexpr bool MRAY_IS_DEBUG = true;
#else
    constexpr bool MRAY_IS_DEBUG = false;
#endif

// We are on Device Compiler
#ifdef MRAY_DEVICE_CODE_PATH
    #define UNROLL_LOOP _Pragma("unroll")
    #define UNROLL_LOOP_COUNT(count) _Pragma("unroll")(count)

#else  // We are on Host Compiler
    #define UNROLL_LOOP
    #define UNROLL_LOOP_COUNT(count)
#endif

// Hybrid function inline
#define MRAY_CGPU_INLINE inline

// Comes from build system
#define MRAY_SPECTRA_PER_SPECTRUM 4

static constexpr int SpectraPerSpectrum = MRAY_SPECTRA_PER_SPECTRUM;
static_assert(SpectraPerSpectrum <= 4,
              "Spectra per spectrum can at most be 4"
              " (Due to Vector template at most hold 4 floats).");

// Until c++23, we custom define this
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

// Aliasing the byte here
using Byte = std::byte;


class EmptyType{};

// Idea is from here,
// https://stackoverflow.com/questions/47495384/default-lambda-as-templated-parameter-of-a-function
// One difference is that we just have a non-template operator()
struct EmptyFunctor
{
    constexpr void operator()() const {}
};

// Common enumerations that many
// internal libraries require

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

enum class MRayColorSpaceEnum : uint8_t
{
    // These are more or less mapped from OIIO/OCIO
    // https://opencolorio.readthedocs.io/en/latest/configurations/aces_1.0.3.html#colorspaces
    //
    // https://openimageio.readthedocs.io/en/latest/stdmetadata.html#color-information
    //
    // TODO: check D60 variants
    // TODO: Color spaces are overwhelming... Reiterate over these
    // Luminance linearity are not covered by this enum, a separate enum (or float)
    // will define if the color space is log or linear.
    // TODO: There are different eotfs etc. make a better approach
    MR_ACES2065_1,
    MR_ACES_CG,
    MR_REC_709,
    MR_REC_2020,
    MR_DCI_P3,
    MR_ADOBE_RGB,
    MR_DEFAULT,         // Use it "as is" disregard everything about color spaces
    //MR_SPECTRAL_PACK,   // Discrete spectral samples
    MR_END
};

// Texture Related
enum class MRayTextureInterpEnum : uint8_t
{
    MR_NEAREST,
    MR_LINEAR,
    MR_END
};

enum class MRayTextureEdgeResolveEnum : uint8_t
{
    MR_WRAP,
    MR_CLAMP,
    MR_MIRROR,
    MR_END
};

enum class MRayTextureReadMode : uint8_t
{
    // Directly read the texture
    MR_PASSTHROUGH,
    // Drop 1/2/3 Channels
    // i.e, 2channel pixel will be 1 channel when
    // MR_DROP_1 is stated
    // It is an error when MR_DROP_X (X >= channels)
    MR_DROP_1,
    MR_DROP_2,
    MR_DROP_3,
    // Special 2C->3C expansion for normals.
    // Basic mapping given Signed/Unsigned 2Channel type
    // For unsigned textures "[0, 1] -> [-1, 1]"
    // conversion will be applied (which is r/g * 2 - 1);
    // Find tangent space normal sqrt(1 - r^2 - g^2)
    MR_AS_3C_TS_NORMAL_BASIC,
    // Find tangent space normal via concentric octahedral
    // mapping of hemisphere (The two channels will be
    // used as uv coordinates of the mapping [-1, 1]
    // unsigned->signed conversion will be applied similarly.
    MR_AS_3C_TS_NORMAL_COOCTA,
    //
    MR_END
};

struct MRayDataTypeStringifier
{
    using enum MRayDataEnum;
    static constexpr std::array<const std::string_view, static_cast<size_t>(MR_END)> Names =
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
    static constexpr std::array<const std::string_view, static_cast<size_t>(MR_END)> Names =
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
    static constexpr std::array<const std::string_view, static_cast<size_t>(MR_END)> Names =
    {
        "ACES2065_1",
        "ACES_CG",
        "REC_709",
        "REC_2020",
        "DCI_P3",
        "ADOBE_RGB",
        "DEFAULT"
    };

    static constexpr std::string_view   ToString(MRayColorSpaceEnum e);
    static constexpr MRayColorSpaceEnum FromString(std::string_view);
};

struct MRayTextureInterpStringifier
{
    using enum MRayTextureInterpEnum;
    static constexpr std::array<const std::string_view, static_cast<size_t>(MR_END)> Names =
    {
        "Nearest",
        "Linear"
    };
    static constexpr std::string_view   ToString(MRayTextureInterpEnum e);
    static constexpr
    MRayTextureInterpEnum               FromString(std::string_view);
};

struct MRayTextureEdgeResolveStringifier
{
    using enum MRayTextureEdgeResolveEnum;
    static constexpr std::array<const std::string_view, static_cast<size_t>(MR_END)> Names =
    {
        "Wrap",
        "Clamp",
        "Mirror"
    };
    static constexpr std::string_view   ToString(MRayTextureEdgeResolveEnum e);
    static constexpr
    MRayTextureEdgeResolveEnum          FromString(std::string_view);
};

struct MRayTextureReadModeStringifier
{
    using enum MRayTextureReadMode;
    static constexpr std::array<const std::string_view, static_cast<size_t>(MR_END)> Names =
    {
        "Passthrough",
        "Drop1",
        "Drop2",
        "Drop3",
        "To3C_TsNormalBasic",
        "To3C_TsNormalCoOcta"
    };
    static constexpr std::string_view   ToString(MRayTextureReadMode e);
    static constexpr
    MRayTextureReadMode                 FromString(std::string_view);
};

// Block Compressed pixel "types"
// These are aligned with Vector<> template to match the types
// on templates
// "Tag" is here to differ types
template<unsigned int Channels,
         class PixelType,
         unsigned int BLOCK_SIZE,
         unsigned int TILE_SIZE,
         unsigned int Tag = 0>
struct BlockCompressedType
{
    using InnerType = PixelType;
    // This is "Dims" because this class emulates
    // Vector2/3/4 member constexpr variable "Dims".
    static constexpr size_t Dims = Channels;
    // Specific to block compression
    static constexpr size_t BlockSize = BLOCK_SIZE;
    static constexpr size_t TileSize = TILE_SIZE;
};

using PixelBC1  = BlockCompressedType<4, uint8_t,  8, 4, 0>;
using PixelBC2  = BlockCompressedType<4, uint8_t, 16, 4, 1>;
using PixelBC3  = BlockCompressedType<4, uint8_t, 16, 4, 3>;
using PixelBC4U = BlockCompressedType<1, uint8_t,  8, 4, 4>;
using PixelBC4S = BlockCompressedType<1, int8_t ,  8, 4, 5>;
using PixelBC5U = BlockCompressedType<2, uint8_t, 16, 4, 6>;
using PixelBC5S = BlockCompressedType<2, int8_t , 16, 4, 7>;
// We do not have half so put uint16_t
using PixelBC6U = BlockCompressedType<3, uint16_t, 16, 4, 8>;
using PixelBC6S = BlockCompressedType<3, int16_t , 16, 4, 9>;
using PixelBC7  = BlockCompressedType<4, uint8_t , 16, 4, 10>;

// Sanity check
static_assert(std::is_same_v<PixelBC1, PixelBC2>    == false);
static_assert(std::is_same_v<PixelBC2, PixelBC3>    == false);
static_assert(std::is_same_v<PixelBC3, PixelBC4U>   == false);
static_assert(std::is_same_v<PixelBC4U, PixelBC4S>  == false);
static_assert(std::is_same_v<PixelBC4S, PixelBC5U>  == false);
static_assert(std::is_same_v<PixelBC5U, PixelBC5S>  == false);
static_assert(std::is_same_v<PixelBC5S, PixelBC6U>  == false);
static_assert(std::is_same_v<PixelBC6U, PixelBC6S>  == false);
static_assert(std::is_same_v<PixelBC6S, PixelBC7>   == false);

// Block compressed concepts
template <class BCType>
static constexpr bool IsBlockCompressedPixel = (std::is_same_v<BCType, PixelBC1> ||
                                                std::is_same_v<BCType, PixelBC2> ||
                                                std::is_same_v<BCType, PixelBC3> ||
                                                std::is_same_v<BCType, PixelBC4U> ||
                                                std::is_same_v<BCType, PixelBC4S> ||
                                                std::is_same_v<BCType, PixelBC5U> ||
                                                std::is_same_v<BCType, PixelBC5S> ||
                                                std::is_same_v<BCType, PixelBC6U> ||
                                                std::is_same_v<BCType, PixelBC6S> ||
                                                std::is_same_v<BCType, PixelBC7>);

template<class BCType>
concept BlockCompressedPixelC = IsBlockCompressedPixel<BCType>;

template<class BCType>
concept NotBlockCompressedPixelC = !IsBlockCompressedPixel<BCType>;

constexpr std::string_view MRayDataTypeStringifier::ToString(MRayDataEnum e)
{
    assert(e < MRayDataEnum::MR_END);
    return Names[static_cast<uint32_t>(e)];
}

constexpr std::string_view MRayPixelTypeStringifier::ToString(MRayPixelEnum e)
{
    assert(e < MRayPixelEnum::MR_END);
    return Names[static_cast<uint32_t>(e)];
}

constexpr std::string_view MRayColorSpaceStringifier::ToString(MRayColorSpaceEnum e)
{
    assert(e < MRayColorSpaceEnum::MR_END);
    return Names[static_cast<uint32_t>(e)];
}

constexpr MRayColorSpaceEnum
MRayColorSpaceStringifier::FromString(std::string_view sv)
{
    auto loc = std::find(Names.cbegin(), Names.cend(), sv);
    return static_cast<MRayColorSpaceEnum>(std::distance(Names.cbegin(), loc));
}

constexpr std::string_view MRayTextureInterpStringifier::ToString(MRayTextureInterpEnum e)
{
    assert(e < MRayTextureInterpEnum::MR_END);
    return Names[static_cast<uint32_t>(e)];
}

constexpr MRayTextureInterpEnum
MRayTextureInterpStringifier::FromString(std::string_view sv)
{
    auto loc = std::find(Names.cbegin(), Names.cend(), sv);
    return static_cast<MRayTextureInterpEnum>(std::distance(Names.cbegin(), loc));
}

constexpr std::string_view MRayTextureEdgeResolveStringifier::ToString(MRayTextureEdgeResolveEnum e)
{
    assert(e < MRayTextureEdgeResolveEnum::MR_END);
    return Names[static_cast<uint32_t>(e)];
}

constexpr MRayTextureEdgeResolveEnum
MRayTextureEdgeResolveStringifier::FromString(std::string_view sv)
{
    auto loc = std::find(Names.cbegin(), Names.cend(), sv);
    return static_cast<MRayTextureEdgeResolveEnum>(std::distance(Names.cbegin(), loc));
}

constexpr std::string_view
MRayTextureReadModeStringifier::ToString(MRayTextureReadMode e)
{
    assert(e < MRayTextureReadMode::MR_END);
    return Names[static_cast<uint32_t>(e)];
}

constexpr MRayTextureReadMode
MRayTextureReadModeStringifier::FromString(std::string_view sv)
{
    auto loc = std::find(Names.cbegin(), Names.cend(), sv);
    return static_cast<MRayTextureReadMode>(std::distance(Names.cbegin(), loc));
}