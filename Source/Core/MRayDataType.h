#pragma once

#include "TypeFinder.h"
#include "Definitions.h"
#include "Vector.h"
#include "Matrix.h"
#include "Quaternion.h"
#include "NormTypes.h"
#include "Ray.h"
#include "AABB.h"

// This implementation is quite taxing on compilation times.
// It is used to "automatically" create switch/case statements
// for the given Pixel/Data type.
// Here is the generated code (Only copy pasted the Pixel portion)
// https://godbolt.org/z/d99s1v7zP
//
// MSVC is slightly better at optimizing these
// ; merges same case statements into one.
// Clang/GCC does not do that (Maybe it does not worth it?)


namespace MRayDataDetail
{
    using M = TypeFinder::E_TMapper<MRayDataEnum>;
    using Mapper = M::Map
    <
        typename M:: template ETPair<MRayDataEnum::MR_INT8,         int8_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2C,    Vector2c>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3C,    Vector3c>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4C,    Vector4c>,

        typename M:: template ETPair<MRayDataEnum::MR_INT16,        int16_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2S,    Vector2s>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3S,    Vector3s>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4S,    Vector4s>,

        typename M:: template ETPair<MRayDataEnum::MR_INT32,        int32_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2I,    Vector2i>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3I,    Vector3i>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4I,    Vector4i>,

        typename M:: template ETPair<MRayDataEnum::MR_INT64,        int64_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2L,    Vector2l>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3L,    Vector3l>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4L,    Vector4l>,

        typename M:: template ETPair<MRayDataEnum::MR_UINT8,        uint8_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2UC,   Vector2uc>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3UC,   Vector3uc>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4UC,   Vector4uc>,

        typename M:: template ETPair<MRayDataEnum::MR_UINT16,       uint16_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2US,   Vector2us>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3US,   Vector3us>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4US,   Vector4us>,

        typename M:: template ETPair<MRayDataEnum::MR_UINT32,       uint32_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2UI,   Vector2ui>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3UI,   Vector3ui>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4UI,   Vector4ui>,

        typename M:: template ETPair<MRayDataEnum::MR_UINT64,       uint64_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2UL,   Vector2ul>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3UL,   Vector3ul>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4UL,   Vector4ul>,

        typename M:: template ETPair<MRayDataEnum::MR_FLOAT,        float>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2,     Vector2>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3,     Vector3>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4,     Vector4>,

        typename M:: template ETPair<MRayDataEnum::MR_QUATERNION,   Quaternion>,
        typename M:: template ETPair<MRayDataEnum::MR_MATRIX_4x4,   Matrix4x4>,
        typename M:: template ETPair<MRayDataEnum::MR_MATRIX_3x3,   Matrix2x2>,
        typename M:: template ETPair<MRayDataEnum::MR_AABB3,        AABB3>,
        typename M:: template ETPair<MRayDataEnum::MR_RAY,          Ray>,

        typename M:: template ETPair<MRayDataEnum::MR_UNORM_4x8,    UNorm4x8>,
        typename M:: template ETPair<MRayDataEnum::MR_UNORM_2x16,   UNorm2x16>,
        typename M:: template ETPair<MRayDataEnum::MR_SNORM_4x8,    SNorm4x8>,
        typename M:: template ETPair<MRayDataEnum::MR_SNORM_2x16,   SNorm2x16>,

        typename M:: template ETPair<MRayDataEnum::MR_STRING,       std::string>,
        typename M:: template ETPair<MRayDataEnum::MR_BOOL,         bool>
    >;

    template<MRayDataEnum E>
    using Type = typename Mapper::Find<E>;

}

namespace MRayPixelDetail
{
    using M = TypeFinder::E_TMapper<MRayPixelEnum>;
    using Mapper = M::Map
    <
        typename M:: template ETPair<MRayPixelEnum::MR_R8_UNORM,         uint8_t>,
        typename M:: template ETPair<MRayPixelEnum::MR_RG8_UNORM,        Vector2uc>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGB8_UNORM,       Vector3uc>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGBA8_UNORM,      Vector4uc>,

        typename M:: template ETPair<MRayPixelEnum::MR_R16_UNORM,        uint16_t>,
        typename M:: template ETPair<MRayPixelEnum::MR_RG16_UNORM,       Vector2us>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGB16_UNORM,      Vector3us>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGBA16_UNORM,     Vector4us>,

        typename M:: template ETPair<MRayPixelEnum::MR_R8_SNORM,         int8_t>,
        typename M:: template ETPair<MRayPixelEnum::MR_RG8_SNORM,        Vector2c>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGB8_SNORM,       Vector3c>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGBA8_SNORM,      Vector4c>,

        typename M:: template ETPair<MRayPixelEnum::MR_R16_SNORM,        int16_t>,
        typename M:: template ETPair<MRayPixelEnum::MR_RG16_SNORM,       Vector2s>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGB16_SNORM,      Vector3s>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGBA16_SNORM,     Vector4s>,

        // TODO: Find a proper half lib and change this
        typename M:: template ETPair<MRayPixelEnum::MR_R_HALF,           uint16_t>,
        typename M:: template ETPair<MRayPixelEnum::MR_RG_HALF,          Vector2us>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGB_HALF,         Vector3us>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGBA_HALF,        Vector4us>,

        typename M:: template ETPair<MRayPixelEnum::MR_R_FLOAT,          float>,
        typename M:: template ETPair<MRayPixelEnum::MR_RG_FLOAT,         Vector2f>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGB_FLOAT,        Vector3f>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGBA_FLOAT,       Vector4f>,

        typename M:: template ETPair<MRayPixelEnum::MR_BC1_UNORM,        PixelBC1>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC2_UNORM,        PixelBC2>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC3_UNORM,        PixelBC3>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC4_UNORM,        PixelBC4U>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC4_SNORM,        PixelBC4S>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC5_UNORM,        PixelBC5U>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC5_SNORM,        PixelBC5S>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC6H_UFLOAT,      PixelBC6U>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC6H_SFLOAT,      PixelBC6S>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC7_UNORM,        PixelBC7>
    >;

    template<MRayPixelEnum E>
    using Type = typename Mapper::Find<E>;

}

//namespace MRayDataDetail
//{
//
//
//
//    using namespace TypeFinder;
//    constexpr auto MRayDataEnumTypeMapping = std::make_tuple
//    (
//        ValKeyTypePair<MRayDataEnum::MR_INT8,         int8_t>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2C,    Vector2c>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3C,    Vector3c>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4C,    Vector4c>{},
//
//        ValKeyTypePair<MRayDataEnum::MR_INT16,        int16_t>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2S,    Vector2s>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3S,    Vector3s>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4S,    Vector4s>{},
//
//        ValKeyTypePair<MRayDataEnum::MR_INT32,        int32_t>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2I,    Vector2i>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3I,    Vector3i>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4I,    Vector4i>{},
//
//        ValKeyTypePair<MRayDataEnum::MR_INT64,        int64_t>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2L,    Vector2l>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3L,    Vector3l>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4L,    Vector4l>{},
//
//        ValKeyTypePair<MRayDataEnum::MR_UINT8,        uint8_t>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2UC,   Vector2uc>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3UC,   Vector3uc>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4UC,   Vector4uc>{},
//
//        ValKeyTypePair<MRayDataEnum::MR_UINT16,       uint16_t>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2US,   Vector2us>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3US,   Vector3us>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4US,   Vector4us>{},
//
//        ValKeyTypePair<MRayDataEnum::MR_UINT32,       uint32_t>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2UI,   Vector2ui>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3UI,   Vector3ui>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4UI,   Vector4ui>{},
//
//        ValKeyTypePair<MRayDataEnum::MR_UINT64,       uint64_t>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2UL,   Vector2ul>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3UL,   Vector3ul>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4UL,   Vector4ul>{},
//
//        ValKeyTypePair<MRayDataEnum::MR_FLOAT,        float>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2,     Vector2>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3,     Vector3>{},
//        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4,     Vector4>{},
//
//        ValKeyTypePair<MRayDataEnum::MR_QUATERNION,   Quaternion>{},
//        ValKeyTypePair<MRayDataEnum::MR_MATRIX_4x4,   Matrix4x4>{},
//        ValKeyTypePair<MRayDataEnum::MR_MATRIX_3x3,   Matrix2x2>{},
//        ValKeyTypePair<MRayDataEnum::MR_AABB3,        AABB3>{},
//        ValKeyTypePair<MRayDataEnum::MR_RAY,          Ray>{},
//
//        ValKeyTypePair<MRayDataEnum::MR_UNORM_4x8,    UNorm4x8>{},
//        ValKeyTypePair<MRayDataEnum::MR_UNORM_2x16,   UNorm2x16>{},
//        ValKeyTypePair<MRayDataEnum::MR_SNORM_4x8,    SNorm4x8>{},
//        ValKeyTypePair<MRayDataEnum::MR_SNORM_2x16,   SNorm2x16>{},
//
//        ValKeyTypePair<MRayDataEnum::MR_STRING,       std::string>{},
//        ValKeyTypePair<MRayDataEnum::MR_BOOL,         bool>{}
//    );
//
//    template<MRayDataEnum E>
//    constexpr auto DataEnumToType()
//    {
//        return GetTupleElement<E>(MRayDataEnumTypeMapping);
//    }
//}

//namespace MRayPixelDetail
//{
//    using namespace TypeFinder;
//    constexpr auto MRayPixelEnumTypeMapping = std::make_tuple
//    (
//        ValKeyTypePair<MRayPixelEnum::MR_R8_UNORM,         uint8_t>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RG8_UNORM,        Vector2uc>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RGB8_UNORM,       Vector3uc>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RGBA8_UNORM,      Vector4uc>{},
//
//        ValKeyTypePair<MRayPixelEnum::MR_R16_UNORM,        uint16_t>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RG16_UNORM,       Vector2us>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RGB16_UNORM,      Vector3us>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RGBA16_UNORM,     Vector4us>{},
//
//        ValKeyTypePair<MRayPixelEnum::MR_R8_SNORM,         int8_t>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RG8_SNORM,        Vector2c>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RGB8_SNORM,       Vector3c>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RGBA8_SNORM,      Vector4c>{},
//
//        ValKeyTypePair<MRayPixelEnum::MR_R16_SNORM,        int16_t>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RG16_SNORM,       Vector2s>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RGB16_SNORM,      Vector3s>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RGBA16_SNORM,     Vector4s>{},
//
//        // TODO: Find a proper half lib and change this
//        ValKeyTypePair<MRayPixelEnum::MR_R_HALF,           uint16_t>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RG_HALF,          Vector2us>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RGB_HALF,         Vector3us>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RGBA_HALF,        Vector4us>{},
//
//        ValKeyTypePair<MRayPixelEnum::MR_R_FLOAT,          float>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RG_FLOAT,         Vector2f>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RGB_FLOAT,        Vector3f>{},
//        ValKeyTypePair<MRayPixelEnum::MR_RGBA_FLOAT,       Vector4f>{},
//
//        ValKeyTypePair<MRayPixelEnum::MR_BC1_UNORM,        PixelBC1>{},
//        ValKeyTypePair<MRayPixelEnum::MR_BC2_UNORM,        PixelBC2>{},
//        ValKeyTypePair<MRayPixelEnum::MR_BC3_UNORM,        PixelBC3>{},
//        ValKeyTypePair<MRayPixelEnum::MR_BC4_UNORM,        PixelBC4U>{},
//        ValKeyTypePair<MRayPixelEnum::MR_BC4_SNORM,        PixelBC4S>{},
//        ValKeyTypePair<MRayPixelEnum::MR_BC5_UNORM,        PixelBC5U>{},
//        ValKeyTypePair<MRayPixelEnum::MR_BC5_SNORM,        PixelBC5S>{},
//        ValKeyTypePair<MRayPixelEnum::MR_BC6H_UFLOAT,      PixelBC6U>{},
//        ValKeyTypePair<MRayPixelEnum::MR_BC6H_SFLOAT,      PixelBC6S>{},
//        ValKeyTypePair<MRayPixelEnum::MR_BC7_UNORM,        PixelBC7>{}
//    );
//
//    template<MRayPixelEnum E>
//    constexpr auto PixelEnumToType()
//    {
//        return GetTupleElement<E>(MRayPixelEnumTypeMapping);
//    }
//}

template<MRayDataEnum E>
struct MRayDataType
{
    using Type = MRayDataDetail::Type<E>;
    static constexpr size_t Size       = sizeof(Type);
    static constexpr size_t Alignment  = alignof(Type);
    static constexpr MRayDataEnum Name = E;
};

template<MRayPixelEnum E>
struct MRayPixelType
{
    private:
    template<class T>
    requires (std::is_integral_v<T> || std::is_floating_point_v<T>)
    static constexpr size_t FindChannelCount();
    template<class T>
    requires (!std::is_integral_v<T> && !std::is_floating_point_v<T>)
    static constexpr size_t FindChannelCount();

    public:
    using Type = MRayPixelDetail::Type<E>;
    static constexpr size_t ChannelCount = FindChannelCount<Type>();
    static constexpr MRayPixelEnum Name = E;
    static constexpr bool IsBCPixel = IsBlockCompressedPixel<Type>;
};

template<MRayDataEnum L, MRayDataEnum R>
constexpr bool operator==(MRayDataType<L>,
                          MRayDataType<R>)
{
    return L == R;
}

template<MRayPixelEnum E>
template<class T>
requires (std::is_integral_v<T> || std::is_floating_point_v<T>)
constexpr size_t MRayPixelType<E>::FindChannelCount()
{
    return 1u;
}

template<MRayPixelEnum E>
template<class T>
requires (!std::is_integral_v<T> && !std::is_floating_point_v<T>)
constexpr size_t MRayPixelType<E>::FindChannelCount()
{
    return Type::Dims;
}

template<MRayPixelEnum L, MRayPixelEnum R>
constexpr bool operator==(MRayPixelType<L>,
                          MRayPixelType<R>)
{
    return L == R;
}


// Lets see how good are the compilers are
// This is used to generate switch/case code
// For type reading on scene loader
using MRayDataTypeBase = Variant
<
    MRayDataType<MRayDataEnum::MR_INT8>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2C>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3C>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4C>,

    MRayDataType<MRayDataEnum::MR_INT16>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2S>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3S>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4S>,

    MRayDataType<MRayDataEnum::MR_INT32>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2I>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3I>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4I>,

    MRayDataType<MRayDataEnum::MR_INT64>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2L>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3L>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4L>,

    MRayDataType<MRayDataEnum::MR_UINT8>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2UC>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3UC>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4UC>,

    MRayDataType<MRayDataEnum::MR_UINT16>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2US>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3US>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4US>,

    MRayDataType<MRayDataEnum::MR_UINT32>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2UI>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3UI>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4UI>,

    MRayDataType<MRayDataEnum::MR_UINT64>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2UL>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3UL>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4UL>,

    MRayDataType<MRayDataEnum::MR_FLOAT>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4>,

    MRayDataType<MRayDataEnum::MR_QUATERNION>,
    MRayDataType<MRayDataEnum::MR_MATRIX_4x4>,
    MRayDataType<MRayDataEnum::MR_MATRIX_3x3>,
    MRayDataType<MRayDataEnum::MR_AABB3>,
    MRayDataType<MRayDataEnum::MR_RAY>,

    MRayDataType<MRayDataEnum::MR_UNORM_4x8>,
    MRayDataType<MRayDataEnum::MR_UNORM_2x16>,
    MRayDataType<MRayDataEnum::MR_SNORM_4x8>,
    MRayDataType<MRayDataEnum::MR_SNORM_2x16>,

    MRayDataType<MRayDataEnum::MR_STRING>,
    MRayDataType<MRayDataEnum::MR_BOOL>

    // TODO: Report MSVC about the excessive recursion
    // Adding more types makes the code unable to compile.
>;

using MRayPixelTypeBase = Variant
<
    MRayPixelType<MRayPixelEnum::MR_R8_UNORM>,
    MRayPixelType<MRayPixelEnum::MR_RG8_UNORM>,
    MRayPixelType<MRayPixelEnum::MR_RGB8_UNORM>,
    MRayPixelType<MRayPixelEnum::MR_RGBA8_UNORM>,

    MRayPixelType<MRayPixelEnum::MR_R16_UNORM>,
    MRayPixelType<MRayPixelEnum::MR_RG16_UNORM>,
    MRayPixelType<MRayPixelEnum::MR_RGB16_UNORM>,
    MRayPixelType<MRayPixelEnum::MR_RGBA16_UNORM>,

    MRayPixelType<MRayPixelEnum::MR_R8_SNORM>,
    MRayPixelType<MRayPixelEnum::MR_RG8_SNORM>,
    MRayPixelType<MRayPixelEnum::MR_RGB8_SNORM>,
    MRayPixelType<MRayPixelEnum::MR_RGBA8_SNORM>,

    MRayPixelType<MRayPixelEnum::MR_R16_SNORM>,
    MRayPixelType<MRayPixelEnum::MR_RG16_SNORM>,
    MRayPixelType<MRayPixelEnum::MR_RGB16_SNORM>,
    MRayPixelType<MRayPixelEnum::MR_RGBA16_SNORM>,

    MRayPixelType<MRayPixelEnum::MR_R_HALF>,
    MRayPixelType<MRayPixelEnum::MR_RG_HALF>,
    MRayPixelType<MRayPixelEnum::MR_RGB_HALF>,
    MRayPixelType<MRayPixelEnum::MR_RGBA_HALF>,

    MRayPixelType<MRayPixelEnum::MR_R_FLOAT>,
    MRayPixelType<MRayPixelEnum::MR_RG_FLOAT>,
    MRayPixelType<MRayPixelEnum::MR_RGB_FLOAT>,
    MRayPixelType<MRayPixelEnum::MR_RGBA_FLOAT>,

    MRayPixelType<MRayPixelEnum::MR_BC1_UNORM>,
    MRayPixelType<MRayPixelEnum::MR_BC2_UNORM>,
    MRayPixelType<MRayPixelEnum::MR_BC3_UNORM>,
    MRayPixelType<MRayPixelEnum::MR_BC4_UNORM>,
    MRayPixelType<MRayPixelEnum::MR_BC4_SNORM>,
    MRayPixelType<MRayPixelEnum::MR_BC5_UNORM>,
    MRayPixelType<MRayPixelEnum::MR_BC5_SNORM>,
    MRayPixelType<MRayPixelEnum::MR_BC6H_UFLOAT>,
    MRayPixelType<MRayPixelEnum::MR_BC6H_SFLOAT>,
    MRayPixelType<MRayPixelEnum::MR_BC7_UNORM>
>;

struct MRayDataTypeRT : public MRayDataTypeBase
{
    using enum MRayDataEnum;
    MRayDataEnum  Name() const;
    size_t        Size() const;
    size_t        Alignment() const;

    using MRayDataTypeBase::MRayDataTypeBase;
};

struct MRayPixelTypeRT : public MRayPixelTypeBase
{
    using enum MRayPixelEnum;
    MRayPixelEnum     Name() const;
    size_t            ChannelCount() const;
    bool              IsBlockCompressed() const;

    using MRayPixelTypeBase::MRayPixelTypeBase;
};
