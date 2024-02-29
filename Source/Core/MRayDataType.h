#pragma once

#include "TypeFinder.h"
#include "Definitions.h"
#include "Error.h"
#include "Vector.h"
#include "Matrix.h"
#include "Quaternion.h"
#include "NormTypes.h"
#include "Ray.h"
#include "AABB.h"

// Block Compressed pixel "types"
// These are aligned with Vector<> template to match the types
// on templates
// "Tag" is here to differ types
template<unsigned int Channel, class T, unsigned int Tag = 0>
struct BlockCompressedType
{
    using InnerType = T;
    static constexpr size_t Dims = Channel;
};

using PixelBC1  = BlockCompressedType<4, uint8_t, 0>;
using PixelBC2  = BlockCompressedType<4, uint8_t, 1>;
using PixelBC3  = BlockCompressedType<4, uint8_t, 2>;
using PixelBC4U = BlockCompressedType<1, uint8_t>;
using PixelBC4S = BlockCompressedType<1, int8_t>;
using PixelBC5U = BlockCompressedType<2, uint16_t>;
using PixelBC5S = BlockCompressedType<2, int16_t>;
using PixelBC6U = BlockCompressedType<3, uint16_t>;
using PixelBC6S = BlockCompressedType<3, int16_t>;
using PixelBC7  = BlockCompressedType<4, uint8_t, 3>;

// Sanity check
static_assert(std::is_same_v<PixelBC1, PixelBC2> == false);
static_assert(std::is_same_v<PixelBC2, PixelBC3> == false);
static_assert(std::is_same_v<PixelBC3, PixelBC7> == false);

namespace MRayDataDetail
{
    using namespace TypeFinder;
    constexpr auto MRayDataEnumTypeMapping = std::make_tuple
    (
        ValKeyTypePair<MRayDataEnum::MR_CHAR,         int8_t>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2C,    Vector2c>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3C,    Vector3c>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4C,    Vector4c>{},

        ValKeyTypePair<MRayDataEnum::MR_SHORT,        int16_t>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2S,    Vector2s>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3S,    Vector3s>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4S,    Vector4s>{},

        ValKeyTypePair<MRayDataEnum::MR_INT,          int32_t>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2I,    Vector2i>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3I,    Vector3i>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4I,    Vector4i>{},

        ValKeyTypePair<MRayDataEnum::MR_UCHAR,        uint8_t>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2UC,   Vector2uc>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3UC,   Vector3uc>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4UC,   Vector4uc>{},

        ValKeyTypePair<MRayDataEnum::MR_USHORT,       uint16_t>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2US,   Vector2us>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3US,   Vector3us>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4US,   Vector4us>{},

        ValKeyTypePair<MRayDataEnum::MR_UINT,         uint32_t>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2UI,   Vector2ui>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3UI,   Vector3ui>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4UI,   Vector4ui>{},

        ValKeyTypePair<MRayDataEnum::MR_FLOAT,        float>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2F,    Vector2f>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3F,    Vector3f>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4F,    Vector4f>{},

        ValKeyTypePair<MRayDataEnum::MR_DOUBLE,       double>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2D,    Vector2d>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3D,    Vector3d>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4D,    Vector4d>{},

        ValKeyTypePair<MRayDataEnum::MR_DEFAULT_FLT,  Float>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_2,     Vector2>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_3,     Vector3>{},
        ValKeyTypePair<MRayDataEnum::MR_VECTOR_4,     Vector4>{},

        ValKeyTypePair<MRayDataEnum::MR_QUATERNION,   Quaternion>{},
        ValKeyTypePair<MRayDataEnum::MR_MATRIX_4x4,   Matrix4x4>{},
        ValKeyTypePair<MRayDataEnum::MR_MATRIX_3x3,   Matrix2x2>{},
        ValKeyTypePair<MRayDataEnum::MR_AABB3_ENUM,   AABB3>{},
        ValKeyTypePair<MRayDataEnum::MR_RAY,          Ray>{},

        ValKeyTypePair<MRayDataEnum::MR_UNORM_4x8,    UNorm4x8>{},
        ValKeyTypePair<MRayDataEnum::MR_UNORM_2x16,   UNorm2x16>{},
        ValKeyTypePair<MRayDataEnum::MR_SNORM_4x8,    SNorm4x8>{},
        ValKeyTypePair<MRayDataEnum::MR_SNORM_2x16,   SNorm2x16>{},

        ValKeyTypePair<MRayDataEnum::MR_UNORM_8x8,    UNorm8x8>{},
        ValKeyTypePair<MRayDataEnum::MR_UNORM_4x16,   UNorm4x16>{},
        ValKeyTypePair<MRayDataEnum::MR_UNORM_2x32,   UNorm2x32>{},
        ValKeyTypePair<MRayDataEnum::MR_SNORM_8x8,    SNorm8x8>{},
        ValKeyTypePair<MRayDataEnum::MR_SNORM_4x16,   SNorm4x16>{},
        ValKeyTypePair<MRayDataEnum::MR_SNORM_2x32,   SNorm2x32>{},

        ValKeyTypePair<MRayDataEnum::MR_UNORM_16x8,   UNorm16x8>{},
        ValKeyTypePair<MRayDataEnum::MR_UNORM_8x16,   UNorm8x16>{},
        ValKeyTypePair<MRayDataEnum::MR_UNORM_4x32,   UNorm4x32>{},
        ValKeyTypePair<MRayDataEnum::MR_UNORM_2x64,   UNorm2x64>{},
        ValKeyTypePair<MRayDataEnum::MR_SNORM_16x8,   SNorm16x8>{},
        ValKeyTypePair<MRayDataEnum::MR_SNORM_8x16,   SNorm8x16>{},
        ValKeyTypePair<MRayDataEnum::MR_SNORM_4x32,   SNorm4x32>{},
        ValKeyTypePair<MRayDataEnum::MR_SNORM_2x64,   SNorm2x64>{},

        ValKeyTypePair<MRayDataEnum::MR_UNORM_32x8,   UNorm32x8>{},
        ValKeyTypePair<MRayDataEnum::MR_UNORM_16x16,  UNorm16x16>{},
        ValKeyTypePair<MRayDataEnum::MR_UNORM_8x32,   UNorm8x32>{},
        ValKeyTypePair<MRayDataEnum::MR_UNORM_4x64,   UNorm4x64>{},
        ValKeyTypePair<MRayDataEnum::MR_SNORM_32x8,   SNorm32x8>{},
        ValKeyTypePair<MRayDataEnum::MR_SNORM_16x16,  SNorm16x16>{},
        ValKeyTypePair<MRayDataEnum::MR_SNORM_8x32,   SNorm8x32>{},
        ValKeyTypePair<MRayDataEnum::MR_SNORM_4x64,   SNorm4x64>{},

        ValKeyTypePair<MRayDataEnum::MR_STRING,       std::string>{}
    );

    template<MRayDataEnum E>
    constexpr auto DataEnumToType()
    {
        return GetTupleElement<E>(MRayDataEnumTypeMapping);
    }
}

namespace MRayPixelDetail
{
    using namespace TypeFinder;
    constexpr auto MRayPixelEnumTypeMapping = std::make_tuple
    (
        ValKeyTypePair<MRayPixelEnum::MR_R8_UNORM,         uint8_t>{},
        ValKeyTypePair<MRayPixelEnum::MR_RG8_UNORM,        Vector2uc>{},
        ValKeyTypePair<MRayPixelEnum::MR_RGB8_UNORM,       Vector3uc>{},
        ValKeyTypePair<MRayPixelEnum::MR_RGBA8_UNORM,      Vector4uc>{},

        ValKeyTypePair<MRayPixelEnum::MR_R16_UNORM,        uint16_t>{},
        ValKeyTypePair<MRayPixelEnum::MR_RG16_UNORM,       Vector2us>{},
        ValKeyTypePair<MRayPixelEnum::MR_RGB16_UNORM,      Vector3us>{},
        ValKeyTypePair<MRayPixelEnum::MR_RGBA16_UNORM,     Vector4us>{},

        ValKeyTypePair<MRayPixelEnum::MR_R8_SNORM,         int8_t>{},
        ValKeyTypePair<MRayPixelEnum::MR_RG8_SNORM,        Vector2c>{},
        ValKeyTypePair<MRayPixelEnum::MR_RGB8_SNORM,       Vector3c>{},
        ValKeyTypePair<MRayPixelEnum::MR_RGBA8_SNORM,      Vector4c>{},

        ValKeyTypePair<MRayPixelEnum::MR_R16_SNORM,        int16_t>{},
        ValKeyTypePair<MRayPixelEnum::MR_RG16_SNORM,       Vector2s>{},
        ValKeyTypePair<MRayPixelEnum::MR_RGB16_SNORM,      Vector3s>{},
        ValKeyTypePair<MRayPixelEnum::MR_RGBA16_SNORM,     Vector4s>{},

        // TODO: Find a proper half lib and change this
        ValKeyTypePair<MRayPixelEnum::MR_R_HALF,           uint16_t>{},
        ValKeyTypePair<MRayPixelEnum::MR_RG_HALF,          Vector2us>{},
        ValKeyTypePair<MRayPixelEnum::MR_RGB_HALF,         Vector3us>{},
        ValKeyTypePair<MRayPixelEnum::MR_RGBA_HALF,        Vector4us>{},

        ValKeyTypePair<MRayPixelEnum::MR_R_FLOAT,          float>{},
        ValKeyTypePair<MRayPixelEnum::MR_RG_FLOAT,         Vector2f>{},
        ValKeyTypePair<MRayPixelEnum::MR_RGB_FLOAT,        Vector3f>{},
        ValKeyTypePair<MRayPixelEnum::MR_RGBA_FLOAT,       Vector4f>{},

        ValKeyTypePair<MRayPixelEnum::MR_BC1_UNORM,        PixelBC1>{},
        ValKeyTypePair<MRayPixelEnum::MR_BC2_UNORM,        PixelBC2>{},
        ValKeyTypePair<MRayPixelEnum::MR_BC3_UNORM,        PixelBC3>{},
        ValKeyTypePair<MRayPixelEnum::MR_BC4_UNORM,        PixelBC4U>{},
        ValKeyTypePair<MRayPixelEnum::MR_BC4_SNORM,        PixelBC4S>{},
        ValKeyTypePair<MRayPixelEnum::MR_BC5_UNORM,        PixelBC5U>{},
        ValKeyTypePair<MRayPixelEnum::MR_BC5_SNORM,        PixelBC5S>{},
        ValKeyTypePair<MRayPixelEnum::MR_BC6H_UFLOAT,      PixelBC6U>{},
        ValKeyTypePair<MRayPixelEnum::MR_BC6H_SFLOAT,      PixelBC6S>{},
        ValKeyTypePair<MRayPixelEnum::MR_BC7_UNORM,        PixelBC7>{}
    );

    template<MRayPixelEnum E>
    constexpr auto PixelEnumToType()
    {
        return GetTupleElement<E>(MRayPixelEnumTypeMapping);
    }
}

template<MRayDataEnum E>
struct MRayDataType
{
    using Type = typename decltype(MRayDataDetail::DataEnumToType<E>())::Result;
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
    using Type = typename decltype(MRayPixelDetail::PixelEnumToType<E>())::Result;
    static constexpr size_t ChannelCount = FindChannelCount<Type>();
    static constexpr MRayPixelEnum Name = E;
};

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

// Lets see how good are the compilers are
// This is used to generate switch/case code
// For type reading on scene loader
using MRayDataTypeBase = Variant
<
    MRayDataType<MRayDataEnum::MR_CHAR>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2C>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3C>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4C>,

    MRayDataType<MRayDataEnum::MR_SHORT>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2S>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3S>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4S>,

    MRayDataType<MRayDataEnum::MR_INT>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2I>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3I>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4I>,

    MRayDataType<MRayDataEnum::MR_UCHAR>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2UC>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3UC>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4UC>,

    MRayDataType<MRayDataEnum::MR_USHORT>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2US>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3US>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4US>,

    MRayDataType<MRayDataEnum::MR_UINT>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2UI>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3UI>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4UI>,

    MRayDataType<MRayDataEnum::MR_FLOAT>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2F>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3F>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4F>,

    MRayDataType<MRayDataEnum::MR_DOUBLE>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2D>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3D>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4D>,

    MRayDataType<MRayDataEnum::MR_DEFAULT_FLT>,
    MRayDataType<MRayDataEnum::MR_VECTOR_2>,
    MRayDataType<MRayDataEnum::MR_VECTOR_3>,
    MRayDataType<MRayDataEnum::MR_VECTOR_4>,

    MRayDataType<MRayDataEnum::MR_QUATERNION>,
    MRayDataType<MRayDataEnum::MR_MATRIX_4x4>,
    MRayDataType<MRayDataEnum::MR_MATRIX_3x3>,
    MRayDataType<MRayDataEnum::MR_AABB3_ENUM>,
    MRayDataType<MRayDataEnum::MR_RAY>,

    MRayDataType<MRayDataEnum::MR_UNORM_4x8>,
    MRayDataType<MRayDataEnum::MR_UNORM_2x16>,
    MRayDataType<MRayDataEnum::MR_SNORM_4x8>,
    MRayDataType<MRayDataEnum::MR_SNORM_2x16>,

    MRayDataType<MRayDataEnum::MR_UNORM_8x8>,
    MRayDataType<MRayDataEnum::MR_UNORM_4x16>,
    MRayDataType<MRayDataEnum::MR_UNORM_2x32>,
    MRayDataType<MRayDataEnum::MR_SNORM_8x8>,
    MRayDataType<MRayDataEnum::MR_SNORM_4x16>,
    MRayDataType<MRayDataEnum::MR_SNORM_2x32>,

    MRayDataType<MRayDataEnum::MR_UNORM_16x8>,
    MRayDataType<MRayDataEnum::MR_UNORM_8x16>,
    MRayDataType<MRayDataEnum::MR_UNORM_4x32>,
    MRayDataType<MRayDataEnum::MR_UNORM_2x64>,
    MRayDataType<MRayDataEnum::MR_SNORM_16x8>,
    MRayDataType<MRayDataEnum::MR_SNORM_8x16>,
    MRayDataType<MRayDataEnum::MR_SNORM_4x32>,
    MRayDataType<MRayDataEnum::MR_SNORM_2x64>,

    MRayDataType<MRayDataEnum::MR_STRING>

    // These are removed because of excessive recursion
    // of template instantiation
    //MRayDataType<MRayDataEnum::MR_UNORM_32x8>,
    //MRayDataType<MRayDataEnum::MR_UNORM_16x16>,
    //MRayDataType<MRayDataEnum::MR_UNORM_8x32>,
    //MRayDataType<MRayDataEnum::MR_UNORM_4x64>,
    //MRayDataType<MRayDataEnum::MR_SNORM_32x8>,
    //MRayDataType<MRayDataEnum::MR_SNORM_16x16>,
    //MRayDataType<MRayDataEnum::MR_SNORM_8x32>,
    //MRayDataType<MRayDataEnum::MR_SNORM_4x64>,
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
    constexpr MRayDataEnum  Name() const;
    constexpr size_t        Size() const;
    constexpr size_t        Alignment() const;

    using MRayDataTypeBase::MRayDataTypeBase;
};

struct MRayPixelTypeRT : public MRayPixelTypeBase
{
    using enum MRayPixelEnum;
    constexpr MRayPixelEnum     Name() const;
    constexpr size_t            ChannelCount() const;

    using MRayPixelTypeBase::MRayPixelTypeBase;
};

inline constexpr MRayDataEnum MRayDataTypeRT::Name() const
{
    return std::visit([](auto&& d) -> MRayDataEnum
    {
        return std::remove_cvref_t<decltype(d)>::Name;
    }, *this);
}

inline constexpr size_t MRayDataTypeRT::Size() const
{
    return std::visit([](auto&& d) -> size_t
    {
        return std::remove_cvref_t<decltype(d)>::Size;
    }, *this);
}

inline constexpr size_t MRayDataTypeRT::Alignment() const
{
    return std::visit([](auto&& d) -> size_t
    {
        return std::remove_cvref_t<decltype(d)>::Alignment;
    }, *this);
}

inline constexpr MRayPixelEnum MRayPixelTypeRT::Name() const
{
    return std::visit([](auto&& d) -> MRayPixelEnum
    {
        return std::remove_cvref_t<decltype(d)>::Name;
    }, *this);
}

inline constexpr size_t MRayPixelTypeRT::ChannelCount() const
{
    return std::visit([](auto&& d) -> size_t
    {
        return std::remove_cvref_t<decltype(d)>::ChannelCount;
    }, *this);
}
