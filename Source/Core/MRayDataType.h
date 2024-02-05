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

template<MRayDataEnum E>
struct MRayDataType
{
    using Type = typename decltype(MRayDataDetail::DataEnumToType<E>())::Result;
    static constexpr size_t Size       = sizeof(Type);
    static constexpr size_t Alignment  = alignof(Type);
    static constexpr MRayDataEnum Name = E;
};


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

struct MRayDataTypeRT : public MRayDataTypeBase
{
    constexpr MRayDataEnum Name() const;
    constexpr size_t Size() const;
    constexpr size_t Alignment() const;

    using MRayDataTypeBase::MRayDataTypeBase;
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