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
    template<MRayDataEnum E>
    constexpr auto DataEnumToType()
    {
        using namespace TypeFinder;
        using enum MRayDataEnum;
        constexpr auto MRayDataEnumTypeMapping = std::make_tuple
        (
            ValKeyTypePair<MR_CHAR,         int8_t>{},
            ValKeyTypePair<MR_VECTOR_2C,    Vector2c>{},
            ValKeyTypePair<MR_VECTOR_3C,    Vector3c>{},
            ValKeyTypePair<MR_VECTOR_4C,    Vector4c>{},

            ValKeyTypePair<MR_SHORT,        int16_t>{},
            ValKeyTypePair<MR_VECTOR_2S,    Vector2s>{},
            ValKeyTypePair<MR_VECTOR_3S,    Vector3s>{},
            ValKeyTypePair<MR_VECTOR_4S,    Vector4s>{},

            ValKeyTypePair<MR_INT,          int32_t>{},
            ValKeyTypePair<MR_VECTOR_2I,    Vector2i>{},
            ValKeyTypePair<MR_VECTOR_3I,    Vector3i>{},
            ValKeyTypePair<MR_VECTOR_4I,    Vector4i>{},

            ValKeyTypePair<MR_UCHAR,        uint8_t>{},
            ValKeyTypePair<MR_VECTOR_2UC,   Vector2uc>{},
            ValKeyTypePair<MR_VECTOR_3UC,   Vector3uc>{},
            ValKeyTypePair<MR_VECTOR_4UC,   Vector4uc>{},

            ValKeyTypePair<MR_USHORT,       uint16_t>{},
            ValKeyTypePair<MR_VECTOR_2US,   Vector2us>{},
            ValKeyTypePair<MR_VECTOR_3US,   Vector3us>{},
            ValKeyTypePair<MR_VECTOR_4US,   Vector4us>{},

            ValKeyTypePair<MR_UINT,         uint32_t>{},
            ValKeyTypePair<MR_VECTOR_2UI,   Vector2ui>{},
            ValKeyTypePair<MR_VECTOR_3UI,   Vector3ui>{},
            ValKeyTypePair<MR_VECTOR_4UI,   Vector4ui>{},

            ValKeyTypePair<MR_FLOAT,        float>{},
            ValKeyTypePair<MR_VECTOR_2F,    Vector2f>{},
            ValKeyTypePair<MR_VECTOR_3F,    Vector3f>{},
            ValKeyTypePair<MR_VECTOR_4F,    Vector4f>{},

            ValKeyTypePair<MR_DOUBLE,       double>{},
            ValKeyTypePair<MR_VECTOR_2D,    Vector2d>{},
            ValKeyTypePair<MR_VECTOR_3D,    Vector3d>{},
            ValKeyTypePair<MR_VECTOR_4D,    Vector4d>{},

            ValKeyTypePair<MR_DEFAULT_FLT,  Float>{},
            ValKeyTypePair<MR_VECTOR_2,     Vector2>{},
            ValKeyTypePair<MR_VECTOR_3,     Vector3>{},
            ValKeyTypePair<MR_VECTOR_4,     Vector4>{},

            ValKeyTypePair<MR_QUATERNION,   Quaternion>{},
            ValKeyTypePair<MR_MATRIX_4x4,   Matrix4x4>{},
            ValKeyTypePair<MR_MATRIX_3x3,   Matrix2x2>{},
            ValKeyTypePair<MR_AABB3_ENUM,   AABB3>{},
            ValKeyTypePair<MR_RAY,          Ray>{},

            ValKeyTypePair<MR_UNORM_4x8,    UNorm4x8>{},
            ValKeyTypePair<MR_UNORM_2x16,   UNorm2x16>{},
            ValKeyTypePair<MR_SNORM_4x8,    SNorm4x8>{},
            ValKeyTypePair<MR_SNORM_2x16,   SNorm2x16>{},

            ValKeyTypePair<MR_UNORM_8x8,    UNorm8x8>{},
            ValKeyTypePair<MR_UNORM_4x16,   UNorm4x16>{},
            ValKeyTypePair<MR_UNORM_2x32,   UNorm2x32>{},
            ValKeyTypePair<MR_SNORM_8x8,    SNorm8x8>{},
            ValKeyTypePair<MR_SNORM_4x16,   SNorm4x16>{},
            ValKeyTypePair<MR_SNORM_2x32,   SNorm2x32>{},

            ValKeyTypePair<MR_UNORM_16x8,   UNorm16x8>{},
            ValKeyTypePair<MR_UNORM_8x16,   UNorm8x16>{},
            ValKeyTypePair<MR_UNORM_4x32,   UNorm4x32>{},
            ValKeyTypePair<MR_UNORM_2x64,   UNorm2x64>{},
            ValKeyTypePair<MR_SNORM_16x8,   SNorm16x8>{},
            ValKeyTypePair<MR_SNORM_8x16,   SNorm8x16>{},
            ValKeyTypePair<MR_SNORM_4x32,   SNorm4x32>{},
            ValKeyTypePair<MR_SNORM_2x64,   SNorm2x64>{},

            ValKeyTypePair<MR_UNORM_32x8,   UNorm32x8>{},
            ValKeyTypePair<MR_UNORM_16x16,  UNorm16x16>{},
            ValKeyTypePair<MR_UNORM_8x32,   UNorm8x32>{},
            ValKeyTypePair<MR_UNORM_4x64,   UNorm4x64>{},
            ValKeyTypePair<MR_SNORM_32x8,   SNorm32x8>{},
            ValKeyTypePair<MR_SNORM_16x16,  SNorm16x16>{},
            ValKeyTypePair<MR_SNORM_8x32,   SNorm8x32>{},
            ValKeyTypePair<MR_SNORM_4x64,   SNorm4x64>{}
        );
        constexpr auto EList = MRayDataEnumTypeMapping;
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

struct MRayDataTypeRT
{
    const size_t size;
    const size_t alignment;
    const MRayDataEnum name;

    template<MRayDataEnum E>
    MRayDataTypeRT(MRayDataType<E>);
};

// Runtime query of an data enum
// Good old switch case here, did not bother for metaprogramming stuff
static std::pair<size_t, size_t> FindSizeAndAlignment(MRayDataEnum e)
{
    auto GetSizeAlign = []<MRayDataEnum E>()
    {
        return std::make_pair(MRayDataType<E>::Size, MRayDataType<E>::Alignment);
    };

    using enum MRayDataEnum;
    switch(e)
    {
        case MR_CHAR:         return GetSizeAlign.operator()<MR_CHAR>();
        case MR_VECTOR_2C:    return GetSizeAlign.operator()<MR_VECTOR_2C>();
        case MR_VECTOR_3C:    return GetSizeAlign.operator()<MR_VECTOR_3C>();
        case MR_VECTOR_4C:    return GetSizeAlign.operator()<MR_VECTOR_4C>();
        case MR_SHORT:        return GetSizeAlign.operator()<MR_SHORT>();
        case MR_VECTOR_2S:    return GetSizeAlign.operator()<MR_VECTOR_2S>();
        case MR_VECTOR_3S:    return GetSizeAlign.operator()<MR_VECTOR_3S>();
        case MR_VECTOR_4S:    return GetSizeAlign.operator()<MR_VECTOR_4S>();
        case MR_INT:          return GetSizeAlign.operator()<MR_INT>();
        case MR_VECTOR_2I:    return GetSizeAlign.operator()<MR_VECTOR_2I>();
        case MR_VECTOR_3I:    return GetSizeAlign.operator()<MR_VECTOR_3I>();
        case MR_VECTOR_4I:    return GetSizeAlign.operator()<MR_VECTOR_4I>();
        case MR_UCHAR:        return GetSizeAlign.operator()<MR_UCHAR>();
        case MR_VECTOR_2UC:   return GetSizeAlign.operator()<MR_VECTOR_2UC>();
        case MR_VECTOR_3UC:   return GetSizeAlign.operator()<MR_VECTOR_3UC>();
        case MR_VECTOR_4UC:   return GetSizeAlign.operator()<MR_VECTOR_4UC>();
        case MR_USHORT:       return GetSizeAlign.operator()<MR_USHORT>();
        case MR_VECTOR_2US:   return GetSizeAlign.operator()<MR_VECTOR_2US>();
        case MR_VECTOR_3US:   return GetSizeAlign.operator()<MR_VECTOR_3US>();
        case MR_VECTOR_4US:   return GetSizeAlign.operator()<MR_VECTOR_4US>();
        case MR_UINT:         return GetSizeAlign.operator()<MR_UINT>();
        case MR_VECTOR_2UI:   return GetSizeAlign.operator()<MR_VECTOR_2UI>();
        case MR_VECTOR_3UI:   return GetSizeAlign.operator()<MR_VECTOR_3UI>();
        case MR_VECTOR_4UI:   return GetSizeAlign.operator()<MR_VECTOR_4UI>();
        case MR_FLOAT:        return GetSizeAlign.operator()<MR_FLOAT>();
        case MR_VECTOR_2F:    return GetSizeAlign.operator()<MR_VECTOR_2F>();
        case MR_VECTOR_3F:    return GetSizeAlign.operator()<MR_VECTOR_3F>();
        case MR_VECTOR_4F:    return GetSizeAlign.operator()<MR_VECTOR_4F>();
        case MR_DOUBLE:       return GetSizeAlign.operator()<MR_DOUBLE>();
        case MR_VECTOR_2D:    return GetSizeAlign.operator()<MR_VECTOR_2D>();
        case MR_VECTOR_3D:    return GetSizeAlign.operator()<MR_VECTOR_3D>();
        case MR_VECTOR_4D:    return GetSizeAlign.operator()<MR_VECTOR_4D>();
        case MR_DEFAULT_FLT:  return GetSizeAlign.operator()<MR_DEFAULT_FLT>();
        case MR_VECTOR_2:     return GetSizeAlign.operator()<MR_VECTOR_2>();
        case MR_VECTOR_3:     return GetSizeAlign.operator()<MR_VECTOR_3>();
        case MR_VECTOR_4:     return GetSizeAlign.operator()<MR_VECTOR_4>();
        case MR_QUATERNION:   return GetSizeAlign.operator()<MR_QUATERNION>();
        case MR_MATRIX_4x4:   return GetSizeAlign.operator()<MR_MATRIX_4x4>();
        case MR_MATRIX_3x3:   return GetSizeAlign.operator()<MR_MATRIX_3x3>();
        case MR_AABB3_ENUM:   return GetSizeAlign.operator()<MR_AABB3_ENUM>();
        case MR_RAY:          return GetSizeAlign.operator()<MR_RAY>();
        case MR_UNORM_4x8:    return GetSizeAlign.operator()<MR_UNORM_4x8>();
        case MR_UNORM_2x16:   return GetSizeAlign.operator()<MR_UNORM_2x16>();
        case MR_SNORM_4x8:    return GetSizeAlign.operator()<MR_SNORM_4x8>();
        case MR_SNORM_2x16:   return GetSizeAlign.operator()<MR_SNORM_2x16>();
        case MR_UNORM_8x8:    return GetSizeAlign.operator()<MR_UNORM_8x8>();
        case MR_UNORM_4x16:   return GetSizeAlign.operator()<MR_UNORM_4x16>();
        case MR_UNORM_2x32:   return GetSizeAlign.operator()<MR_UNORM_2x32>();
        case MR_SNORM_8x8:    return GetSizeAlign.operator()<MR_SNORM_8x8>();
        case MR_SNORM_4x16:   return GetSizeAlign.operator()<MR_SNORM_4x16>();
        case MR_SNORM_2x32:   return GetSizeAlign.operator()<MR_SNORM_2x32>();
        case MR_UNORM_16x8:   return GetSizeAlign.operator()<MR_UNORM_16x8>();
        case MR_UNORM_8x16:   return GetSizeAlign.operator()<MR_UNORM_8x16>();
        case MR_UNORM_4x32:   return GetSizeAlign.operator()<MR_UNORM_4x32>();
        case MR_UNORM_2x64:   return GetSizeAlign.operator()<MR_UNORM_2x64>();
        case MR_SNORM_16x8:   return GetSizeAlign.operator()<MR_SNORM_16x8>();
        case MR_SNORM_8x16:   return GetSizeAlign.operator()<MR_SNORM_8x16>();
        case MR_SNORM_4x32:   return GetSizeAlign.operator()<MR_SNORM_4x32>();
        case MR_SNORM_2x64:   return GetSizeAlign.operator()<MR_SNORM_2x64>();
        case MR_UNORM_32x8:   return GetSizeAlign.operator()<MR_UNORM_32x8>();
        case MR_UNORM_16x16:  return GetSizeAlign.operator()<MR_UNORM_16x16>();
        case MR_UNORM_8x32:   return GetSizeAlign.operator()<MR_UNORM_8x32>();
        case MR_UNORM_4x64:   return GetSizeAlign.operator()<MR_UNORM_4x64>();
        case MR_SNORM_32x8:   return GetSizeAlign.operator()<MR_SNORM_32x8>();
        case MR_SNORM_16x16:  return GetSizeAlign.operator()<MR_SNORM_16x16>();
        case MR_SNORM_8x32:   return GetSizeAlign.operator()<MR_SNORM_8x32>();
        case MR_SNORM_4x64:   return GetSizeAlign.operator()<MR_SNORM_4x64>();
        default: throw MRayError("Unkown MRayDataType!");
    }
}

template<MRayDataEnum E>
inline MRayDataTypeRT::MRayDataTypeRT(MRayDataType<E>)
    : size(MRayDataType<E>::Size)
    , alignment(MRayDataType<E>::Alignment)
    , name(E)
{}