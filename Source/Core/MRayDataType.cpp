#include "MRayDataType.h"

// Runtime query of an data enum
// Good old switch case here, did not bother for metaprogramming stuff
Pair<size_t, size_t> FindSizeAndAlignment(MRayDataEnum e)
{
    auto SA = []<MRayDataEnum E>()
    {
        return Pair<size_t,size_t>(MRayDataType<E>::Size, MRayDataType<E>::Alignment);
    };
    using enum MRayDataEnum;
    switch(e)
    {
        case MR_CHAR:         return SA.operator()<MR_CHAR>();
        case MR_VECTOR_2C:    return SA.operator()<MR_VECTOR_2C>();
        case MR_VECTOR_3C:    return SA.operator()<MR_VECTOR_3C>();
        case MR_VECTOR_4C:    return SA.operator()<MR_VECTOR_4C>();
        case MR_SHORT:        return SA.operator()<MR_SHORT>();
        case MR_VECTOR_2S:    return SA.operator()<MR_VECTOR_2S>();
        case MR_VECTOR_3S:    return SA.operator()<MR_VECTOR_3S>();
        case MR_VECTOR_4S:    return SA.operator()<MR_VECTOR_4S>();
        case MR_INT:          return SA.operator()<MR_INT>();
        case MR_VECTOR_2I:    return SA.operator()<MR_VECTOR_2I>();
        case MR_VECTOR_3I:    return SA.operator()<MR_VECTOR_3I>();
        case MR_VECTOR_4I:    return SA.operator()<MR_VECTOR_4I>();
        case MR_UCHAR:        return SA.operator()<MR_UCHAR>();
        case MR_VECTOR_2UC:   return SA.operator()<MR_VECTOR_2UC>();
        case MR_VECTOR_3UC:   return SA.operator()<MR_VECTOR_3UC>();
        case MR_VECTOR_4UC:   return SA.operator()<MR_VECTOR_4UC>();
        case MR_USHORT:       return SA.operator()<MR_USHORT>();
        case MR_VECTOR_2US:   return SA.operator()<MR_VECTOR_2US>();
        case MR_VECTOR_3US:   return SA.operator()<MR_VECTOR_3US>();
        case MR_VECTOR_4US:   return SA.operator()<MR_VECTOR_4US>();
        case MR_UINT:         return SA.operator()<MR_UINT>();
        case MR_VECTOR_2UI:   return SA.operator()<MR_VECTOR_2UI>();
        case MR_VECTOR_3UI:   return SA.operator()<MR_VECTOR_3UI>();
        case MR_VECTOR_4UI:   return SA.operator()<MR_VECTOR_4UI>();
        case MR_FLOAT:        return SA.operator()<MR_FLOAT>();
        case MR_VECTOR_2F:    return SA.operator()<MR_VECTOR_2F>();
        case MR_VECTOR_3F:    return SA.operator()<MR_VECTOR_3F>();
        case MR_VECTOR_4F:    return SA.operator()<MR_VECTOR_4F>();
        case MR_DOUBLE:       return SA.operator()<MR_DOUBLE>();
        case MR_VECTOR_2D:    return SA.operator()<MR_VECTOR_2D>();
        case MR_VECTOR_3D:    return SA.operator()<MR_VECTOR_3D>();
        case MR_VECTOR_4D:    return SA.operator()<MR_VECTOR_4D>();
        case MR_DEFAULT_FLT:  return SA.operator()<MR_DEFAULT_FLT>();
        case MR_VECTOR_2:     return SA.operator()<MR_VECTOR_2>();
        case MR_VECTOR_3:     return SA.operator()<MR_VECTOR_3>();
        case MR_VECTOR_4:     return SA.operator()<MR_VECTOR_4>();
        case MR_QUATERNION:   return SA.operator()<MR_QUATERNION>();
        case MR_MATRIX_4x4:   return SA.operator()<MR_MATRIX_4x4>();
        case MR_MATRIX_3x3:   return SA.operator()<MR_MATRIX_3x3>();
        case MR_AABB3_ENUM:   return SA.operator()<MR_AABB3_ENUM>();
        case MR_RAY:          return SA.operator()<MR_RAY>();
        case MR_UNORM_4x8:    return SA.operator()<MR_UNORM_4x8>();
        case MR_UNORM_2x16:   return SA.operator()<MR_UNORM_2x16>();
        case MR_SNORM_4x8:    return SA.operator()<MR_SNORM_4x8>();
        case MR_SNORM_2x16:   return SA.operator()<MR_SNORM_2x16>();
        case MR_UNORM_8x8:    return SA.operator()<MR_UNORM_8x8>();
        case MR_UNORM_4x16:   return SA.operator()<MR_UNORM_4x16>();
        case MR_UNORM_2x32:   return SA.operator()<MR_UNORM_2x32>();
        case MR_SNORM_8x8:    return SA.operator()<MR_SNORM_8x8>();
        case MR_SNORM_4x16:   return SA.operator()<MR_SNORM_4x16>();
        case MR_SNORM_2x32:   return SA.operator()<MR_SNORM_2x32>();
        case MR_UNORM_16x8:   return SA.operator()<MR_UNORM_16x8>();
        case MR_UNORM_8x16:   return SA.operator()<MR_UNORM_8x16>();
        case MR_UNORM_4x32:   return SA.operator()<MR_UNORM_4x32>();
        case MR_UNORM_2x64:   return SA.operator()<MR_UNORM_2x64>();
        case MR_SNORM_16x8:   return SA.operator()<MR_SNORM_16x8>();
        case MR_SNORM_8x16:   return SA.operator()<MR_SNORM_8x16>();
        case MR_SNORM_4x32:   return SA.operator()<MR_SNORM_4x32>();
        case MR_SNORM_2x64:   return SA.operator()<MR_SNORM_2x64>();
        case MR_UNORM_32x8:   return SA.operator()<MR_UNORM_32x8>();
        case MR_UNORM_16x16:  return SA.operator()<MR_UNORM_16x16>();
        case MR_UNORM_8x32:   return SA.operator()<MR_UNORM_8x32>();
        case MR_UNORM_4x64:   return SA.operator()<MR_UNORM_4x64>();
        case MR_SNORM_32x8:   return SA.operator()<MR_SNORM_32x8>();
        case MR_SNORM_16x16:  return SA.operator()<MR_SNORM_16x16>();
        case MR_SNORM_8x32:   return SA.operator()<MR_SNORM_8x32>();
        case MR_SNORM_4x64:   return SA.operator()<MR_SNORM_4x64>();
        default: throw MRayError("Unkown MRayDataType!");
    }
}