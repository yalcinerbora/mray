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
        case MR_INT8:         return SA.operator()<MR_INT8>();
        case MR_VECTOR_2C:    return SA.operator()<MR_VECTOR_2C>();
        case MR_VECTOR_3C:    return SA.operator()<MR_VECTOR_3C>();
        case MR_VECTOR_4C:    return SA.operator()<MR_VECTOR_4C>();
        case MR_INT16:        return SA.operator()<MR_INT16>();
        case MR_VECTOR_2S:    return SA.operator()<MR_VECTOR_2S>();
        case MR_VECTOR_3S:    return SA.operator()<MR_VECTOR_3S>();
        case MR_VECTOR_4S:    return SA.operator()<MR_VECTOR_4S>();
        case MR_INT32:        return SA.operator()<MR_INT32>();
        case MR_VECTOR_2I:    return SA.operator()<MR_VECTOR_2I>();
        case MR_VECTOR_3I:    return SA.operator()<MR_VECTOR_3I>();
        case MR_VECTOR_4I:    return SA.operator()<MR_VECTOR_4I>();
        case MR_INT64:        return SA.operator()<MR_INT64>();
        case MR_VECTOR_2L:    return SA.operator()<MR_VECTOR_2L>();
        case MR_VECTOR_3L:    return SA.operator()<MR_VECTOR_3L>();
        case MR_VECTOR_4L:    return SA.operator()<MR_VECTOR_4L>();

        case MR_UINT8:        return SA.operator()<MR_UINT8>();
        case MR_VECTOR_2UC:   return SA.operator()<MR_VECTOR_2UC>();
        case MR_VECTOR_3UC:   return SA.operator()<MR_VECTOR_3UC>();
        case MR_VECTOR_4UC:   return SA.operator()<MR_VECTOR_4UC>();
        case MR_UINT16:       return SA.operator()<MR_UINT16>();
        case MR_VECTOR_2US:   return SA.operator()<MR_VECTOR_2US>();
        case MR_VECTOR_3US:   return SA.operator()<MR_VECTOR_3US>();
        case MR_VECTOR_4US:   return SA.operator()<MR_VECTOR_4US>();
        case MR_UINT32:       return SA.operator()<MR_UINT32>();
        case MR_VECTOR_2UI:   return SA.operator()<MR_VECTOR_2UI>();
        case MR_VECTOR_3UI:   return SA.operator()<MR_VECTOR_3UI>();
        case MR_VECTOR_4UI:   return SA.operator()<MR_VECTOR_4UI>();
        case MR_UINT64:       return SA.operator()<MR_INT64>();
        case MR_VECTOR_2UL:   return SA.operator()<MR_VECTOR_2L>();
        case MR_VECTOR_3UL:   return SA.operator()<MR_VECTOR_3L>();
        case MR_VECTOR_4UL:   return SA.operator()<MR_VECTOR_4L>();

        case MR_FLOAT:        return SA.operator()<MR_FLOAT>();
        case MR_VECTOR_2:     return SA.operator()<MR_VECTOR_2>();
        case MR_VECTOR_3:     return SA.operator()<MR_VECTOR_3>();
        case MR_VECTOR_4:     return SA.operator()<MR_VECTOR_4>();

        case MR_QUATERNION:   return SA.operator()<MR_QUATERNION>();
        case MR_MATRIX_4x4:   return SA.operator()<MR_MATRIX_4x4>();
        case MR_MATRIX_3x3:   return SA.operator()<MR_MATRIX_3x3>();
        case MR_AABB3:        return SA.operator()<MR_AABB3>();
        case MR_RAY:          return SA.operator()<MR_RAY>();
        case MR_UNORM_4x8:    return SA.operator()<MR_UNORM_4x8>();
        case MR_UNORM_2x16:   return SA.operator()<MR_UNORM_2x16>();
        case MR_SNORM_4x8:    return SA.operator()<MR_SNORM_4x8>();
        case MR_SNORM_2x16:   return SA.operator()<MR_SNORM_2x16>();
        case MR_BOOL:         return SA.operator()<MR_BOOL>();
        default: throw MRayError("Unkown MRayDataType!");
    }
}