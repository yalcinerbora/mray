#pragma once

#include "TextureView.h"
#include "StreamingTextureView.hpp" // IWYU pragma: keep

template<class T>
MRAY_GPU MRAY_GPU_INLINE
void PostprocessTexel(T& out, TextureReadMode mode, Vector4 in)
{
    auto ConvertType = [](T& out, Vector4 in)
    {
        if constexpr(std::is_same_v<T, Float>)
            out = in[0];
        else
            out = T(in);
    };

    if constexpr(std::is_same_v<T, Vector3>)
    {
        Vector2 t = Vector2(in);
        switch(mode)
        {
            using enum TextureReadMode;
            case TO_3C_BASIC_NORMAL_FROM_UNSIGNED_2C:
            {
                t = t * Vector2(2) - Vector2(1);
            }
            [[fallthrough]];
            case TO_3C_BASIC_NORMAL_FROM_SIGNED_2C:
            {
                Float z = Math::SqrtMax(Float(1) - t[0] * t[0] - t[1] * t[1]);
                out = Vector3(t[0], t[1], z);
                break;
            }
            // Octahedral mapping
            // From https://jcgt.org/published/0003/02/01/paper.pdf
            // Listing 6.
            case TO_3C_OCTA_NORMAL_FROM_UNSIGNED_2C:
            {
                t = t * Vector2(2) - Vector2(1);
            }
            [[fallthrough]];
            case TO_3C_OCTA_NORMAL_FROM_SIGNED_2C:
            {
                t = Vector2(t[0] + t[1], t[0] - t[1]) * Float(0.5);
                out = Vector3(t, Float(1) - t.Abs().Sum());
                break;
            }
            default: ConvertType(out, in);
        }
    }
    else ConvertType(out, in);
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
T TracerTexView<2, T>::operator()(UV uv) const
{
    T r = T(0);
    if(flipY) uv[1] = Float(1) - uv[1];
    switch(index)
    {
        case 0: PostprocessTexel(r, mode, Vector4(tF (uv), 0, 0, 0)); break;
        case 1: PostprocessTexel(r, mode, Vector4(tV2(uv), 0, 0)); break;
        case 2: PostprocessTexel(r, mode, Vector4(tV3(uv), 0)); break;
        case 3: PostprocessTexel(r, mode,         tV4(uv)); break;
    }
    // Do not ask why this is not in the switch statement above...
    // nvcc does not properly compiles it on release mode?
    // (I am not sure), image will become yellowish if you put this
    // case above
    if(index == 4) PostprocessTexel(r, mode, sT(uv));

    return r;
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
T TracerTexView<2, T>::operator()(UV uv, UV dpdx, UV dpdy) const
{
    T r = T(0);
    if(flipY) uv[1] = Float(1) - uv[1];
    switch(index)
    {
        case 0: PostprocessTexel(r, mode, Vector4(tF (uv, dpdx, dpdy), 0, 0, 0)); break;
        case 1: PostprocessTexel(r, mode, Vector4(tV2(uv, dpdx, dpdy), 0, 0)); break;
        case 2: PostprocessTexel(r, mode, Vector4(tV3(uv, dpdx, dpdy), 0)); break;
        case 3: PostprocessTexel(r, mode,         tV4(uv, dpdx, dpdy)); break;
    }
    // Do not ask why this is not in the switch statement above...
    // nvcc does not properly compiles it on release mode?
    // (I am not sure), image will become yellowish if you put this
    // case above
    if(index == 4) PostprocessTexel(r, mode, sT(uv, dpdx, dpdy));

    return r;
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
T TracerTexView<2, T>::operator()(UV uv, Float mipLevel) const
{
    T r = T(0);
    if(flipY) uv[1] = Float(1) - uv[1];
    switch(index)
    {
        case 0: PostprocessTexel(r, mode, Vector4(tF (uv, mipLevel), 0, 0, 0)); break;
        case 1: PostprocessTexel(r, mode, Vector4(tV2(uv, mipLevel), 0, 0)); break;
        case 2: PostprocessTexel(r, mode, Vector4(tV3(uv, mipLevel), 0)); break;
        case 3: PostprocessTexel(r, mode,         tV4(uv, mipLevel)); break;
    }
    // Do not ask why this is not in the switch statement above...
    // nvcc does not properly compiles it on release mode?
    // (I am not sure), image will become yellowish if you put this
    // case above
    if(index == 4) PostprocessTexel(r, mode, sT(uv, mipLevel));

    return r;
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
bool TracerTexView<2, T>::IsResident(UV, Float) const
{
    return false;
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
bool TracerTexView<2, T>::IsResident(UV, UV, UV) const
{
    // TODO:
    return false;
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
bool TracerTexView<2, T>::IsResident(UV) const
{
    // TODO:
    return false;
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
T TracerTexView<3, T>::operator()(UV uv) const
{
    T r = T(0);
    if(flipY) uv[1] = Float(1) - uv[1];
    switch(index)
    {
        case 0: PostprocessTexel(r, mode, Vector4(tF(uv), 0, 0, 0)); break;
        case 1: PostprocessTexel(r, mode, Vector4(tV2(uv), 0, 0)); break;
        case 2: PostprocessTexel(r, mode, Vector4(tV3(uv), 0)); break;
        case 3: PostprocessTexel(r, mode, tV4(uv)); break;
    }
    return r;
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
T TracerTexView<3, T>::operator()(UV uv,
                                  UV dpdx,
                                  UV dpdy) const
{
    T r = T(0);
    if(flipY) uv[1] = Float(1) - uv[1];
    switch(index)
    {
        case 0: PostprocessTexel(r, mode, Vector4(tF(uv, dpdx, dpdy), 0, 0, 0)); break;
        case 1: PostprocessTexel(r, mode, Vector4(tV2(uv, dpdx, dpdy), 0, 0)); break;
        case 2: PostprocessTexel(r, mode, Vector4(tV3(uv, dpdx, dpdy), 0)); break;
        case 3: PostprocessTexel(r, mode, tV4(uv, dpdx, dpdy)); break;
    }
    return r;
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
T TracerTexView<3, T>::operator()(UV uv, Float mipLevel) const
{
    T r = T(0);
    if(flipY) uv[1] = Float(1) - uv[1];
    switch(index)
    {
        case 0: PostprocessTexel(r, mode, Vector4(tF (uv, mipLevel), 0, 0, 0)); break;
        case 1: PostprocessTexel(r, mode, Vector4(tV2(uv, mipLevel), 0, 0)); break;
        case 2: PostprocessTexel(r, mode, Vector4(tV3(uv, mipLevel), 0)); break;
        case 3: PostprocessTexel(r, mode,         tV4(uv, mipLevel)); break;
    }
    return r;
}
