#pragma once

#include "TracerTypes.h"
#include "TextureView.h"

// Meta Texture Type
template <uint32_t DIMS, class T>
class ParamVaryingData;

template <class T>
class ParamVaryingData<2, T>
{
    using Texture = TracerTexView<2, T>;
    using UV      = Vector2;

    private:
    union
    {
        Texture tex;
        T       data;
    };
    bool isTexture;

    public:
    MR_HF_DECL  ParamVaryingData(const T&);
    MR_HF_DECL  ParamVaryingData(const Texture&);

    // Base Access
    MR_GF_DECL T   operator()(UV uvCoords) const;
    // Gradient Access
    MR_GF_DECL T   operator()(UV uvCoords,
                              UV dpdx,
                              UV dpdy) const;
    // Direct Mip Access
    MR_GF_DECL T   operator()(UV uvCoords, Float mipLevel) const;
    //
    MR_HF_DECL bool IsConstant() const;
    MR_HF_DECL bool IsResident(UV uvCoords) const;
    MR_HF_DECL bool IsResident(UV uvCoords,
                               UV dpdx,
                               UV dpdy) const;
    MR_HF_DECL bool IsResident(UV uvCoords, Float mipLevel) const;
};

template <class T>
class ParamVaryingData<3, T>
{
    using Texture = TracerTexView<3, T>;
    using UV      = Vector2;

    enum Type : uint8_t
    {
        SCALAR,
        DENSE_TEX,
        SPARSE_TEX
    };

    private:
    union
    {
        Texture     tex;
        T           data;
        const Byte* sparseData;
    };
    Type           type;
    MRayPixelEnum  sparseDataType;

    public:
    MR_HF_DECL  ParamVaryingData(const T&);
    MR_HF_DECL  ParamVaryingData(const Texture&);
    MR_HF_DECL  ParamVaryingData(const Byte*, MRayPixelEnum);

    // Sparse Access
    MR_GF_DECL T   operator()(uint32_t i) const;
    // Base Access
    MR_GF_DECL T   operator()(UV uvCoords) const;
    // Gradient Access
    MR_GF_DECL T   operator()(UV uvCoords,
                              UV dpdx,
                              UV dpdy) const;
    // Direct Mip Access
    MR_GF_DECL T   operator()(UV uvCoords, Float mipLevel) const;
    //
    MR_HF_DECL bool IsScalar() const;
    MR_HF_DECL bool IsDenseTex() const;
    MR_HF_DECL bool IsSparseTex() const;
};

// ============ //
//      2D      //
// ============ //
template <class T>
MR_HF_DEF
ParamVaryingData<2, T>::ParamVaryingData(const T& d)
    : data(d)
    , isTexture(false)
{}

template <class T>
MR_HF_DEF
ParamVaryingData<2, T>::ParamVaryingData(const Texture& t)
    : tex(t)
    , isTexture(true)
{}

template <class T>
MR_GF_DEF
T ParamVaryingData<2, T>::operator()(UV uvCoords) const
{
    if(isTexture) return tex(uvCoords);
    else          return data;
}

template <class T>
MR_GF_DEF
T ParamVaryingData<2, T>::operator()(UV uvCoords, UV dpdx, UV dpdy) const
{
    if(isTexture)   return tex(uvCoords, dpdx, dpdy);
    else            return data;
}

template <class T>
MR_GF_DEF
T ParamVaryingData<2, T>::operator()(UV uvCoords, Float mipLevel) const
{
    if(isTexture) return tex(uvCoords, mipLevel);
    else          return data;
}

template <class T>
MR_HF_DEF
bool ParamVaryingData<2, T>::IsConstant() const
{
    return !isTexture;
}

template <class T>
MR_HF_DEF
bool ParamVaryingData<2, T>::IsResident(UV uvCoords) const
{
    if(isTexture) return tex.IsResident(uvCoords);
    else          return true;
}

template <class T>
MR_HF_DEF
bool ParamVaryingData<2, T>::IsResident(UV uvCoords,
                                        UV dpdx,
                                        UV dpdy) const
{
    if(isTexture) return tex.IsResident(uvCoords, dpdx, dpdy);
    else          return true;
}

template <class T>
MR_HF_DEF
bool ParamVaryingData<2, T>::IsResident(UV uvCoords, Float mipLevel) const
{
    if(isTexture) return tex.IsResident(uvCoords, mipLevel);
    else          return true;
}

// ============ //
//      3D      //
// ============ //
template <class T>
MR_HF_DEF
ParamVaryingData<3, T>::ParamVaryingData(const T& d)
    : data(d)
    , type(SCALAR)
{}

template <class T>
MR_HF_DEF
ParamVaryingData<3, T>::ParamVaryingData(const Texture& t)
    : tex(t)
    , type(DENSE_TEX)
{}

template <class T>
MR_HF_DEF
ParamVaryingData<3, T>::ParamVaryingData(const Byte* ptr,
                                         MRayPixelEnum sType)
    : sparseData(ptr)
    , type(SPARSE_TEX)
    , sparseDataType(sType)
{}

template <class T>
MR_GF_DEF
T ParamVaryingData<3, T>::operator()(uint32_t i) const
{
    if(type == SPARSE_TEX)
    {
        if constexpr(std::is_same_v<Float, T>)
            return ReadGenericTexelData(sparseData, i, sparseDataType)[0];
        else
            return T(ReadGenericTexelData(sparseData, i, sparseDataType));
    }
    return T();
}

template <class T>
MR_GF_DEF
T ParamVaryingData<3, T>::operator()(UV uvCoords) const
{
    switch(type)
    {
        case SCALAR:    return data;
        case DENSE_TEX: return tex(uvCoords);
        default:        return T();
    }
}

template <class T>
MR_GF_DEF
T ParamVaryingData<3, T>::operator()(UV uvCoords, UV dpdx, UV dpdy) const
{
    switch(type)
    {
        case SCALAR:    return data;
        case DENSE_TEX: return tex(uvCoords, dpdx, dpdy);
        default:        return T();
    }
}

template <class T>
MR_GF_DEF
T ParamVaryingData<3, T>::operator()(UV uvCoords, Float mipLevel) const
{
    switch(type)
    {
        case SCALAR:    return data;
        case DENSE_TEX: return tex(uvCoords, mipLevel);
        default:        return T();
    }
}

template <class T>
MR_HF_DEF
bool ParamVaryingData<3, T>::IsScalar() const
{
    return type == SCALAR;
}

template <class T>
MR_HF_DEF
bool ParamVaryingData<3, T>::IsDenseTex() const
{
    return type == DENSE_TEX;
}

template <class T>
MR_HF_DEF
bool ParamVaryingData<3, T>::IsSparseTex() const
{
    return type == SPARSE_TEX;
}

