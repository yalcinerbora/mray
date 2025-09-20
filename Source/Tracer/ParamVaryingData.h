#pragma once

#include "TracerTypes.h"
#include "TextureView.h"

// Meta Texture Type
template <uint32_t DIMS, class T>
class ParamVaryingData
{
    static_assert(DIMS == 1 || DIMS == 2 || DIMS == 3,
                  "Surface varying data at most have 3 dimensions");
    using Texture = TracerTexView<DIMS, T>;
    using UV = Vector<DIMS, Float>;

    private:
    Variant<Texture, T> t;

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

template <uint32_t DIMS, class T>
MR_HF_DEF
ParamVaryingData<DIMS, T>::ParamVaryingData(const T& tt)
    : t(tt)
{}

template <uint32_t DIMS, class T>
MR_HF_DEF
ParamVaryingData<DIMS, T>::ParamVaryingData(const Texture& tt)
    : t(tt)
{}

template <uint32_t DIMS, class T>
MR_GF_DEF
T ParamVaryingData<DIMS, T>::operator()(Vector<DIMS, Float> uvCoords) const
{
    if(std::holds_alternative<Texture>(t))
        return std::get<Texture>(t)(uvCoords);
    return std::get<T>(t);
}

template <uint32_t DIMS, class T>
MR_GF_DEF
T ParamVaryingData<DIMS, T>::operator()(Vector<DIMS, Float> uvCoords,
                                        Vector<DIMS, Float> dpdx,
                                        Vector<DIMS, Float> dpdy) const
{
    if(std::holds_alternative<Texture>(t))
        return std::get<Texture>(t)(uvCoords, dpdx, dpdy);
    return std::get<T>(t);
}

template <uint32_t DIMS, class T>
MR_GF_DEF
T ParamVaryingData<DIMS, T>::operator()(Vector<DIMS, Float> uvCoords,
                                        Float mipLevel) const
{
    if(std::holds_alternative<Texture>(t))
        return std::get<Texture>(t)(uvCoords, mipLevel);
    return std::get<T>(t);
}

template <uint32_t DIMS, class T>
MR_HF_DEF
bool ParamVaryingData<DIMS, T>::IsConstant() const
{
    return (!std::holds_alternative<Texture>(t));
}

template <uint32_t DIMS, class T>
MR_HF_DEF
bool ParamVaryingData<DIMS, T>::IsResident(UV uvCoords) const
{
    if(IsConstant()) return true;
    return std::get<Texture>(t).IsResident(uvCoords);
}

template <uint32_t DIMS, class T>
MR_HF_DEF
bool ParamVaryingData<DIMS, T>::IsResident(UV uvCoords,
                                           UV dpdx,
                                           UV dpdy) const
{
    if(IsConstant()) return true;
    return std::get<Texture>(t).IsResident(uvCoords, dpdx, dpdy);
}

template <uint32_t DIMS, class T>
MR_HF_DEF
bool ParamVaryingData<DIMS, T>::IsResident(UV uvCoords, Float mipLevel) const
{
    if(IsConstant()) return true;
    return std::get<Texture>(t).IsResident(uvCoords, mipLevel);
}
