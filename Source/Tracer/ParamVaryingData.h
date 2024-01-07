#pragma once

#include "Device/GPUSystem.h"

//using DefaultTextureAccessor = TextureAccessorCUDA;
//
//
//using DefaultTextureAccessor = TextureAccessorHost;

// Meta Texture Type
template <uint32_t DIMS, class T, class TextureAccessor = DefaultTextureAccessor>
class ParamVaryingData
{
    static_assert(DIMS == 1 || DIMS == 2 || DIMS == 3,
                  "Surface varying data at most have 3 dimensions");
    using Texture = TextureAccessor::template TextureType<DIMS, T>;

    private:
    bool                        isTex;
    Texture                     t;
    T                           baseData;

    public:
    // Base Access
    MRAY_HYBRID Optional<T>     operator()(Vector<DIMS, Float> uvCoords) const;
    // Gradient Access
    MRAY_HYBRID Optional<T>     operator()(Vector<DIMS, Float> uvCoords,
                                           Vector<DIMS, Float> dpdu,
                                           Vector<DIMS, Float> dpdv) const;
    // Direct Mip Access
    MRAY_HYBRID Optional<T>     operator()(Vector<DIMS, Float> uvCoords,
                                           uint32_t mipLevel) const;
};

template <class C>
concept SpectrumConverterC = requires(C c)
{
    {c.Convert(Optional<Spectrum>{})} -> std::same_as<Optional<Spectrum>>;
};

template <class C>
concept SpectrumConverterContextC = requires()
{
    typename C::Converter;
    requires SpectrumConverterC<typename C::Converter>;
};

// Most barebone spectrum conversion (delegation)
namespace SpectrumConverterDetail
{

template<class Converter>
struct RendererSpectrum
{
    private:
    const Converter& c;

    public:
    MRAY_HYBRID             RendererSpectrum(const Converter& c);
    MRAY_HYBRID Spectrum    operator()(const Spectrum& s) const;
};

template<class Converter, uint32_t DIMS, class TextureAccessor = DefaultTextureAccessor>
class RendererParamVaryingData
{
    private:
    const Converter&                            converter;
    const ParamVaryingData<DIMS, Spectrum,
                           TextureAccessor>&    input;

    public:
    //
    MRAY_HYBRID     RendererParamVaryingData(const Converter& c,
                                             const ParamVaryingData<DIMS, Spectrum,
                                                                    TextureAccessor>& p);

    MRAY_HYBRID Optional<Spectrum>  operator()(Vector<DIMS, Float> uvCoords) const;
    // Gradient Access
    MRAY_HYBRID Optional<Spectrum>  operator()(Vector<DIMS, Float> uvCoords,
                                               Vector<DIMS, Float> dpdu,
                                               Vector<DIMS, Float> dpdv) const;
    // Direct Mip Access
    MRAY_HYBRID Optional<Spectrum>  operator()(Vector<DIMS, Float> uvCoords,
                                               uint32_t mipLevel) const;
};

}

template <SpectrumConverterC C>
struct SpectrumConverterContext
{
    using Converter = C;

    template<uint32_t DIMS, class TA = DefaultTextureAccessor>
    using RendererParamVaryingData = SpectrumConverterDetail:: template RendererParamVaryingData<C, DIMS, TA>;
    using RendererSpectrum = SpectrumConverterDetail:: template RendererSpectrum<C>;
};

// Concerete Identity Spectrum Converter
struct SpectrumConverterIdentity
{
    MRAY_HYBRID Optional<Spectrum> Convert(const Optional<Spectrum>& c) const;
};

using SpectrumConverterContextIdentity = SpectrumConverterContext<SpectrumConverterIdentity>;
static_assert(SpectrumConverterC<SpectrumConverterIdentity>,
              "\"SpectrumConverterIdentity\" do not satistfy \"SpectrumConverterC\" concept.");
static_assert(SpectrumConverterContextC<SpectrumConverterContextIdentity>,
              "\"SpectrumConverterContextIdentity\" do not satistfy \"SpectrumConverterContextC\" concept." );

template <uint32_t DIMS, class T, class TA>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> ParamVaryingData<DIMS, T, TA>::operator()(Vector<DIMS, Float> uvCoords) const
{
    if(isTex)
        return t(uvCoords);
    return baseData;
}

template <uint32_t DIMS, class T, class TA>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> ParamVaryingData<DIMS, T, TA>::operator()(Vector<DIMS, Float> uvCoords,
                                                      Vector<DIMS, Float> dpdu,
                                                      Vector<DIMS, Float> dpdv) const
{
    if(isTex)
        return t(uvCoords, dpdu, dpdv);
    return baseData;
}

template <uint32_t DIMS, class T, class TA>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> ParamVaryingData<DIMS, T, TA>::operator()(Vector<DIMS, Float> uvCoords,
                                                      uint32_t mipLevel) const
{
    if(isTex)
        return t(uvCoords, mipLevel);
    return baseData;
}

namespace SpectrumConverterDetail
{

template<class C>
MRAY_HYBRID MRAY_CGPU_INLINE
RendererSpectrum<C>::RendererSpectrum(const C& c)
    : c(c)
{}

template<class C>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum RendererSpectrum<C>::operator()(const Spectrum& s) const
{
    return c.Convert(s);
}

template<class Converter, uint32_t D, class TA>
MRAY_HYBRID MRAY_CGPU_INLINE
RendererParamVaryingData<Converter, D, TA>::RendererParamVaryingData(const Converter& c,
                                                                     const ParamVaryingData<D, Spectrum, TA>& p)
    : converter(c)
    , input(p)
{}

template<class C, uint32_t D, class TA>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<Spectrum> RendererParamVaryingData<C, D, TA>::operator()(Vector<D, Float> uvCoords) const
{
    return converter(input(uvCoords));
}

template<class C, uint32_t D, class TA>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<Spectrum> RendererParamVaryingData<C, D, TA>::operator()(Vector<D, Float> uvCoords,
                                                                  Vector<D, Float> dpdu,
                                                                  Vector<D, Float> dpdv) const
{
    return converter.Convert(input(uvCoords, dpdu, dpdv));
}

template<class C, uint32_t D, class TA>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<Spectrum> RendererParamVaryingData<C, D, TA>::operator()(Vector<D, Float> uvCoords,
                                                                  uint32_t mipLevel) const
{
    return converter.Convert(input(uvCoords, mipLevel));
}

}

MRAY_HYBRID MRAY_CGPU_INLINE
Optional<Spectrum> SpectrumConverterIdentity::Convert(const Optional<Spectrum>& c) const
{
    return c;
}