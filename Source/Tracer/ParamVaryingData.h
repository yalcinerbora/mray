#pragma once

#include "Device/GPUSystem.h"
#include "TracerTypes.h"

// Meta Texture Type
template <uint32_t DIMS, class T>
class ParamVaryingData
{
    static_assert(DIMS == 1 || DIMS == 2 || DIMS == 3,
                  "Surface varying data at most have 3 dimensions");
    using Texture = TextureView<DIMS, T>;

    private:
    Variant<Texture, T>         t;

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

namespace SpectrumConverterDetail
{

template<class Converter>
struct RendererSpectrum
{
    private:
    Ref<const Converter> c;

    public:
    MRAY_HYBRID             RendererSpectrum(const Converter& c);
    MRAY_HYBRID Spectrum    operator()(const Spectrum& s) const;
};

template<class Converter, uint32_t DIMS>
class RendererParamVaryingSpectrum
{
    using PVD = ParamVaryingData<DIMS, Spectrum>;

    private:
    Ref<const Converter>        converter;
    Ref<const PVD>              input;

    public:
    //
    MRAY_HYBRID RendererParamVaryingSpectrum(const Converter& c, const PVD& p);

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

    template<uint32_t DIMS>
    using RendererParamVaryingData = SpectrumConverterDetail:: template RendererParamVaryingSpectrum<C, DIMS>;
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

template <uint32_t DIMS, class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> ParamVaryingData<DIMS, T>::operator()(Vector<DIMS, Float> uvCoords) const
{
    if(std::holds_alternative<Texture>(t))
        return t(uvCoords);
    return t;
}

template <uint32_t DIMS, class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> ParamVaryingData<DIMS, T>::operator()(Vector<DIMS, Float> uvCoords,
                                                  Vector<DIMS, Float> dpdu,
                                                  Vector<DIMS, Float> dpdv) const
{
    if(std::holds_alternative<Texture>(t))
        return t(uvCoords, dpdu, dpdv);
    return t;
}

template <uint32_t DIMS, class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> ParamVaryingData<DIMS, T>::operator()(Vector<DIMS, Float> uvCoords,
                                                  uint32_t mipLevel) const
{
    if(std::holds_alternative<Texture>(t))
        return t(uvCoords, mipLevel);
    return t;
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

template<class C, uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
RendererParamVaryingSpectrum<C, D>::RendererParamVaryingSpectrum(const C& c, const ParamVaryingData<D, Spectrum>& p)
    : converter(c)
    , input(p)
{}

template<class C, uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<Spectrum> RendererParamVaryingSpectrum<C, D>::operator()(Vector<D, Float> uvCoords) const
{
    return converter.get().Convert(input.get()(uvCoords));
}

template<class C, uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<Spectrum> RendererParamVaryingSpectrum<C, D>::operator()(Vector<D, Float> uvCoords,
                                                                  Vector<D, Float> dpdu,
                                                                  Vector<D, Float> dpdv) const
{
    return converter.get().Convert(input.get()(uvCoords, dpdu, dpdv));
}

template<class C, uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<Spectrum> RendererParamVaryingSpectrum<C, D>::operator()(Vector<D, Float> uvCoords,
                                                                  uint32_t mipLevel) const
{
    return converter.get().Convert(input.get()(uvCoords, mipLevel));
}

}

MRAY_HYBRID MRAY_CGPU_INLINE
Optional<Spectrum> SpectrumConverterIdentity::Convert(const Optional<Spectrum>& c) const
{
    return c;
}