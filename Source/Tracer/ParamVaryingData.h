#pragma once

#include "Device/GPUSystem.h"
#include "TracerTypes.h"
#include "TextureView.h"

// Meta Texture Type
template <uint32_t DIMS, class T>
class ParamVaryingData
{
    static_assert(DIMS == 1 || DIMS == 2 || DIMS == 3,
                  "Surface varying data at most have 3 dimensions");
    using Texture = TracerTexView<DIMS, T>;

    private:
    Variant<Texture, T>         t;

    public:
    MRAY_HYBRID                 ParamVaryingData(const T&);
    MRAY_HYBRID                 ParamVaryingData(const Texture&);

    // Base Access
    MRAY_HYBRID Optional<T>     operator()(Vector<DIMS, Float> uvCoords) const;
    // Gradient Access
    MRAY_HYBRID Optional<T>     operator()(Vector<DIMS, Float> uvCoords,
                                           Vector<DIMS, Float> dpdu,
                                           Vector<DIMS, Float> dpdv) const;
    // Direct Mip Access
    MRAY_HYBRID Optional<T>     operator()(Vector<DIMS, Float> uvCoords,
                                           uint32_t mipLevel) const;
    MRAY_HYBRID bool            IsConstant() const;
};

template <class C>
concept SpectrumConverterC = requires(const C c)
{
    {c.Convert(Optional<Vector3>{})} -> std::same_as<Optional<Spectrum>>;
    {c.Convert(Spectrum{})} -> std::same_as<Spectrum>;
    {c.Wavelengths()} -> std::same_as<SpectrumWaves>;
    // TODO: Why this does not work?
    // ====================
    //{C::IsRGB} -> std::same_as<const bool>;
    // ====================
    C::IsRGB;
    requires std::is_same_v<decltype(C::IsRGB), const bool>;
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
    MRAY_HYBRID Spectrum    operator()(const Vector3& s) const;
};

template<class Converter, uint32_t DIMS>
class RendererParamVaryingSpectrum
{
    using PVD = ParamVaryingData<DIMS, Vector3>;

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
    MRAY_HYBRID bool                IsConstant() const;
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
    MRAY_HYBRID Optional<Spectrum>  Convert(const Optional<Vector3>&) const;
    MRAY_HYBRID Spectrum            Convert(const Spectrum&) const;
    // Adding this for IoR conversion (dispersion)
    // We could've added ConvertIoR function, but the input arguments
    // may change depending on the medium (unless we dictate a protocol)
    // This is little bit more scalable but for Identity converter this does not
    // makes sense. Medium logic should be careful to incorporate these
    MRAY_HYBRID SpectrumWaves Wavelengths() const;
    // By design the identity converter is RGB
    // (MRay holds every texture, value etc. as RGB)
    static constexpr bool IsRGB = true;

};

using SpectrumConverterContextIdentity = SpectrumConverterContext<SpectrumConverterIdentity>;
static_assert(SpectrumConverterC<SpectrumConverterIdentity>,
              "\"SpectrumConverterIdentity\" do not satistfy \"SpectrumConverterC\" concept.");
static_assert(SpectrumConverterContextC<SpectrumConverterContextIdentity>,
              "\"SpectrumConverterContextIdentity\" do not satistfy \"SpectrumConverterContextC\" concept." );

template <uint32_t DIMS, class T>
MRAY_HYBRID MRAY_CGPU_INLINE
ParamVaryingData<DIMS, T>::ParamVaryingData(const T& tt)
    : t(tt)
{}

template <uint32_t DIMS, class T>
MRAY_HYBRID MRAY_CGPU_INLINE
ParamVaryingData<DIMS, T>::ParamVaryingData(const Texture& tt)
    : t(tt)
{}

template <uint32_t DIMS, class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> ParamVaryingData<DIMS, T>::operator()(Vector<DIMS, Float> uvCoords) const
{
    if(std::holds_alternative<Texture>(t))
        return std::get<Texture>(t)(uvCoords);
    return std::get<T>(t);
}

template <uint32_t DIMS, class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> ParamVaryingData<DIMS, T>::operator()(Vector<DIMS, Float> uvCoords,
                                                  Vector<DIMS, Float> dpdu,
                                                  Vector<DIMS, Float> dpdv) const
{
    if(std::holds_alternative<Texture>(t))
        return std::get<Texture>(t)(uvCoords, dpdu, dpdv);
    return std::get<T>(t);
}

template <uint32_t DIMS, class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<T> ParamVaryingData<DIMS, T>::operator()(Vector<DIMS, Float> uvCoords,
                                                  uint32_t mipLevel) const
{
    if(std::holds_alternative<Texture>(t))
        return std::get<Texture>(t)(uvCoords, mipLevel);
    return std::get<T>(t);
}

template <uint32_t DIMS, class T>
MRAY_HYBRID MRAY_CGPU_INLINE
bool ParamVaryingData<DIMS, T>::IsConstant() const
{
    return (!std::holds_alternative<Texture>(t));
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
Spectrum RendererSpectrum<C>::operator()(const Vector3& s) const
{
    return c.Convert(s);
}

template<class C, uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
RendererParamVaryingSpectrum<C, D>::RendererParamVaryingSpectrum(const C& c, const ParamVaryingData<D, Vector3>& p)
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

template<class C, uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
bool RendererParamVaryingSpectrum<C, D>::IsConstant() const
{
    return input.get().IsConstant();
}

}

MRAY_HYBRID MRAY_CGPU_INLINE
Optional<Spectrum> SpectrumConverterIdentity::Convert(const Optional<Vector3>& c) const
{
    if(!c.has_value()) return std::nullopt;
    return Spectrum(c.value(), Float(0));
}
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum SpectrumConverterIdentity::Convert(const Spectrum& c) const
{
    return c;
}

MRAY_HYBRID MRAY_CGPU_INLINE
SpectrumWaves SpectrumConverterIdentity::Wavelengths() const
{
    return SpectrumWaves(VisibleSpectrumMiddle);
}