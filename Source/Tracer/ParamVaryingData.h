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
    std::variant<Texture, T> t;

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

template <class C>
concept SpectrumConverterC = requires(const C c)
{
    {c.Convert(Vector3{})} -> std::same_as<Spectrum>;
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
    MR_HF_DECL          RendererSpectrum(const Converter& c);
    MR_HF_DECL Spectrum operator()(const Vector3& s) const;
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
    MR_HF_DECL RendererParamVaryingSpectrum(const Converter& c, const PVD& p);

    MR_HF_DECL Spectrum operator()(Vector<DIMS, Float> uvCoords) const;
    // Gradient Access
    MR_HF_DECL Spectrum operator()(Vector<DIMS, Float> uvCoords,
                                   Vector<DIMS, Float> dpdx,
                                   Vector<DIMS, Float> dpdy) const;
    // Direct Mip Access
    MR_HF_DECL Spectrum operator()(Vector<DIMS, Float> uvCoords,
                                   uint32_t mipLevel) const;
    MR_HF_DECL bool IsConstant() const;
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

// Concrete Identity Spectrum Converter
struct SpectrumConverterIdentity
{
    MR_PF_DECL Spectrum Convert(const Vector3&) const noexcept;
    MR_PF_DECL Spectrum Convert(const Spectrum&) const noexcept;
    // Adding this for IoR conversion (dispersion)
    // We could've added ConvertIoR function, but the input arguments
    // may change depending on the medium (unless we dictate a protocol)
    // This is little bit more scalable but for Identity converter this does not
    // makes sense. Medium logic should be careful to incorporate these
    MR_PF_DECL SpectrumWaves Wavelengths() const noexcept;
    // By design the identity converter is RGB
    // (MRay holds every texture, value etc. as RGB)
    static constexpr bool IsRGB = true;
};

using SpectrumConverterContextIdentity = SpectrumConverterContext<SpectrumConverterIdentity>;
static_assert(SpectrumConverterC<SpectrumConverterIdentity>,
              "\"SpectrumConverterIdentity\" do not satisfy \"SpectrumConverterC\" concept.");
static_assert(SpectrumConverterContextC<SpectrumConverterContextIdentity>,
              "\"SpectrumConverterContextIdentity\" do not satisfy \"SpectrumConverterContextC\" concept." );

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

namespace SpectrumConverterDetail
{

template<class C>
MR_HF_DEF
RendererSpectrum<C>::RendererSpectrum(const C& c)
    : c(c)
{}

template<class C>
MR_HF_DEF
Spectrum RendererSpectrum<C>::operator()(const Vector3& s) const
{
    return c.Convert(s);
}

template<class C, uint32_t D>
MR_HF_DEF
RendererParamVaryingSpectrum<C, D>::RendererParamVaryingSpectrum(const C& c, const ParamVaryingData<D, Vector3>& p)
    : converter(c)
    , input(p)
{}

template<class C, uint32_t D>
MR_HF_DEF
Spectrum RendererParamVaryingSpectrum<C, D>::operator()(Vector<D, Float> uvCoords) const
{
    return converter.get().Convert(input.get()(uvCoords));
}

template<class C, uint32_t D>
MR_HF_DEF
Spectrum RendererParamVaryingSpectrum<C, D>::operator()(Vector<D, Float> uvCoords,
                                                        Vector<D, Float> dpdx,
                                                        Vector<D, Float> dpdy) const
{
    return converter.get().Convert(input.get()(uvCoords, dpdx, dpdy));
}

template<class C, uint32_t D>
MR_HF_DEF
Spectrum RendererParamVaryingSpectrum<C, D>::operator()(Vector<D, Float> uvCoords,
                                                        uint32_t mipLevel) const
{
    return converter.get().Convert(input.get()(uvCoords, mipLevel));
}

template<class C, uint32_t D>
MR_HF_DEF
bool RendererParamVaryingSpectrum<C, D>::IsConstant() const
{
    return input.get().IsConstant();
}

}

MR_PF_DEF
Spectrum SpectrumConverterIdentity::Convert(const Vector3& c) const noexcept
{
    return Spectrum(c, Float(0));
}

MR_PF_DEF
Spectrum SpectrumConverterIdentity::Convert(const Spectrum& c) const noexcept
{
    return c;
}

MR_PF_DEF
SpectrumWaves SpectrumConverterIdentity::Wavelengths() const noexcept
{
    return SpectrumWaves(VisibleSpectrumMiddle);
}