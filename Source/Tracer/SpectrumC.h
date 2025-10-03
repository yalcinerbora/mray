#pragma once

#include "Core/Vector.h"

#include "TracerTypes.h"
#include "ParamVaryingData.h"

template <class C>
concept SpectrumConverterC = requires(const C cConst,
                                      C c)
{
    typename C::Data;

    {cConst.ConvertAlbedo(Vector3{})} -> std::same_as<Spectrum>;
    {cConst.ConvertRadiance(Vector3{})} -> std::same_as<Spectrum>;
    {cConst.Wavelengths()} -> std::same_as<SpectrumWaves>;
    {c.DisperseWaves()} -> std::same_as<void>;
    {cConst.StoreWaves()} -> std::same_as<void>;

    // TODO: Why this does not work?
    // ====================
    //{C::IsRGB} -> std::same_as<const bool>;
    // ====================
    C::IsRGB;
    requires std::is_same_v<decltype(C::IsRGB), const bool>;
};

template <class C>
concept SpectrumContextC = requires(const C c, const GPUQueue & queue)
{
    typename C::Converter;
    requires SpectrumConverterC<typename C::Converter>;

    typename C::Data;

    //
    typename C::template ParamVaryingAlbedo<2>;
    typename C::template ParamVaryingRadiance<2>;

    // TODO: Move these to virtual inheritance maybe.
    {c.GetData()} -> std::same_as<typename C::Data>;
    {c.ColorSpace()} -> std::same_as<MRayColorSpaceEnum>;
    {c.SampleSpectrumRNList()} -> std::same_as<RNRequestList>;
    {c.SampleSpectrumWavelengthsIndirect(// Output
                                         Span<SpectrumWaves>(),
                                         Span<Spectrum>(),
                                         // Input
                                         Span<const RandomNumber>(),
                                         Span<const RayIndex>(),
                                         // Constants
                                         queue)} -> std::same_as<void>;
    {c.ConvertSpectrumToRGBIndirect(// I-O
                                    Span<Spectrum>(),
                                    // Input
                                    Span<const SpectrumWaves>(),
                                    Span<const Spectrum>(),
                                    Span<const RayIndex>(),
                                    // Constants
                                    queue)} -> std::same_as<void>;
    {c.GPUMemoryUsage()} -> std::same_as<size_t>;
};

template<class Converter, uint32_t DIMS, bool IsRadiance = false>
class SpectralParamVaryingData
{
    using PVD = ParamVaryingData<DIMS, Vector3>;

    private:
    const Converter*    converter;
    const PVD*          input;

    public:
    //
    MR_HF_DECL SpectralParamVaryingData(const Converter& c, const PVD& p);

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

// Concrete Identity Spectrum Converter
struct SpectrumConverterIdentity
{
    using Data = EmptyType;

    MR_PF_DECL Spectrum ConvertAlbedo(const Vector3&) const noexcept;
    MR_PF_DECL Spectrum ConvertRadiance(const Vector3&) const noexcept;
    // Adding this for IoR conversion (dispersion)
    // We could've added ConvertIoR function, but the input arguments
    // may change depending on the medium (unless we dictate a protocol)
    // This is little bit more scalable but for Identity converter this does not
    // makes sense. Medium logic should be careful to incorporate these
    MR_PF_DECL SpectrumWaves Wavelengths() const noexcept;
    MR_PF_DECL_V void DisperseWaves() const noexcept;
    MR_PF_DECL_V void StoreWaves() const noexcept;
    // By design the identity converter is RGB
    // (MRay holds every texture, value etc. as RGB)
    static constexpr bool IsRGB = true;
};

struct SpectrumContextIdentity
{
    using Converter = SpectrumConverterIdentity;
    using Data      = EmptyType;

    // Texture-backed Data
    template<uint32_t DIMS>
    using ParamVaryingAlbedo = SpectralParamVaryingData<Converter, DIMS, false>;

    template<uint32_t DIMS>
    using ParamVaryingRadiance = SpectralParamVaryingData<Converter, DIMS, true>;

    public:
    MRayColorSpaceEnum colorSpace = MRayColorSpaceEnum::MR_DEFAULT;

    public:
    // Consturctors & Destructor
    SpectrumContextIdentity() = default;
    SpectrumContextIdentity(MRayColorSpaceEnum cs, WavelengthSampleMode, const GPUSystem&)
        : colorSpace(cs)
    {}

    // TODO: Move these to virtual inheritance maybe.
    EmptyType GetData() const { return EmptyType{}; }
    MRayColorSpaceEnum ColorSpace() const { return colorSpace; }
    RNRequestList SampleSpectrumRNList() const { return RNRequestList(); }
    //
    void SampleSpectrumWavelengthsIndirect(// Output
                                           Span<SpectrumWaves>,
                                           Span<Spectrum>,
                                           // Input
                                           Span<const RandomNumber>,
                                           Span<const RayIndex>,
                                           // Constants
                                           const GPUQueue&) const {}

    void ConvertSpectrumToRGBIndirect(// I-O
                                      Span<Spectrum>,
                                      // Input
                                      Span<const SpectrumWaves>,
                                      Span<const Spectrum>,
                                      Span<const RayIndex>,
                                      // Constants
                                      const GPUQueue&) const {}

    size_t GPUMemoryUsage() const { return 0; }
};

template<class C, uint32_t D, bool IsRadiance>
MR_HF_DEF
SpectralParamVaryingData<C, D, IsRadiance>::SpectralParamVaryingData(const C& c, const ParamVaryingData<D, Vector3>& p)
    : converter(&c)
    , input(&p)
{}

template<class C, uint32_t D, bool IsRadiance>
MR_HF_DEF
Spectrum
SpectralParamVaryingData<C, D, IsRadiance>::operator()(Vector<D, Float> uvCoords) const
{
    if constexpr(IsRadiance)
        return converter->ConvertRadiance((*input)(uvCoords));
    else
        return converter->ConvertAlbedo((*input)(uvCoords));
}

template<class C, uint32_t D, bool IsRadiance>
MR_HF_DEF
Spectrum
SpectralParamVaryingData<C, D, IsRadiance>::operator()(Vector<D, Float> uvCoords,
                                                       Vector<D, Float> dpdx,
                                                       Vector<D, Float> dpdy) const
{
    if constexpr(IsRadiance)
        return converter->ConvertRadiance((*input)(uvCoords, dpdx, dpdy));
    else
        return converter->ConvertAlbedo((*input)(uvCoords, dpdx, dpdy));
}

template<class C, uint32_t D, bool IsRadiance>
MR_HF_DEF
Spectrum
SpectralParamVaryingData<C, D, IsRadiance>::operator()(Vector<D, Float> uvCoords,
                                                       uint32_t mipLevel) const
{
    if constexpr(IsRadiance)
        return converter->ConvertRadiance((*input)(uvCoords, mipLevel));
    else
        return converter->ConvertAlbedo((*input)(uvCoords, mipLevel));
}

template<class C, uint32_t D, bool IsRadiance>
MR_HF_DEF
bool SpectralParamVaryingData<C, D, IsRadiance>::IsConstant() const
{
    return input->IsConstant();
}

MR_PF_DEF
Spectrum SpectrumConverterIdentity::ConvertAlbedo(const Vector3& c) const noexcept
{
    return Spectrum(c, Float(0));
}

MR_PF_DEF
Spectrum SpectrumConverterIdentity::ConvertRadiance(const Vector3& c) const noexcept
{
    return Spectrum(c, Float(0));
}

MR_PF_DEF
SpectrumWaves SpectrumConverterIdentity::Wavelengths() const noexcept
{
    return SpectrumWaves(VisibleSpectrumMiddle);
}

MR_PF_DEF_V
void SpectrumConverterIdentity::DisperseWaves() const noexcept
{}

MR_PF_DEF_V
void SpectrumConverterIdentity::StoreWaves() const noexcept
{}

static_assert(SpectrumConverterC<SpectrumConverterIdentity>,
              "\"SpectrumConverterIdentity\" do not satisfy \"SpectrumConverterC\" concept.");
static_assert(SpectrumContextC<SpectrumContextIdentity>,
              "\"SpectrumContextIdentity\" do not satisfy \"SpectrumConverterContextC\" concept.");