#pragma once

#include "SpectrumC.h"
#include "ParamVaryingData.h"

#include "Device/GPUTexture.h"

// TODO: Move this to a user facing system (user can set this via config etc.
enum WavelengthSampleMode : uint32_t
{
    UNIFORM,
    GAUSSIAN_MIS
};

// Jakob2019 Spectrum to RGB
// https://rgl.epfl.ch/publications/Jakob2019Spectral
namespace Jacob2019Detail
{
    template<uint32_t Resolution>
    struct DataT
    {
        static constexpr uint32_t N = Resolution;
        // GPU textures pad extra float if we use 3 channel
        // textures
        struct Table3D
        {
            // These are polynomial coeffcients
            // given lambda(l), evaluation is
            // (c0 * l^2) + (c1 * l) + c2
            // Then according to paper, we will do nonlinear mapping
            TextureView<3, Float> c0;
            TextureView<3, Float> c1;
            TextureView<3, Float> c2;
        };
        // RGBToXYZ conversion matrices
        // These are not compiletime static since generating
        // a converter for each colorspace probably not worth it.
        Matrix3x3   RGBToXYZ;
        Matrix3x3   XYZToRGB;
        //
        // TODO: Rename these when we actually implement this functionality
        // after throughly reading the paper.
        std::array<Table3D, 3>    lut;
        TextureView<1, Vector3>   spdObserverXYZ;
    };

    using Data = DataT<64>;

    class Converter
    {
        // Path local context
        const Data&     data;
        SpectrumWaves   wavelengths;

        MR_GF_DECL
        Vector3                  FetchCoeffs(uint32_t i, Vector3 uv) const;

        public:
        // Constructors & Destructor
        MR_HF_DECL
                                 Converter(SpectrumWaves wavelengths, const Data&);
        // Interface
        MR_GF_DECL Spectrum      ConvertAlbedo(const Vector3&) const noexcept;
        MR_GF_DECL Spectrum      ConvertRadiance(const Vector3&) const noexcept;
        MR_PF_DECL SpectrumWaves Wavelengths() const noexcept;
        //
        static constexpr bool IsRGB = false;
    };
}

class SpectrumContextJakob2019
{
    using LUTTextureList = std::array<Texture<3, Float>, 9>;

    public:
    using Converter = Jacob2019Detail::Converter;
    // Texture-backed Data
    template<uint32_t DIMS>
    using ParamVaryingAlbedo = SpectralParamVaryingData<Converter, DIMS, false>;

    template<uint32_t DIMS>
    using ParamVaryingRadiance = SpectralParamVaryingData<Converter, DIMS, true>;

    private:
    const GPUSystem&        gpuSystem;
    TextureBackingMemory    texMem;
    Texture<1, Vector3>     texCIE1931_XYZ;
    LUTTextureList          lutTextures;
    Jacob2019Detail::Data   data;
    WavelengthSampleMode    sampleMode;

    // Helpers
    LUTTextureList LoadSpectraLUT(MRayColorSpaceEnum globalColorSpace,
                                  const GPUDevice& device);

    public:
    // Constructors Destructor
    // Class name is too long...
    using This = SpectrumContextJakob2019;
            SpectrumContextJakob2019(MRayColorSpaceEnum, WavelengthSampleMode, const GPUSystem&);
            SpectrumContextJakob2019(const This&) = delete;
            SpectrumContextJakob2019(This&&) = default;
    This&   operator=(const This&) = delete;
    This&   operator=(This&&) = default;
            ~SpectrumContextJakob2019() = default;

    // Methods...
    Jacob2019Detail::Data Data() const;

    //
    void SampleSpectrumWavelengths(// Output
                                   Span<SpectrumWaves> dWavelengths,
                                   Span<Spectrum> dThroughputs,
                                   // I-O
                                   Span<BackupRNGState> dRNGStates,
                                   // Constants
                                   const GPUQueue& queue);
    void SampleSpectrumWavelengthsIndirect(// Output
                                           Span<SpectrumWaves> dWavelengths,
                                           Span<Spectrum> dThroughputs,
                                           // I-O
                                           Span<BackupRNGState> dRNGStates,
                                           // Input
                                           Span<const RayIndex> dRayIndices,
                                           // Constants
                                           const GPUQueue& queue);

    //
    void ConvertSpectrumToRGB(// I-O
                              Span<Spectrum> dValues,
                              // Input
                              Span<const SpectrumWaves> dWavelengths,
                              // Constants
                              const GPUQueue& queue);
    void ConvertSpectrumToRGBIndirect(// I-O
                                      Span<Spectrum> dValues,
                                      // Input
                                      Span<const SpectrumWaves> dWavelengths,
                                      Span<const RayIndex> dRayIndices,
                                      // Constants
                                      const GPUQueue& queue);

};

inline Jacob2019Detail::Data
SpectrumContextJakob2019::Data() const
{
    return data;
}

#include "SpectrumContext.hpp"

static_assert(SpectrumConverterC<Jacob2019Detail::Converter>,
              "\"Jacob2019Detail::Converter\" do not satisfy \"SpectrumConverterC\" concept.");
static_assert(SpectrumConverterContextC<SpectrumContextJakob2019>,
              "\"SpectrumContextJakob2019\" do not satisfy \"SpectrumConverterContextC\" concept.");

