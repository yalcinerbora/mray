#pragma once

#include "SpectrumC.h"
#include "ParamVaryingData.h"

#include "Device/GPUTexture.h"
#include "Core/TracerEnums.h"

// Jakob2019 Spectrum to RGB
// https://rgl.epfl.ch/publications/Jakob2019Spectral
namespace Jakob2019Detail
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
        // Normalized SPD of Std. Observer and
        // Colorspace's Std. Illuminant
        TextureView<1, Vector3>   spdObserverXYZ;
        TextureView<1, Float>     spdIlluminant;
    };

    using Data = DataT<64>;

    class Converter
    {
        public:
        using Data = Jakob2019Detail::Data;

        private:
        // Path local context
        const Data&     data;
        SpectrumWaves&  dWavelengthsRef;
        SpectrumWaves   wavelengths;

        MR_GF_DECL
        Vector3     FetchCoeffs(uint32_t i, Vector3 uv) const noexcept;

        public:
        // Constructors & Destructor
        MR_HF_DECL               Converter(SpectrumWaves& wavelengths, const Data&);

        // Interface
        MR_GF_DECL Spectrum      ConvertAlbedo(const Vector3&) const noexcept;
        MR_GF_DECL Spectrum      ConvertRadiance(const Vector3&) const noexcept;
        MR_PF_DECL SpectrumWaves Wavelengths() const noexcept;
        MR_PF_DECL_V void        DisperseWaves() noexcept;
        MR_PF_DECL_V void        StoreWaves() const noexcept;
        //
        static constexpr bool IsRGB = false;
    };
}

class SpectrumContextJakob2019
{
    using LUTTextureList = std::array<Texture<3, Float>, 9>;

    public:
    using Converter = Jakob2019Detail::Converter;
    using Data      = Jakob2019Detail::Data;
    // Texture-backed Data
    template<uint32_t DIMS>
    using ParamVaryingAlbedo = SpectralParamVaryingData<Converter, DIMS, false>;

    template<uint32_t DIMS>
    using ParamVaryingRadiance = SpectralParamVaryingData<Converter, DIMS, true>;

    private:
    const GPUSystem&        gpuSystem;
    TextureBackingMemory    texMem;
    Texture<1, Vector3>     texCIE1931_XYZ;
    Texture<1, Float>       texStdIlluminant;
    LUTTextureList          lutTextures;
    Data                    data;
    WavelengthSampleMode    sampleMode;
    MRayColorSpaceEnum      colorSpace;

    // Helpers
    LUTTextureList LoadSpectraLUT(MRayColorSpaceEnum globalColorSpace,
                                  const GPUDevice& device);

    public:
    // Constructors Destructor
    // Class name is too long...
    using This = SpectrumContextJakob2019;
            SpectrumContextJakob2019(MRayColorSpaceEnum, WavelengthSampleMode, const GPUSystem&);
            SpectrumContextJakob2019(const This&) = delete;
            SpectrumContextJakob2019(This&&) = delete;
    This&   operator=(const This&) = delete;
    This&   operator=(This&&) = delete;
            ~SpectrumContextJakob2019() = default;

    // Methods...
    Data GetData() const;

    //
    void SampleSpectrumWavelengths(// Output
                                   Span<SpectrumWaves> dWavelengths,
                                   Span<Spectrum> dWavePDFs,
                                   // I-O
                                   Span<const RandomNumber> dRandomNumbers,
                                   // Constants
                                   const GPUQueue& queue) const;
    void SampleSpectrumWavelengthsIndirect(// Output
                                           Span<SpectrumWaves> dWavelengths,
                                           Span<Spectrum> dWavePDFs,
                                           // Input
                                           Span<const RandomNumber> dRandomNumbers,
                                           Span<const RayIndex> dRayIndices,
                                           // Constants
                                           const GPUQueue& queue) const;
    RNRequestList SampleSpectrumRNList() const;

    //
    void ConvertSpectrumToRGB(// I-O
                              Span<Spectrum> dValues,
                              // Input
                              Span<const SpectrumWaves> dWavelengths,
                              Span<const Spectrum> dWavePDFs,
                              // Constants
                              const GPUQueue& queue) const;
    void ConvertSpectrumToRGBIndirect(// I-O
                                      Span<Spectrum> dValues,
                                      // Input
                                      Span<const SpectrumWaves> dWavelengths,
                                      Span<const Spectrum> dWavePDFs,
                                      Span<const RayIndex> dRayIndices,
                                      // Constants
                                      const GPUQueue& queue) const;
    //
    MRayColorSpaceEnum  ColorSpace() const;
    size_t              GPUMemoryUsage() const;
};

inline Jakob2019Detail::Data
SpectrumContextJakob2019::GetData() const
{
    return data;
}

inline MRayColorSpaceEnum
SpectrumContextJakob2019::ColorSpace() const
{
    return colorSpace;
}

inline size_t
SpectrumContextJakob2019::GPUMemoryUsage() const
{
    return texMem.Size();
}

#include "SpectrumContext.hpp"

static_assert(SpectrumConverterC<Jakob2019Detail::Converter>,
              "\"Jacob2019Detail::Converter\" do not satisfy \"SpectrumConverterC\" concept.");
static_assert(SpectrumContextC<SpectrumContextJakob2019>,
              "\"SpectrumContextJakob2019\" do not satisfy \"SpectrumConverterContextC\" concept.");

