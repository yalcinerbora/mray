#include "SpectrumContext.h"

#include "Device/GPUTexture.h"
#include "Device/GPUSystem.hpp"

#include "Core/System.h"
#include "Core/ColorFunctions.h"

#include <filesystem>
#include <fstream>

#include "DistributionFunctions.h"

MR_GF_DECL
void SingleSampleSpectrumWavelength(// Output
                                    SpectrumWaves& wavelengths,
                                    Spectrum& pdfOut,
                                    // I-O
                                    RNGDispenser& rng,
                                    // Constant
                                    WavelengthSampleMode mode)
{
    static constexpr auto START = Float(Color::CIE_1931_RANGE[0]);
    static constexpr auto END = Float(Color::CIE_1931_RANGE[1] - 1);
    // Similar to the PBRT, be bisect the sample space
    // and use single sample to sample multiple times.
    static constexpr auto SAMPLE_SPACE_OFFSETS = []()
    {
        constexpr auto DELTA = Float(1) / Float(SpectraPerSpectrum);
        std::array<Float, SpectraPerSpectrum> result = {};
        constexpr int32_t mid = int32_t(SpectraPerSpectrum / 2);
        for(int32_t i = 0; i < SpectraPerSpectrum; i++)
        {
            result[i] = Float(i - mid) * DELTA;
        }
        return result;
    }();

    static_assert([]()
    {
        bool allInRange = true;
        for(auto offset : SAMPLE_SPACE_OFFSETS)
            allInRange &= (Math::Abs(offset) < Float(1));
        return allInRange;
    }(),
    "The code below assumes offsets does not exceed the sample range. "
    "It seems that is not the case!");

    // Sample expansion
    Float xi0 = rng.NextFloat<0>();
    std::array<Float, SpectraPerSpectrum> xi;
    MRAY_UNROLL_LOOP_N(SpectraPerSpectrum)
    for(uint32_t i = 0; i < SpectraPerSpectrum; i++)
    {
        Float x = xi0 + SAMPLE_SPACE_OFFSETS[i];
        // If offset overflows the range, rollover
        if(x < Float(0))    x += Float(1);
        if(x >= Float(1))   x -= Float(1);

        xi[i] = x;
    }

    // We pre-divide with the PDF here,
    // thats why throughput is required

    switch(mode.e)
    {
        case WavelengthSampleMode::UNIFORM:
        {
            Float pdf = Float(0);
            MRAY_UNROLL_LOOP_N(SpectraPerSpectrum)
            for(uint32_t i = 0; i < SpectraPerSpectrum; i++)
            {
                using Distribution::Common::SampleUniformRange;
                SampleT<Float> sample = SampleUniformRange(xi[i], START, END);
                wavelengths[i] = sample.value;
                if(i == 0) pdf = sample.pdf;
            }
            pdfOut = Spectrum(pdf);
            break;
        }
        case WavelengthSampleMode::GAUSSIAN_MIS:
        {
            using namespace Distribution;
            using namespace Distribution::Common;
            // This is for CUDA, it does not like inline constexprs
            constexpr auto SIGMA = Color::CIE_1931_GAUSS_SIGMA;
            constexpr auto MU    = Color::CIE_1931_GAUSS_MU;
            constexpr auto MIS   = Color::CIE_1931_MIS;

            MRAY_UNROLL_LOOP_N(SpectraPerSpectrum)
            for(uint32_t i = 0; i < SpectraPerSpectrum; i++)
            {
                auto [sampleI, localXi] = BisectSample2(xi[i], MIS, true);
                uint32_t otherI = (sampleI + 1) & 0b1;

                SampleT<Float> sample = SampleGaussian(localXi, SIGMA[sampleI], MU[sampleI]);
                Float otherPDF = PDFGaussian(sample.value, SIGMA[otherI], MU[otherI]);

                auto pdfs = std::array{sample.pdf, otherPDF};
                auto weights = std::array{MIS[sampleI], MIS[otherI]};
                auto pdf = MIS::BalanceCancelled(Span<Float, 2>(pdfs),
                                                 Span<Float, 2>(weights));
                // Gaussian can be funky, check the values
                assert(Math::IsFinite(pdf));
                assert(Math::IsFinite(sample.value));

                wavelengths[i] = sample.value;
                pdfOut[i] = pdf;
            }
            break;
        }
        case WavelengthSampleMode::HYPERBOLIC_PBRT:
        {
            auto PDF = [](Float lambda)
            {
                Float denom = Math::CosH(Float(0.0072) * (lambda - Float(538)));
                denom = denom * denom;
                return Float(0.0039398042) / denom;
            };
            auto Sample = [](Float xi)
            {
                Float a = Float(0.85691062) - Float(1.82750197) * xi;
                return Float(538) - Float(138.888889) * Math::ArcTanH(a);
            };

            MRAY_UNROLL_LOOP_N(SpectraPerSpectrum)
            for(uint32_t i = 0; i < SpectraPerSpectrum; i++)
            {
                wavelengths[i] = Sample(xi[i]);
                pdfOut[i] = PDF(wavelengths[i]);
            }
            break;
        }
        default: HybridTerminateOrTrap("Unkown wavelength sample mode!");
    }
}

MR_GF_DECL
Spectrum ConvertSpectraToRGBSingle(const Spectrum& value, const SpectrumWaves& waves,
                                   const Spectrum& pdf,
                                   //
                                   const TextureView<1, Vector3>& observerResponseXYZ,
                                   const Matrix3x3& sXYZToRGB)
{
    static constexpr auto WEIGHT = Float(1) / Float(SpectraPerSpectrum);
    Vector3 xyzTotal = Vector3::Zero();

    uint32_t waveCount = SpectraPerSpectrum;

    // We technically done "SpectraPerSpectrum" samples not 1.
    // Compansate for that
    Float weight = WEIGHT;
    if(waves.IsDispersed())
    {
        // If dispersed other weights does mean nothing
        waveCount = 1;
        weight = Float(1);
    }
    //
    for(uint32_t i = 0; i < waveCount; i++)
    {
        static constexpr auto OFFSET = Float(0.5) - Float(Color::CIE_1931_RANGE[0]);
        Vector3 factors = observerResponseXYZ(waves[i] + OFFSET);
        Float val = Distribution::Common::DivideByPDF(value[i], pdf[i]);
        xyzTotal += factors * val;

    }
    //
    xyzTotal *= weight;
    Spectrum result = Spectrum(sXYZToRGB * xyzTotal, 0);
    return result;
}

MRAY_KERNEL
void KCSampleSpectrumWavelengths(// Output
                                 MRAY_GRID_CONSTANT const Span<SpectrumWaves> dWavelengths,
                                 MRAY_GRID_CONSTANT const Span<Spectrum> dWavePDFs,
                                 // I-O
                                 MRAY_GRID_CONSTANT const Span<const RandomNumber> dRandNumbers,
                                 // Constants
                                 MRAY_GRID_CONSTANT const WavelengthSampleMode mode)
{
    // Grid-stride Loop
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < dWavelengths.size(); i += kp.TotalSize())
    {
        RNGDispenser rng = RNGDispenser(dRandNumbers, i, dWavelengths.size());

        SpectrumWaves wavelengths;
        Spectrum pdf;
        SingleSampleSpectrumWavelength(wavelengths, pdf,
                                       rng, mode);

        dWavelengths[i] = wavelengths;
        dWavePDFs[i] = pdf;
    }
}

MRAY_KERNEL
void KCSampleSpectrumWavelengthsIndirect(// Output
                                         MRAY_GRID_CONSTANT const Span<SpectrumWaves> dWavelengths,
                                         MRAY_GRID_CONSTANT const Span<Spectrum> dWavePDFs,
                                         // I-O
                                         MRAY_GRID_CONSTANT const Span<const RandomNumber> dRandNumbers,
                                         // Input
                                         MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices,
                                         // Constants
                                         MRAY_GRID_CONSTANT const WavelengthSampleMode mode)
{
    // Grid-stride Loop
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < dRayIndices.size(); i += kp.TotalSize())
    {
        RNGDispenser rng = RNGDispenser(dRandNumbers, i, dRayIndices.size());

        RayIndex rIndex = dRayIndices[i];

        SpectrumWaves wavelengths;
        Spectrum pdf;
        SingleSampleSpectrumWavelength(wavelengths, pdf, rng, mode);

        dWavelengths[rIndex] = wavelengths;
        dWavePDFs[rIndex] = pdf;
    }
}

MRAY_KERNEL
void KCConvertSpectrumToRGB(// I-O
                            MRAY_GRID_CONSTANT const Span<Spectrum> dValues,
                            // Input
                            MRAY_GRID_CONSTANT const Span<const SpectrumWaves> dWavelengths,
                            MRAY_GRID_CONSTANT const Span<const Spectrum> dWavePDFs,
                            // Constants
                            MRAY_GRID_CONSTANT const Jakob2019Detail::Data contextData)
{
    KernelCallParams kp;

    #ifndef MRAY_GPU_BACKEND_CPU
        MRAY_SHARED_MEMORY Matrix3x3 sXYZToRGB;
        if(kp.threadId < 9)
            sXYZToRGB[kp.threadId] = contextData.XYZToRGB[kp.threadId];
        BlockSynchronize();

    #else
        const Matrix3x3& sXYZToRGB = contextData.XYZToRGB;

    #endif

    // Grid-stride Loop
    for(uint32_t i = kp.GlobalId(); i < dValues.size(); i += kp.TotalSize())
    {
        dValues[i] = ConvertSpectraToRGBSingle(dValues[i], dWavelengths[i], dWavePDFs[i],
                                               contextData.spdObserverXYZ,
                                               sXYZToRGB);
    }
}

MRAY_KERNEL
void KCConvertSpectrumToRGBIndirect(// I-O
                                    MRAY_GRID_CONSTANT const Span<Spectrum> dValues,
                                    // Input
                                    MRAY_GRID_CONSTANT const Span<const SpectrumWaves> dWavelengths,
                                    MRAY_GRID_CONSTANT const Span<const Spectrum> dWavePDFs,
                                    MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices,
                                    // Constants
                                    MRAY_GRID_CONSTANT const Jakob2019Detail::Data contextData)
{
    KernelCallParams kp;
    #ifndef MRAY_GPU_BACKEND_CPU
        MRAY_SHARED_MEMORY Matrix3x3 sXYZToRGB;
        if(kp.threadId < 9)
            sXYZToRGB[kp.threadId] = contextData.XYZToRGB[kp.threadId];
        BlockSynchronize();

    #else
        const Matrix3x3& sXYZToRGB = contextData.XYZToRGB;

    #endif

    // Grid-stride Loop
    for(uint32_t i = kp.GlobalId(); i < dRayIndices.size(); i += kp.TotalSize())
    {
        RayIndex rIndex = dRayIndices[i];
        dValues[rIndex] = ConvertSpectraToRGBSingle(dValues[rIndex], dWavelengths[rIndex],
                                                    dWavePDFs[rIndex], contextData.spdObserverXYZ,
                                                    sXYZToRGB);
    }
}

template<class T>
auto ToCharPtr(T* ptr)
-> std::conditional_t<std::is_const_v<T>, const char*, char*>
{
    using Ret = std::conditional_t<std::is_const_v<T>,
                                   const char*, char*>;
    return reinterpret_cast<Ret>(ptr);
};

MRayError
ReadMRSpectraFileHeader(std::ifstream& dataStartStream,
                        MRayColorSpaceEnum globalColorSpace)
{
    std::string pPath = GetProcessPath();
    namespace fs = std::filesystem;

    std::string_view colorspaceName = MRayColorSpaceStringifier::ToString(globalColorSpace);
    fs::path fullPath = fs::path(pPath) / fs::path(Color::LUT_FOLDER_NAME);
    fullPath /= std::string(colorspaceName) + std::string(Color::LUT_FILE_EXT);

    if(!fs::exists(fullPath))
        return MRayError("Unable to open spectra lut file {}!",
                         fullPath.string());

    //
    constexpr auto CC_SIZE = Color::LUT_FILE_CC.size();
    std::ifstream lutFile(fullPath, std::ios_base::binary);
    // CC
    std::array<char, CC_SIZE> charCode;
    constexpr auto REVERSE_FOUR_CC = []()
    {
        static_assert(CC_SIZE >= 4, "LUT_FILE_CC must be larger than 4");
        std::array<char, 4> result;
        std::copy_n(Color::LUT_FILE_CC.data(), 4,
                    result.data());
        std::reverse(result.begin(), result.end());
        return result;
    }();
    std::string_view charCodeStr(charCode.data(), CC_SIZE);
    std::string_view charCode4CC = charCodeStr.substr(0, 4);
    lutFile.read(charCode.data(), Color::LUT_FILE_CC.size());
    if(!lutFile || charCode4CC == std::string_view(REVERSE_FOUR_CC.data(), 4))
        return MRayError("Wrong character code in .mrspectra file! Endianness issue maybe");
    if(!lutFile || charCodeStr != Color::LUT_FILE_CC)
        return MRayError("Wrong character code in .mrspectra file!");

    // Resolution
    uint32_t resolution = std::numeric_limits<uint32_t>::max();
    lutFile.read(ToCharPtr(&resolution), sizeof(uint32_t));
    if(!lutFile || resolution != Jakob2019Detail::Data::N)
        return MRayError("Wrong size ({}), spectra lut size must be {}!",
                         resolution, Jakob2019Detail::Data::N);

    // Mode
    uint32_t mode = std::numeric_limits<uint32_t>::max();
    lutFile.read(ToCharPtr(&mode), sizeof(uint32_t));
    if(!lutFile || mode != 1)
        return MRayError("Wrong mode ({}), spectra lut must have mode "
                         "\"1\" (aka. Float)!", mode);

    // Actual data is at the cursor
    dataStartStream = std::move(lutFile);
    return MRayError::OK;
}

typename SpectrumContextJakob2019::LUTTextureList
SpectrumContextJakob2019::LoadSpectraLUT(MRayColorSpaceEnum globalColorSpace,
                                         const GPUDevice& device)
{
    const GPUQueue& copyQueue = device.GetTransferQueue();
    // To increase runtime performance LUT size is static
    constexpr uint32_t N = Jakob2019Detail::Data::N;
    constexpr Vector3ui TEX_DIMS = Vector3ui(N);
    TextureInitParams<3> tp =
    {
        .size = TEX_DIMS,
        .normIntegers = false,
        .interp = MRayTextureInterpEnum::MR_LINEAR,
        .eResolve = MRayTextureEdgeResolveEnum::MR_CLAMP
    };
    LUTTextureList outputTextures =
    {
        Texture<3, Float>(device, tp), Texture<3, Float>(device, tp),
        Texture<3, Float>(device, tp), Texture<3, Float>(device, tp),
        Texture<3, Float>(device, tp), Texture<3, Float>(device, tp),
        Texture<3, Float>(device, tp), Texture<3, Float>(device, tp),
        Texture<3, Float>(device, tp)
    };
    //
    std::vector<size_t> sizes;
    std::vector<size_t> alignments;
    for(const Texture<3, Float>& t : outputTextures)
    {
        sizes.push_back(t.Size());
        alignments.push_back(t.Alignment());
    }
    sizes.push_back(texCIE1931_XYZ.Size());
    alignments.push_back(texCIE1931_XYZ.Alignment());
    sizes.push_back(texStdIlluminant.Size());
    alignments.push_back(texStdIlluminant.Alignment());

    using MemAlloc::AllocateTextureSpace;
    std::vector<size_t> offsets = AllocateTextureSpace(texMem, sizes, alignments);
    for(uint32_t i = 0; i < outputTextures.size(); i++)
    {
        Texture<3, Float>& t = outputTextures[i];
        t.CommitMemory(copyQueue, texMem, offsets[i]);
    }
    size_t lastOffsetI = offsets.size() - size_t(1);
    texCIE1931_XYZ.CommitMemory(copyQueue, texMem, offsets[lastOffsetI - 1]);
    texStdIlluminant.CommitMemory(copyQueue, texMem, offsets[lastOffsetI]);

    // Sadly, CIE_1931_XYZ is not padded...
    // We pad the data here, fortunately
    // we can normalize so that every sample do not have to
    // do 3 extra multiplication.
    std::vector<Vector4> paddedCIE1931ObserverData(Color::CIE_1931_N);
    std::vector<Float> normalizedSPDIlluminant(Color::CIE_1931_N);
    const auto& constIllumSPD = Color::SelectIlluminantSPD(globalColorSpace);
    // Illuminant normalization factor is not the direct integral result,
    // it is factored differently (from PBRT).
    Float illumSPDNormFactor = Color::CIE_1931_Y_INTEGRAL;
    illumSPDNormFactor /= Color::SelectIlluminantSPDNormFactor(globalColorSpace);
    //
    for(uint32_t i = 0; i < Color::CIE_1931_N; i++)
    {
        static constexpr Vector3 CIE_SUM = Vector3(Color::CIE_1931_X_INTEGRAL,
                                                   Color::CIE_1931_Y_INTEGRAL,
                                                   Color::CIE_1931_Z_INTEGRAL);
        static constexpr Vector3 WEIGHT = Vector3(1) / CIE_SUM;
        paddedCIE1931ObserverData[i] = Vector4(Color::CIE_1931_XYZ[i] * WEIGHT, 0);
        //
        normalizedSPDIlluminant[i] = constIllumSPD[i] * illumSPDNormFactor;
    }

    texCIE1931_XYZ.CopyFromAsync(copyQueue, 0, 0, Color::CIE_1931_N,
                                 Span<const Vector4>(paddedCIE1931ObserverData));
    texStdIlluminant.CopyFromAsync(copyQueue, 0, 0, Color::CIE_1931_N,
                                   Span<const Float>(normalizedSPDIlluminant));

    // Lets try a double buffered approach here
    constexpr uint32_t BUFFER_COUNT = 2;
    constexpr uint32_t FLOAT_COUNT = N * N * N;
    constexpr uint32_t BUFFER_SIZE = sizeof(Float) * FLOAT_COUNT;
    HostLocalMemory tempMem(gpuSystem);
    tempMem.ResizeBuffer(BUFFER_SIZE * BUFFER_COUNT);
    //
    Byte* hBufferPtr = static_cast<Byte*>(tempMem);
    Float* hBufferFloatPtr = reinterpret_cast<Float*>(hBufferPtr);
    std::array hBuffers =
    {
        Span<Float>(hBufferFloatPtr , FLOAT_COUNT),
        Span<Float>(hBufferFloatPtr + FLOAT_COUNT, FLOAT_COUNT),
    };
    // Read resolution and header
    std::ifstream fileDataStart;
    MRayError e = ReadMRSpectraFileHeader(fileDataStart, globalColorSpace);
    if(e) throw e;

    // Double buffered read
    std::array fences = {copyQueue.Barrier(), copyQueue.Barrier()};
    uint32_t curI = 0, otherI = 1;
    for(uint32_t i = 0; i < outputTextures.size(); i++)
    {
        if(!fileDataStart.read(ToCharPtr(hBuffers[curI].data()), BUFFER_SIZE))
            throw MRayError("Unable to read sprectum lut file!");
        if constexpr(MRAY_IS_DEBUG)
        {
            for(const auto& v : hBuffers[curI])
                assert(Math::IsFinite(v));
        }

        outputTextures[i].CopyFromAsync(copyQueue, 0,
                                        Vector3ui::Zero(), Vector3ui(N),
                                        ToConstSpan(hBuffers[curI]));
        fences[curI] = copyQueue.Barrier();
        fences[otherI].Wait();
        std::swap(curI, otherI);
    }

    fences[otherI].Wait();
    return outputTextures;
}

SpectrumContextJakob2019::SpectrumContextJakob2019(MRayColorSpaceEnum globalColorspace,
                                                   WavelengthSampleMode sampleMode,
                                                   const GPUSystem& gpuSystem)
    : gpuSystem(gpuSystem)
    // TODO: Multi-GPU support...
    , texMem(gpuSystem.BestDevice())
    , texCIE1931_XYZ(gpuSystem.BestDevice(),
                     TextureInitParams<1>
                     {
                         .size = Color::CIE_1931_N,
                         .normIntegers = false,
                         .normCoordinates = false,
                         .interp = MRayTextureInterpEnum::MR_LINEAR,
                         .eResolve = MRayTextureEdgeResolveEnum::MR_CLAMP
                     })
    , texStdIlluminant(gpuSystem.BestDevice(),
                       TextureInitParams<1>
                       {
                           .size = Color::CIE_1931_N,
                           .normIntegers = false,
                           .normCoordinates = false,
                           .interp = MRayTextureInterpEnum::MR_LINEAR,
                           .eResolve = MRayTextureEdgeResolveEnum::MR_CLAMP
                       })
    , lutTextures(LoadSpectraLUT(globalColorspace, gpuSystem.BestDevice()))
    , data(Jakob2019Detail::Data
           {
               .lut =
               {
                   Jakob2019Detail::Data::Table3D
                   {
                       .c0 = lutTextures[0].View<Float>(),
                       .c1 = lutTextures[1].View<Float>(),
                       .c2 = lutTextures[2].View<Float>()
                   },
                   Jakob2019Detail::Data::Table3D
                   {
                       .c0 = lutTextures[3].View<Float>(),
                       .c1 = lutTextures[4].View<Float>(),
                       .c2 = lutTextures[5].View<Float>()
                   },
                   Jakob2019Detail::Data::Table3D
                   {
                       .c0 = lutTextures[6].View<Float>(),
                       .c1 = lutTextures[7].View<Float>(),
                       .c2 = lutTextures[8].View<Float>()
                   },
               },
               .spdObserverXYZ = texCIE1931_XYZ.View<Vector3>(),
               .spdIlluminant = texStdIlluminant.View<Float>()
           })
    , sampleMode(sampleMode)
    , colorSpace(globalColorspace)
{
    data.RGBToXYZ = Color::SelectRGBToXYZMatrix(globalColorspace);
    data.XYZToRGB = data.RGBToXYZ.Inverse();
}

void SpectrumContextJakob2019::SampleSpectrumWavelengths(// Output
                                                         Span<SpectrumWaves> dWavelengths,
                                                         Span<Spectrum> dWavePDFs,
                                                         // I-O
                                                         Span<const RandomNumber> dRandomNumbers,
                                                         // Constants
                                                         const GPUQueue& queue) const
{
    assert(dWavelengths.size() == dWavePDFs.size());
    assert(dWavePDFs.size() * SampleSpectrumRNList().TotalRNCount() == dRandomNumbers.size());

    queue.IssueWorkKernel<KCSampleSpectrumWavelengths>
    (
        "KCSampleSpectraWavelengths",
        DeviceWorkIssueParams{.workCount = dWavelengths.size()},
        //
        dWavelengths,
        dWavePDFs,
        dRandomNumbers,
        sampleMode
    );

}

void SpectrumContextJakob2019::SampleSpectrumWavelengthsIndirect(// Output
                                                                 Span<SpectrumWaves> dWavelengths,
                                                                 Span<Spectrum> dWavePDFs,
                                                                 // Input
                                                                 Span<const RandomNumber> dRandomNumbers,
                                                                 Span<const RayIndex> dRayIndices,
                                                                 // Constants
                                                                 const GPUQueue& queue) const
{
    assert(dRayIndices.size() * SampleSpectrumRNList().TotalRNCount() == dRandomNumbers.size());
    queue.IssueWorkKernel<KCSampleSpectrumWavelengthsIndirect>
    (
        "KCSampleSpectraWavelengths",
        DeviceWorkIssueParams{.workCount = dRayIndices.size()},
        //
        dWavelengths,
        dWavePDFs,
        dRandomNumbers,
        dRayIndices,
        sampleMode
    );
}

RNRequestList SpectrumContextJakob2019::SampleSpectrumRNList() const
{
    switch(sampleMode.e)
    {
        using enum WavelengthSampleMode::E;
        case UNIFORM:           return GenRNRequestList<1>();
        case GAUSSIAN_MIS:      return GenRNRequestList<1>();
        case HYPERBOLIC_PBRT:   return GenRNRequestList<1>();
        default: throw MRayError("Unkown spectrum sample mode!");
    }
}

void SpectrumContextJakob2019::ConvertSpectrumToRGB(// I-O
                                                    Span<Spectrum> dValues,
                                                    // Input
                                                    Span<const SpectrumWaves> dWavelengths,
                                                    Span<const Spectrum> dWavePDFs,
                                                    // Constants
                                                    const GPUQueue& queue) const
{
    assert(dValues.size() == dWavelengths.size());
    queue.IssueWorkKernel<KCConvertSpectrumToRGB>
    (
        "KCSampleSpectraWavelengths",
        DeviceWorkIssueParams{.workCount = dValues.size()},
        //
        dValues,
        dWavelengths,
        dWavePDFs,
        data
    );
}

void SpectrumContextJakob2019::ConvertSpectrumToRGBIndirect(// I-O
                                                            Span<Spectrum> dValues,
                                                            // Input
                                                            Span<const SpectrumWaves> dWavelengths,
                                                            Span<const Spectrum> dWavePDFs,
                                                            Span<const RayIndex> dRayIndices,
                                                            // Constants
                                                            const GPUQueue& queue) const
{
    queue.IssueWorkKernel<KCConvertSpectrumToRGBIndirect>
    (
        "KCSampleSpectraWavelengths",
        DeviceWorkIssueParams{.workCount = dRayIndices.size()},
        //
        dValues,
        dWavelengths,
        dWavePDFs,
        dRayIndices,
        data
    );
}