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
                                    Spectrum& throughput,
                                    // I-O
                                    BackupRNG& rng,
                                    // Constant
                                    WavelengthSampleMode mode)
{
    // We pre-divide with the PDF here,
    // thats why throughput is required
    switch(mode)
    {
        case UNIFORM:
        {
            constexpr auto START = Float(Color::CIE_1931_RANGE[0]);
            constexpr auto END = Float(Color::CIE_1931_RANGE[1] - 1);
            using Distribution::Common::SampleUniformRange;

            MRAY_UNROLL_LOOP_N(SpectraPerSpectrum)
            for(uint32_t i = 0; i < SpectraPerSpectrum; i++)
            {
                Float xi = rng.NextFloat();
                SampleT<Float> sample = SampleUniformRange(xi, START, END);
                wavelengths[i] = sample.value;
                throughput[i] = Float(1);
            }
            break;
        }
        case GAUSSIAN_MIS:
        {
            using namespace Distribution;
            using namespace Distribution::Common;

            MRAY_UNROLL_LOOP_N(SpectraPerSpectrum)
            for(uint32_t i = 0; i < SpectraPerSpectrum; i++)
            {
                // This is for CUDA, it does not like inline constexprs
                constexpr auto CIE_1931_GAUSS_SIGMA = Color::CIE_1931_GAUSS_SIGMA;
                constexpr auto CIE_1931_GAUSS_MU    = Color::CIE_1931_GAUSS_MU;
                constexpr auto CIE_1931_MIS         = Color::CIE_1931_MIS;

                uint32_t sampleI = 0, otherI = 1;
                if(rng.NextFloat() >= CIE_1931_MIS[0])
                    std::swap(sampleI, otherI);

                // TODO: Normalize reuse sample?
                SampleT<Float> sample = SampleGaussian(rng.NextFloat(),
                                                       CIE_1931_GAUSS_SIGMA[sampleI],
                                                       CIE_1931_GAUSS_MU[sampleI]);
                Float otherPDF = PDFGaussian(rng.NextFloat(),
                                             CIE_1931_GAUSS_SIGMA[otherI],
                                             CIE_1931_GAUSS_MU[otherI]);

                auto pdfs = std::array{sample.pdf, otherPDF};
                auto weights = std::array{CIE_1931_MIS[sampleI], CIE_1931_MIS[otherI]};
                auto pdf = MIS::BalanceCancelled(Span<Float, 2>(pdfs),
                                                 Span<Float, 2>(weights));

                wavelengths[i] = sample.value;
                throughput[i] = Float(1) / pdf;
            }
            break;
        }
        default: HybridTerminateOrTrap("Unkown wavelength sample mode!");
    }
}

MR_GF_DECL
Spectrum ConvertSpectraToRGBSingle(const Spectrum& value, const SpectrumWaves& waves,
                                   //
                                   const TextureView<1, Vector3>& observerResponseXYZ,
                                   const Matrix3x3& xyzToRGB)
{
    Vector3 xyzTotal = Vector3::Zero();
    for(uint32_t i = 0; i < SpectraPerSpectrum; i++)
    {
        Float index = waves[i] - Float(Color::CIE_1931_RANGE[0]);
        Vector3 factors = observerResponseXYZ(index);
        xyzTotal += factors * value[i];
    }
    return Spectrum(xyzToRGB * xyzTotal, 0);
}

MRAY_KERNEL
void KCSampleSpectrumWavelengths(// Output
                                 MRAY_GRID_CONSTANT const Span<SpectrumWaves> dWavelengths,
                                 MRAY_GRID_CONSTANT const Span<Spectrum> dThroughputs,
                                 // I-O
                                 MRAY_GRID_CONSTANT const Span<BackupRNGState> dRNGStates,
                                 // Constants
                                 MRAY_GRID_CONSTANT const WavelengthSampleMode mode)
{
    // Grid-stride Loop
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < dRNGStates.size(); i += kp.TotalSize())
    {
        BackupRNG rng(dRNGStates[i]);

        SpectrumWaves wavelengths;
        Spectrum throughput;
        SingleSampleSpectrumWavelength(wavelengths, throughput,
                                       rng, mode);

        dWavelengths[i] = wavelengths;
        dThroughputs[i] = throughput;
    }
}

MRAY_KERNEL
void KCSampleSpectrumWavelengthsIndirect(// Output
                                         MRAY_GRID_CONSTANT const Span<SpectrumWaves> dWavelengths,
                                         MRAY_GRID_CONSTANT const Span<Spectrum> dThroughputs,
                                         // I-O
                                         MRAY_GRID_CONSTANT const Span<BackupRNGState> dRNGStates,
                                         // Input
                                         MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices,
                                         // Constants
                                         MRAY_GRID_CONSTANT const WavelengthSampleMode mode)
{
    // Grid-stride Loop
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < dRayIndices.size(); i += kp.TotalSize())
    {
        RayIndex rIndex = dRayIndices[i];
        BackupRNG rng(dRNGStates[rIndex]);

        SpectrumWaves wavelengths;
        Spectrum throughput;
        SingleSampleSpectrumWavelength(wavelengths, throughput, rng, mode);

        dWavelengths[rIndex] = wavelengths;
        dThroughputs[rIndex] = throughput;
    }
}

MRAY_KERNEL
void KCConvertSpectrumToRGB(// I-O
                            MRAY_GRID_CONSTANT const Span<Spectrum> dValues,
                            // Input
                            MRAY_GRID_CONSTANT const Span<const SpectrumWaves> dWavelengths,
                            // Constants
                            MRAY_GRID_CONSTANT const Jacob2019Detail::Data contextData)
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
        dValues[i] = ConvertSpectraToRGBSingle(dValues[i], dWavelengths[i],
                                               contextData.spdObserverXYZ,
                                               sXYZToRGB);
    }
}

MRAY_KERNEL
void KCConvertSpectrumToRGBIndirect(// I-O
                                    MRAY_GRID_CONSTANT const Span<Spectrum> dValues,
                                    // Input
                                    MRAY_GRID_CONSTANT const Span<const SpectrumWaves> dWavelengths,
                                    MRAY_GRID_CONSTANT const Span<const RayIndex> dRayIndices,
                                    // Constants
                                    MRAY_GRID_CONSTANT const Jacob2019Detail::Data contextData)
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
                                               contextData.spdObserverXYZ,
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
    if(!lutFile || resolution != Jacob2019Detail::Data::N)
        return MRayError("Wrong size ({}), spectra lut size must be {}!",
                         resolution, Jacob2019Detail::Data::N);

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
    constexpr uint32_t N = Jacob2019Detail::Data::N;
    constexpr Vector3ui TEX_DIMS = Vector3ui(N);
    TextureInitParams<3> tp =
    {
        .size = TEX_DIMS,
        .normIntegers = false,
        .interp = MRayTextureInterpEnum::MR_LINEAR,
        .eResolve = MRayTextureEdgeResolveEnum::MR_CLAMP
    };
    std::array outputTextures =
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

    using MemAlloc::AllocateTextureSpace;
    std::vector<size_t> offsets = AllocateTextureSpace(texMem, sizes, alignments);
    for(uint32_t i = 0; i < 9; i++)
    {
        Texture<3, Float>& t = outputTextures[i];
        t.CommitMemory(copyQueue, texMem, offsets[i]);
    }
    texCIE1931_XYZ.CommitMemory(copyQueue, texMem, offsets.back());

    // Sadly, CIE_1931_XYZ is not padded...
    // We pad the data here
    // TODO: This probably is not a bottleneck but we can store
    // CIE_1931_XYZ in a padded fashion maybe.
    std::vector<Vector4> paddedCIE1931ObserverData(Color::CIE_1931_N);
    for(uint32_t i = 0; i < Color::CIE_1931_N; i++)
        paddedCIE1931ObserverData[i] = Vector4(Color::CIE_1931_XYZ[i], 0);

    texCIE1931_XYZ.CopyFromAsync(copyQueue, 0, 0, Color::CIE_1931_N,
                                 Span<const Vector4>(paddedCIE1931ObserverData));

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
    for(uint32_t i = 0; i < 9; i++)
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
    , lutTextures(LoadSpectraLUT(globalColorspace, gpuSystem.BestDevice()))
    , data(Jacob2019Detail::Data
           {
               .lut =
               {
                   Jacob2019Detail::Data::Table3D
                   {
                       .c0 = lutTextures[0].View<Float>(),
                       .c1 = lutTextures[1].View<Float>(),
                       .c2 = lutTextures[2].View<Float>()
                   },
                   Jacob2019Detail::Data::Table3D
                   {
                       .c0 = lutTextures[3].View<Float>(),
                       .c1 = lutTextures[4].View<Float>(),
                       .c2 = lutTextures[5].View<Float>()
                   },
                   Jacob2019Detail::Data::Table3D
                   {
                       .c0 = lutTextures[6].View<Float>(),
                       .c1 = lutTextures[7].View<Float>(),
                       .c2 = lutTextures[8].View<Float>()
                   },
               },
               .spdObserverXYZ = texCIE1931_XYZ.View<Vector3>()
           })
    , sampleMode(sampleMode)
{
    data.RGBToXYZ = Color::SelectRGBToXYZMatrix(globalColorspace);
    data.XYZToRGB = data.RGBToXYZ.Inverse();
}

void SpectrumContextJakob2019::SampleSpectrumWavelengths(// Output
                                                         Span<SpectrumWaves> dWavelengths,
                                                         Span<Spectrum> dThroughputs,
                                                         // I-O
                                                         Span<BackupRNGState> dRNGStates,
                                                         // Constants
                                                         const GPUQueue& queue)
{
    assert(dWavelengths.size() == dThroughputs.size());
    assert(dThroughputs.size() == dRNGStates.size());

    queue.IssueWorkKernel<KCSampleSpectrumWavelengths>
    (
        "KCSampleSpectraWavelengths",
        DeviceWorkIssueParams{.workCount = dWavelengths.size()},
        //
        dWavelengths,
        dThroughputs,
        dRNGStates,
        sampleMode
    );

}

void SpectrumContextJakob2019::SampleSpectrumWavelengthsIndirect(// Output
                                                                 Span<SpectrumWaves> dWavelengths,
                                                                 Span<Spectrum> dThroughputs,
                                                                 // I-O
                                                                 Span<BackupRNGState> dRNGStates,
                                                                 // Input
                                                                 Span<const RayIndex> dRayIndices,
                                                                 // Constants
                                                                 const GPUQueue& queue)
{
    queue.IssueWorkKernel<KCSampleSpectrumWavelengthsIndirect>
    (
        "KCSampleSpectraWavelengths",
        DeviceWorkIssueParams{.workCount = dRayIndices.size()},
        //
        dWavelengths,
        dThroughputs,
        dRNGStates,
        dRayIndices,
        sampleMode
    );
}

void SpectrumContextJakob2019::ConvertSpectrumToRGB(// I-O
                                                    Span<Spectrum> dValues,
                                                    // Input
                                                    Span<const SpectrumWaves> dWavelengths,
                                                    // Constants
                                                    const GPUQueue& queue)
{
    assert(dValues.size() == dWavelengths.size());
    queue.IssueWorkKernel<KCConvertSpectrumToRGB>
    (
        "KCSampleSpectraWavelengths",
        DeviceWorkIssueParams{.workCount = dValues.size()},
        //
        dValues,
        dWavelengths,
        data
    );
}

void SpectrumContextJakob2019::ConvertSpectrumToRGBIndirect(// I-O
                                                            Span<Spectrum> dValues,
                                                            // Input
                                                            Span<const SpectrumWaves> dWavelengths,
                                                            Span<const RayIndex> dRayIndices,
                                                            // Constants
                                                            const GPUQueue& queue)
{
    queue.IssueWorkKernel<KCConvertSpectrumToRGBIndirect>
    (
        "KCSampleSpectraWavelengths",
        DeviceWorkIssueParams{.workCount = dRayIndices.size()},
        //
        dValues,
        dWavelengths,
        dRayIndices,
        data
    );
}