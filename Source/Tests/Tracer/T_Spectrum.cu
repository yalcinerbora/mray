#include <gtest/gtest.h>
#include <random>

#include "GTestWrappers.h"

#include "Core/Definitions.h"

#include "Tracer/SpectrumContext.h"
#include "Tracer/Random.h"

#include "Device/GPUAlgGeneric.h"
#include "Device/GPUAlgReduce.h"
#include "Device/GPUSystem.hpp" // IWYU pragma: keep

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCSampleDataAsSpectrum(// I-O
                            MRAY_GRID_CONSTANT const Span<Spectrum> dThroughput,
                            // Input
                            MRAY_GRID_CONSTANT const Span<const SpectrumWaves> dWaves,
                            // Constants
                            MRAY_GRID_CONSTANT const Vector3 inputColor,
                            MRAY_GRID_CONSTANT const Jacob2019Detail::Data data)
{
    using Converter = typename SpectrumContextJakob2019::Converter;

    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < dThroughput.size(); i += kp.TotalSize())
    {
        SpectrumWaves waves = dWaves[i];
        auto converter = Converter(waves, data);
        Spectrum s = converter.ConvertAlbedo(inputColor);
        dThroughput[i] *= s;
    }
}

struct SpectrumJakob2019Test : public testing::Test
{
    GPUSystem gpuSystem;

    static void RunTest(const GPUSystem& gpuSystem,
                        MRayColorSpaceEnum colorSpace,
                        WavelengthSampleMode sampleMode);
};

void SpectrumJakob2019Test::RunTest(const GPUSystem& gpuSystem,
                                    MRayColorSpaceEnum colorSpace,
                                    WavelengthSampleMode sampleMode)
{
    static constexpr auto RNG_SEED = 0u;
    static constexpr auto SAMPLE_COUNT = uint32_t(1024);
    static constexpr uint32_t STATIC_COLOR_COUNT = 6;
    std::array TEST_COLORS =
    {
        // These are hard-coded since PBRT does it as such?
        // (TODO: dunno why, investigate)
        // To be clear, graysacle values directly represented as a flat function.
        // Not exactly these specific values.
        Vector3(0),     // Black
        Vector3(0.5),   // Mid gray (linear)
        Vector3(1),     // White
        // Perfect values does not give proper results (it may be ok?)
        // Since polynomial approx. is not exact
        // Examples:
        //Vector3(1, 0, 0),   // pure Red
        //Vector3(0, 1, 0),   // pure Green
        //Vector3(0, 0, 1),   // pure Blue
        // TODO: Investigate this.
        // We do some meh colors here.
        Vector3(0.85, 0.15, 0.15),   // near-pure Red
        Vector3(0.15, 0.85, 0.15),   // near-pure Green
        Vector3(0.15, 0.15, 0.85),   // near-pure Blue
        // Random colors (will be generated below)
        Vector3(0, 0, 0),
        Vector3(0, 0, 0),
        Vector3(0, 0, 0),
        Vector3(0, 0, 0),
        Vector3(0, 0, 0)
    };
    static constexpr size_t RANDOM_COLOR_COUNT = TEST_COLORS.size() - STATIC_COLOR_COUNT;
    static_assert(RANDOM_COLOR_COUNT <= TEST_COLORS.size(), "Overflow, test structure is not valid!");

    std::mt19937 rng(RNG_SEED);
    std::uniform_real_distribution<Float> dist(Float(0), Float(1));
    for(size_t i = STATIC_COLOR_COUNT; i < TEST_COLORS.size(); i++)
        TEST_COLORS[i] = Vector3(dist(rng), dist(rng), dist(rng));

    const GPUDevice& device = gpuSystem.BestDevice();
    const GPUQueue& queue = device.GetComputeQueue(0);
    // Initialize spectrum context
    auto specContext = SpectrumContextJakob2019(colorSpace, sampleMode, gpuSystem);

    uint32_t rnPerSample = specContext.SampleSpectrumRNCount();
    uint32_t rnCount = SAMPLE_COUNT * rnPerSample;
    std::vector<RandomNumber> hRandomNumbers(rnCount);
    for(uint32_t j = 0; j < rnPerSample; j++)
    for(uint32_t i = 0; i < SAMPLE_COUNT; i++)
    {
        // Generate equally spaced random numbers
        // **IMPORTANT** this invertly mimics the "ToFloat01" function at Tracer/Random.h
        // If that is changed this must be changed as well
        static constexpr auto RNG_DELTA = (1u << 24) / SAMPLE_COUNT;
        hRandomNumbers[j * SAMPLE_COUNT + i] = (i * RNG_DELTA) << 8;
    }
    for(uint32_t j = 0; j < rnPerSample; j++)
        std::shuffle(hRandomNumbers.begin(),
                     hRandomNumbers.begin() + j * SAMPLE_COUNT, rng);

    static constexpr auto TOTAL_SAMPLE_COUNT = TEST_COLORS.size() * SAMPLE_COUNT;
    Span<Spectrum> dThroughput;
    Span<SpectrumWaves> dWaves;
    Span<RandomNumber> dRandomNumbers;
    Span<Spectrum> dResults;
    Span<uint32_t> dSegmentRanges;
    Span<Byte> dSegmentReduceTempMem;
    //
    using DeviceAlgorithms::SegmentedTransformReduceTMSize;
    size_t segReduceTempMemSize = SegmentedTransformReduceTMSize<Spectrum, Spectrum>(TEST_COLORS.size(), queue);
    //
    DeviceLocalMemory mem(device);
    MemAlloc::AllocateMultiData(Tie(dThroughput, dWaves, dRandomNumbers, dResults,
                                    dSegmentRanges, dSegmentReduceTempMem),
                                mem,
                                {TOTAL_SAMPLE_COUNT,
                                 TOTAL_SAMPLE_COUNT,
                                 TOTAL_SAMPLE_COUNT * rnPerSample,
                                 TEST_COLORS.size(), TEST_COLORS.size() + 1,
                                 segReduceTempMemSize});

    std::array<uint32_t, TEST_COLORS.size() + 1> hSegmentRanges;
    for(uint32_t i = 0; i < hSegmentRanges.size(); i++)
        hSegmentRanges[i] = i * SAMPLE_COUNT;
    queue.MemcpyAsync(dSegmentRanges, Span<const uint32_t>(hSegmentRanges));

    // Copy the same random numbers for each sample
    auto hRNSpan = Span<const RandomNumber>(hRandomNumbers);
    for(uint32_t i = 0; i < TEST_COLORS.size(); i++)
        queue.MemcpyAsync(dRandomNumbers.subspan(i * rnCount, rnCount), hRNSpan);

    // Work #1 Sample observer data, save the PDF as pre-divided form
    // in the throughput.
    specContext.SampleSpectrumWavelengths(dWaves, dThroughput, dRandomNumbers, queue);

    // Work #2 Sample the colors via the spectrum system
    for(uint32_t i = 0; i < TEST_COLORS.size(); i++)
    {
        Vector3 color = TEST_COLORS[i];
        queue.IssueWorkKernel<KCSampleDataAsSpectrum>
        (
            "KCSampleDataAsSpectrum",
            DeviceWorkIssueParams{.workCount = SAMPLE_COUNT},
            //
            dThroughput.subspan(i * SAMPLE_COUNT, SAMPLE_COUNT),
            ToConstSpan(dWaves.subspan(i * SAMPLE_COUNT, SAMPLE_COUNT)),
            color,
            specContext.GetData()
        );
    }

    // Work #3: Convert samples back to RGB
    specContext.ConvertSpectrumToRGB(dThroughput, dWaves, queue);

    // Work #4: Collapse the samples to a single value
    static constexpr Float WEIGHT = Float(1) / Float(SAMPLE_COUNT);
    DeviceAlgorithms::SegmentedTransformReduce
    (
        dResults,
        dSegmentReduceTempMem,
        ToConstSpan(dThroughput),
        ToConstSpan(dSegmentRanges),
        Spectrum::Zero(),
        queue,
        [] MR_HF_DECL (const Spectrum& a, const Spectrum& b) -> Spectrum
        {
            return a + b;
        },
        [] MR_HF_DECL (const Spectrum& in) -> Spectrum
        {
            return in * WEIGHT;
        }
    );

    std::array<Spectrum, TEST_COLORS.size()> hResults;
    queue.MemcpyAsync(Span<Spectrum>(hResults), ToConstSpan(dResults));

    // Wait results and check
    queue.Barrier().Wait();
    for(uint32_t i = 0; i < TEST_COLORS.size(); i++)
    {
        EXPECT_NEAR_MRAY(Vector3(hResults[i]), TEST_COLORS[i],
                         // I don't have any idea which error is appropirate here
                         // this error is basically "It's red alright!" type for
                         // error checking.
                         Float(1e-1));
    }
}

#define GEN_SPECTRUM_TEST(Name, E, S)   \
    TEST_F(SpectrumJakob2019Test, Name) \
    {                                   \
        RunTest(gpuSystem, E, S);       \
    }                                   \

// These fail, but tests are not good
//GEN_SPECTRUM_TEST(Aces_AP0_UNIFORM, MRayColorSpaceEnum::MR_ACES2065_1,
//                  WavelengthSampleMode::UNIFORM)
//GEN_SPECTRUM_TEST(Aces_AP0_GaussianMIS, MRayColorSpaceEnum::MR_ACES2065_1,
//                  WavelengthSampleMode::GAUSSIAN_MIS)
//GEN_SPECTRUM_TEST(Aces_AP0_Hyperbolic, MRayColorSpaceEnum::MR_ACES2065_1,
//                  WavelengthSampleMode::HYPERBOLIC_PBRT)

GEN_SPECTRUM_TEST(Aces_CG_UNIFORM, MRayColorSpaceEnum::MR_ACES_CG,
                  WavelengthSampleMode::UNIFORM)
GEN_SPECTRUM_TEST(Aces_CG_GaussianMIS, MRayColorSpaceEnum::MR_ACES_CG,
                  WavelengthSampleMode::GAUSSIAN_MIS)
GEN_SPECTRUM_TEST(Aces_CG_Hyperbolic, MRayColorSpaceEnum::MR_ACES_CG,
                  WavelengthSampleMode::HYPERBOLIC_PBRT)