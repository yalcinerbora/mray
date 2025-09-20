#include <gtest/gtest.h>
#include <random>

#include "GTestWrappers.h"

#include "Core/Definitions.h"

#include "Tracer/SpectrumContext.h"
#include "Tracer/Random.h"

#include "Device/GPUAlgGeneric.h"
#include "Device/GPUAlgReduce.h"
#include "Device/GPUSystem.hpp" // IWYU pragma: keep

// TODO:
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
        auto converter = Converter(dWaves[i], data);
        Spectrum s = converter.ConvertAlbedo(inputColor);
        dThroughput[i] *= s;
    }
}

template<MRayColorSpaceEnum COLOR_SPACE,
         WavelengthSampleMode SAMPLE_MODE>
struct Jakob2019TestParams
{
    static constexpr auto ColorSpace = COLOR_SPACE;
    static constexpr auto SampleMode = SAMPLE_MODE;
};

using Implementations = ::testing::Types
<
    //Jakob2019TestParams<MRayColorSpaceEnum::MR_ACES2065_1,
    //                    WavelengthSampleMode::GAUSSIAN_MIS>,
    Jakob2019TestParams<MRayColorSpaceEnum::MR_ACES2065_1,
                        WavelengthSampleMode::UNIFORM>
    //,
    //Jakob2019TestParams<MRayColorSpaceEnum::MR_ACES_CG,
    //                    WavelengthSampleMode::GAUSSIAN_MIS>,
    //Jakob2019TestParams<MRayColorSpaceEnum::MR_ACES_CG,
    //                    WavelengthSampleMode::UNIFORM>
    // TODO: Add more later or selectively open close
    // so that we dont waste time etc.
>;

template<class T>
struct SpectrumJakob2019Test : public testing::Test
{
    //static constexpr MRayColorSpaceEnum ColorSpace = T::ColorSpace;
};

TYPED_TEST_SUITE(SpectrumJakob2019Test, Implementations);

void TestJakob2019Pipeline(MRayColorSpaceEnum CS, WavelengthSampleMode MODE)
{
    static constexpr auto RNG_SEED = 0u;
    static constexpr auto SAMPLE_COUNT = uint32_t(1024);
    //static constexpr uint32_t STATIC_COLOR_COUNT = 5;
    static constexpr uint32_t STATIC_COLOR_COUNT = 1;
    std::array TEST_COLORS =
    {
        //Vector3(0.5, 0.5, 0.5)
        Vector3(0.9, 0, 0)

        //Vector3(0, 0, 0),   // Black
        //Vector3(1, 1, 1),   // White
        //Vector3(1, 0, 0),   // Pure Red
        //Vector3(0, 1, 0),   // Pure Green
        //Vector3(0, 0, 1),   // Pure Blue
        //// Random colors
        //Vector3(0, 0, 0),
        //Vector3(0, 0, 0),
        //Vector3(0, 0, 0),
        //Vector3(0, 0, 0),
        //Vector3(0, 0, 0)
    };
    static constexpr size_t RANDOM_COLOR_COUNT = TEST_COLORS.size() - STATIC_COLOR_COUNT;
    static_assert(RANDOM_COLOR_COUNT <= TEST_COLORS.size(), "Overflow, test structure is not valid!");

    std::mt19937 rng(RNG_SEED);
    std::uniform_real_distribution<Float> dist(Float(0), Float(1));
    for(size_t i = STATIC_COLOR_COUNT; i < TEST_COLORS.size(); i++)
        TEST_COLORS[i] = Vector3(dist(rng), dist(rng), dist(rng));


    GPUSystem gpuSystem;
    //ThreadPool tp(typename ThreadPool::InitParams(1));
    //GPUSystem gpuSystem(tp);

    const GPUDevice& device = gpuSystem.BestDevice();
    const GPUQueue& queue = device.GetComputeQueue(0);
    // Initialize spectrum context
    auto specContext = SpectrumContextJakob2019(CS, MODE, gpuSystem);

    static constexpr auto TOTAL_SAMPLE_COUNT = TEST_COLORS.size() * SAMPLE_COUNT;
    Span<Spectrum> dThroughput;
    Span<SpectrumWaves> dWaves;
    Span<BackupRNGState> dRNGStates;
    Span<Spectrum> dResults;
    Span<uint32_t> dSegmentRanges;
    Span<Byte> dSegmentReduceTempMem;
    //
    using DeviceAlgorithms::SegmentedTransformReduceTMSize;
    size_t segReduceTempMemSize = SegmentedTransformReduceTMSize<Spectrum, Spectrum>(TEST_COLORS.size(), queue);
    //
    DeviceLocalMemory mem(device);
    MemAlloc::AllocateMultiData(Tie(dThroughput, dWaves, dRNGStates, dResults,
                                    dSegmentRanges, dSegmentReduceTempMem),
                                mem,
                                {TOTAL_SAMPLE_COUNT,
                                 TOTAL_SAMPLE_COUNT,
                                 TOTAL_SAMPLE_COUNT,
                                 TEST_COLORS.size(), TEST_COLORS.size() + 1,
                                 segReduceTempMemSize});

    std::array<uint32_t, TEST_COLORS.size() + 1> hSegmentRanges;
    for(uint32_t i = 0; i < hSegmentRanges.size(); i++)
        hSegmentRanges[i] = i * SAMPLE_COUNT;
    queue.MemcpyAsync(dSegmentRanges, Span<const uint32_t>(hSegmentRanges));

    // TODO: We churn the RNG once per sample, this is not proper.
    // each color should use single engine and sampled "SAMPLE_COUNT"
    // amount of times from that engine but SpectrumContext API
    // is not designed for it since it will be used as API suggested in the renderers.
    //
    // In order to save time here (since RNGs cant be churned in parallel) we
    // create a RNG for each sample. Thus; we can issue a single kernel instead of
    // issuing "SAMPLE_COUNT" amount of kernels
    std::vector<BackupRNGState> hRNGStates(TOTAL_SAMPLE_COUNT);
    for(BackupRNGState& state : hRNGStates)
        state = PermutedCG32::GenerateState(rng());
    queue.MemcpyAsync(dRNGStates, Span<const BackupRNGState>(hRNGStates));

    // Work #1 Sample observer data, save the PDF as pre-divided form
    // in the throughput.
    specContext.SampleSpectrumWavelengths(dWaves, dThroughput, dRNGStates, queue);

    queue.Barrier().Wait();

    // Work #2 Sample the colors via the spectrum system
    for(uint32_t i = 0; i < TEST_COLORS.size(); i++)
    {
        Vector3 color = TEST_COLORS[i];
        queue.IssueWorkKernel<KCSampleDataAsSpectrum>
        (
            "KCSampleDataAsSpectrum",
            DeviceWorkIssueParams{.workCount = SAMPLE_COUNT},
            //
            dThroughput,
            ToConstSpan(dWaves),
            color,
            specContext.Data()
        );
    }

    queue.Barrier().Wait();

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
                         MathConstants::LargeEpsilon<Float>());
    }
}

TYPED_TEST(SpectrumJakob2019Test, Pipeline)
{
    static constexpr auto CS = TypeParam::ColorSpace;
    static constexpr auto MODE = TypeParam::SampleMode;

    //for(uint32_t i = 0; i < 64; i++)
        TestJakob2019Pipeline(CS, MODE);
}