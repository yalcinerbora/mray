#include "RunCommand.h"
#include "TracerThread.h"

#include "Core/Timer.h"
#include "Core/MRayDescriptions.h"
#include "Core/Log.h"

#include "Common/RenderImageStructs.h"
#include "Common/TransferQueue.h"

#include <BS/BS_thread_pool.hpp>
#include <CLI/CLI.hpp>
#include <string_view>
#include <immintrin.h>

#define MRAY_EQUALS_CHAR_8 "========"
#define MRAY_EQUALS_CHAR_16 MRAY_EQUALS_CHAR_8 MRAY_EQUALS_CHAR_8
#define MRAY_EQUALS_CHAR_32 MRAY_EQUALS_CHAR_16 MRAY_EQUALS_CHAR_16

#define MRAY_SPACE_CHAR_8 "        "
#define MRAY_SPACE_CHAR_16 MRAY_SPACE_CHAR_8 MRAY_SPACE_CHAR_8
#define MRAY_SPACE_CHAR_32 MRAY_SPACE_CHAR_16 MRAY_SPACE_CHAR_16


namespace EyeAnim
{
    // Instead of pulling std::chorno_literals to global space (it is a single
    // translation unit but w/e), using constructors
    using DurationMS = std::chrono::milliseconds;
    using LegolasLookupElem = Pair<std::string_view, DurationMS>;

    static constexpr auto AnimDurationLong = DurationMS(850);
    static constexpr auto AnimDurationShort = DurationMS(475);
    static constexpr std::array LegolasAnimSheet =
    {
        LegolasLookupElem{"< 0 >", AnimDurationLong},
        LegolasLookupElem{"<  0>", AnimDurationLong},
        LegolasLookupElem{"< 0 >", AnimDurationLong},
        LegolasLookupElem{"< _ >", AnimDurationShort},
        LegolasLookupElem{"< 0 >", AnimDurationLong},
        LegolasLookupElem{"<0  >", AnimDurationLong},
        LegolasLookupElem{"< 0 >", AnimDurationLong},
        LegolasLookupElem{"< _ >", AnimDurationShort}
    };

    constexpr auto ScanLegolasAnimDurations()
    {
        uint64_t offset = 0;
        std::array<uint64_t, LegolasAnimSheet.size() + 1> result = {};
        for(size_t i = 0; i < LegolasAnimSheet.size(); i++)
        {
            const auto& keyFrame = LegolasAnimSheet[i];
            result[i] = offset;
            offset += keyFrame.second.count();
        }
        result.back() = offset;
        return result;
    }
    static constexpr std::array LegolasAnimDurations = ScanLegolasAnimDurations();

    class SimpleProgressBar
    {
        private:
        static constexpr std::string_view EQUALS = MRAY_EQUALS_CHAR_32;
        static constexpr std::string_view SPACES = MRAY_SPACE_CHAR_32;
        static constexpr size_t MAX_WIDTH = EQUALS.size();

        public:
        void Display(Float ratio, uint64_t timeMS, std::string_view prefix);
    };
}

void EyeAnim::SimpleProgressBar::Display(Float ratio, uint64_t timeMS,
                                         std::string_view prefix)
{
    // We query this everytime for adjusting the size
    auto terminalSize = GetTerminalSize();
    // TODO: Bound limit
    size_t leftover = terminalSize[0] - prefix.size() - 2 - 5 - 3;

    leftover = std::min(leftover, MAX_WIDTH);
    size_t eqCount = static_cast<uint32_t>(std::round(Float(leftover) * ratio));
    size_t spaceCount = leftover - eqCount;

    uint64_t localTime = timeMS % LegolasAnimDurations.back();
    auto loc = std::upper_bound(LegolasAnimDurations.cbegin(),
                                LegolasAnimDurations.cend(), localTime);
    std::ptrdiff_t animIndex = std::distance(LegolasAnimDurations.begin(), loc) - 1;
    assert(loc != LegolasAnimDurations.end());

    std::string_view animSprite = LegolasAnimSheet[animIndex].first;
    fmt::print("[{:s}{:s}] {:s} {:s}\n",
               EQUALS.substr(0, eqCount),
               SPACES.substr(0, spaceCount),
               animSprite, prefix);
}

namespace Accum
{
    static constexpr size_t RenderBufferAlignment = 4096;

    template<std::floating_point T>
    using RGBWeightSpan = SoASpan<T, T, T, T>;

    static constexpr uint32_t R = 0;
    static constexpr uint32_t G = 1;
    static constexpr uint32_t B = 2;
    static constexpr uint32_t W = 3;

    void AccumulateScanline(RGBWeightSpan<double> output,
                            RGBWeightSpan<const Float> input);

    BS::multi_future<void>
    AccumulateImage(RGBWeightSpan<double> output,
                    BS::thread_pool& threadPool,
                    const RenderImageSection&,
                    const RenderBufferInfo&);
}

void Accum::AccumulateScanline(RGBWeightSpan<double> output,
                               RGBWeightSpan<const Float> input)
{
    assert(output.Size() == input.Size());
    constexpr size_t SIMD_WIDTH = MRay::HostArchSIMDWidth<double>();

    auto Iteration_AVX512 = [&](size_t i)
    {
        __m512d rOut = _mm512_loadu_pd(output.Get<R>().data() + i * SIMD_WIDTH);
        __m512d gOut = _mm512_loadu_pd(output.Get<G>().data() + i * SIMD_WIDTH);
        __m512d bOut = _mm512_loadu_pd(output.Get<B>().data() + i * SIMD_WIDTH);
        __m512d sOut = _mm512_loadu_pd(output.Get<W>().data() + i * SIMD_WIDTH);
        //
        __m512d rIn, gIn, bIn, sIn;
        if constexpr(std::is_same_v<Float, float>)
        {
            rIn = _mm512_cvtps_pd(_mm256_loadu_ps(input.Get<R>().data() + i * SIMD_WIDTH));
            gIn = _mm512_cvtps_pd(_mm256_loadu_ps(input.Get<G>().data() + i * SIMD_WIDTH));
            bIn = _mm512_cvtps_pd(_mm256_loadu_ps(input.Get<B>().data() + i * SIMD_WIDTH));
            sIn = _mm512_cvtps_pd(_mm256_loadu_ps(input.Get<W>().data() + i * SIMD_WIDTH));
        }
        else
        {
            rIn = _mm512_loadu_pd(input.Get<R>().data() + i * SIMD_WIDTH);
            gIn = _mm512_loadu_pd(input.Get<G>().data() + i * SIMD_WIDTH);
            bIn = _mm512_loadu_pd(input.Get<B>().data() + i * SIMD_WIDTH);
            sIn = _mm512_loadu_pd(input.Get<W>().data() + i * SIMD_WIDTH);
        }
        //
        __m512d sTotal = _mm512_add_pd(sOut, sIn);
        __m512d sRecip = _mm512_rcp14_pd(sTotal);
        __m512d newR = _mm512_fmadd_pd(rOut, sOut, rIn);
        __m512d newG = _mm512_fmadd_pd(gOut, sOut, gIn);
        __m512d newB = _mm512_fmadd_pd(bOut, sOut, bIn);
        //
        newR = _mm512_mul_pd(newR, sRecip);
        newG = _mm512_mul_pd(newG, sRecip);
        newB = _mm512_mul_pd(newB, sRecip);
        //
        _mm512_storeu_pd(output.Get<R>().data() + i * SIMD_WIDTH, newR);
        _mm512_storeu_pd(output.Get<G>().data() + i * SIMD_WIDTH, newG);
        _mm512_storeu_pd(output.Get<B>().data() + i * SIMD_WIDTH, newB);
        _mm512_storeu_pd(output.Get<W>().data() + i * SIMD_WIDTH, sTotal);
    };

    auto Iteration_AVX2 = [&](size_t i)
    {
        __m256d rOut = _mm256_loadu_pd(output.Get<R>().data() + i * SIMD_WIDTH);
        __m256d gOut = _mm256_loadu_pd(output.Get<G>().data() + i * SIMD_WIDTH);
        __m256d bOut = _mm256_loadu_pd(output.Get<B>().data() + i * SIMD_WIDTH);
        __m256d sOut = _mm256_loadu_pd(output.Get<W>().data() + i * SIMD_WIDTH);
        //
        __m256d rIn, gIn, bIn, sIn;
        if constexpr(std::is_same_v<Float, float>)
        {
            rIn = _mm256_cvtps_pd(_mm_loadu_ps(input.Get<R>().data() + i * SIMD_WIDTH));
            gIn = _mm256_cvtps_pd(_mm_loadu_ps(input.Get<G>().data() + i * SIMD_WIDTH));
            bIn = _mm256_cvtps_pd(_mm_loadu_ps(input.Get<B>().data() + i * SIMD_WIDTH));
            sIn = _mm256_cvtps_pd(_mm_loadu_ps(input.Get<W>().data() + i * SIMD_WIDTH));
        }
        else
        {
            rIn = _mm256_loadu_pd(input.Get<R>().data() + i * SIMD_WIDTH);
            gIn = _mm256_loadu_pd(input.Get<G>().data() + i * SIMD_WIDTH);
            bIn = _mm256_loadu_pd(input.Get<B>().data() + i * SIMD_WIDTH);
            sIn = _mm256_loadu_pd(input.Get<W>().data() + i * SIMD_WIDTH);
        }
        //
        __m256d sTotal = _mm256_add_pd(sOut, sIn);
        __m256d sRecip = _mm256_rcp14_pd(sTotal);
        __m256d newR = _mm256_fmadd_pd(rOut, sOut, rIn);
        __m256d newG = _mm256_fmadd_pd(gOut, sOut, gIn);
        __m256d newB = _mm256_fmadd_pd(bOut, sOut, bIn);
        //
        newR = _mm256_mul_pd(newR, sRecip);
        newG = _mm256_mul_pd(newG, sRecip);
        newB = _mm256_mul_pd(newB, sRecip);
        //
        _mm256_storeu_pd(output.Get<R>().data() + i * SIMD_WIDTH, newR);
        _mm256_storeu_pd(output.Get<G>().data() + i * SIMD_WIDTH, newG);
        _mm256_storeu_pd(output.Get<B>().data() + i * SIMD_WIDTH, newB);
        _mm256_storeu_pd(output.Get<W>().data() + i * SIMD_WIDTH, sTotal);
    };

    auto Iteration_Common = [&](size_t i)
    {
        double rOut = output.Get<R>()[i];
        double gOut = output.Get<G>()[i];
        double bOut = output.Get<B>()[i];
        double sOut = output.Get<W>()[i];
        //
        Float rIn = input.Get<R>()[i];
        Float gIn = input.Get<G>()[i];
        Float bIn = input.Get<B>()[i];
        Float sIn = input.Get<W>()[i];
        //
        double totalSample = sOut + double(sIn);
        double recip = double(1) / totalSample;
        double newR = (rOut * sOut + rIn) * recip;
        double newG = (gOut * sOut + gIn) * recip;
        double newB = (bOut * sOut + bIn) * recip;
        output.Get<R>()[i] = newR;
        output.Get<G>()[i] = newG;
        output.Get<B>()[i] = newB;
        output.Get<W>()[i] = totalSample;
    };

    size_t loopSize = output.Size() / SIMD_WIDTH;
    size_t residual = output.Size() % SIMD_WIDTH;
    for(size_t i = 0; i < loopSize; i++)
    {
        using enum MRay::HostArch;
        if constexpr(MRay::MRAY_HOST_ARCH == MRAY_AVX512)
            Iteration_AVX512(i);
        else if constexpr(MRay::MRAY_HOST_ARCH == MRAY_AVX2)
            Iteration_AVX2(i);
        else
            Iteration_Common(i);
    }

    // Calculate the rest via scalar ops
    // TODO: We can decay to 8/4/2/1 etc but is it worth it?
    size_t offset = loopSize * SIMD_WIDTH;
    for(size_t i = offset; i < offset + residual; i++)
    {
        Iteration_Common(i);
    }
}

BS::multi_future<void>
Accum::AccumulateImage(RGBWeightSpan<double> output,
                       BS::thread_pool& threadPool,
                       const RenderImageSection& rIS,
                       const RenderBufferInfo& rBI)
{
    const Float* rStart = reinterpret_cast<const Float*>(rBI.data + rIS.pixStartOffsets[R]);
    const Float* gStart = reinterpret_cast<const Float*>(rBI.data + rIS.pixStartOffsets[G]);
    const Float* bStart = reinterpret_cast<const Float*>(rBI.data + rIS.pixStartOffsets[B]);
    const Float* wStart = reinterpret_cast<const Float*>(rBI.data + rIS.weightStartOffset);

    uint32_t scanlineWidth = rIS.pixelMax[0] - rIS.pixelMin[0];
    uint32_t scanlineCount = rIS.pixelMax[1] - rIS.pixelMin[1];
    //
    auto wait = threadPool.submit_blocks(0u, scanlineCount,
    [&](uint32_t start, uint32_t end) -> void
    {
        for(uint32_t i = start; i < end; i++)
        {
            uint32_t offsetOut = i * rBI.resolution[0] + rIS.pixelMax[0];
            auto scanlineOut = RGBWeightSpan<double>
            (
                output.Get<R>().subspan(offsetOut, scanlineWidth),
                output.Get<G>().subspan(offsetOut, scanlineWidth),
                output.Get<B>().subspan(offsetOut, scanlineWidth),
                output.Get<W>().subspan(offsetOut, scanlineWidth)
            );
            //
            uint32_t offsetIn = i * scanlineWidth;
            auto scanlineIn = RGBWeightSpan<const Float>
            (
                Span(rStart + offsetIn, scanlineWidth),
                Span(gStart + offsetIn, scanlineWidth),
                Span(bStart + offsetIn, scanlineWidth),
                Span(wStart + offsetIn, scanlineWidth)
            );
            //
            AccumulateScanline(scanlineOut, scanlineIn);
        }
    });
    return wait;
}

namespace MRayCLI::RunNames
{
    using namespace std::literals;
    static constexpr auto Name = "run"sv;
    static constexpr auto Description = "Directly runs a given scene file without GUI"sv;
};

bool RunCommand::EventLoop()
{
    auto size = GetTerminalSize();
    timer.Split();

    EyeAnim::SimpleProgressBar p;

    p.Display(Float(0.5), timer.ElapsedIntMS(), "Test");
    return false;
}

MRayError RunCommand::Invoke()
{
    // Transfer queue, responsible for communication between main thread
    // (window render thread) and tracer thread
    static constexpr size_t CommandBufferSize = 8;
    static_assert(CommandBufferSize >= 3,
                  "Command buffer should at least have a size of two. "
                  "We issue two event before starting the tracer.");
    TransferQueue transferQueue(CommandBufferSize, CommandBufferSize,
                                [](){});

    BS::thread_pool threadPool(threadCount);

    // Get the tracer dll
    TracerThread tracerThread(transferQueue, threadPool);
    MRayError e = tracerThread.MTInitialize(tracerConfigFile);
    if(e) return e;

    // Reset the thread pool and initialize the threads with GPU specific
    // initialization routine, also change the name of the threads.
    // We need to do this somewhere here, if we do it on tracer side
    // due to passing between dll boundaries, it crash on destruction.
    threadPool.reset(threadCount, [&tracerThread]()
    {
        auto GPUInit = tracerThread.GetThreadInitFunction();
        GPUInit();
    });
    std::vector<std::thread::native_handle_type> handles;
    handles = threadPool.get_native_handles();
    for(size_t i = 0; i < handles.size(); i++)
    {
        using namespace std::string_literals;
        std::string name = "WorkerThread_"s + std::to_string(i);
        RenameThread(handles[i], name);
    }

    // The "timeline semaphore" (CPU emulation)
    // This will be used to synchronize between MRay and Run
    TimelineSemaphore sem(0);

    // Set resolution
    Vector2ui resolution(imgRes[0], imgRes[1]);
    tracerThread.SetInitialResolution(resolution,
                                      Vector2ui::Zero(),
                                      resolution);
    // TODO: Cleanup this API (why SetInitResolution is a function
    // but these are queue events)
    MRAY_LOG("[Run]: Configuring Tracer via initial render config...");
    transferQueue.GetVisorView().Enqueue(VisorAction
    (
        std::in_place_index<VisorAction::KICKSTART_RENDER>,
        renderConfigFile
    ));

    // Launch the renderer
    transferQueue.GetVisorView().Enqueue(VisorAction
    (
        std::in_place_index<VisorAction::START_STOP_RENDER>,
        true
    ));
    transferQueue.GetVisorView().Enqueue(VisorAction
    (
        std::in_place_index<VisorAction::SEND_SYNC_SEMAPHORE>,
        SemaphoreInfo{&sem, Accum::RenderBufferAlignment}
    ));

    // Finally start the tracer
    //tracerThread.Start("TracerThread");

    // Do the loop
    timer.Start();
    for(bool isTerminated = false; !isTerminated;)
    {
        isTerminated = EventLoop();
    }

    // Order is important here
    // First wait the thread pool
    threadPool.wait();
    // Destroy the transfer queue
    // So that the tracer can drop from queue wait
    transferQueue.Terminate();
    // Invalidate the semaphore,
    // If tracer waits to issue next image section
    // it can terminate
    sem.Invalidate();
    // First stop the tracer, since tracer commands
    // submit glfw "empty event" to trigger visor rendering
    tracerThread.Stop();
    // All Done!
    return e;
}

CLI::App* RunCommand::GenApp(CLI::App& mainApp)
{
    using namespace MRayCLI::RunNames;
    CLI::App* converter = mainApp.add_subcommand(std::string(Name),
                                                 std::string(Description));

    // Input
    // Dummy add visor config here, user may just change visor command
    // to run command, we should provide that functionality.
    converter->add_option("--visorConf, --vConf"s, visorConfString,
                          "Visor configuration file."s);

    converter->add_option("--tracerConf, --tConf"s, tracerConfigFile,
                          "Tracer configuration file, mainly specifies the "
                          "tracer dll name to be loaded."s)
        ->check(CLI::ExistingFile)
        ->required()
        ->expected(1);

    converter->add_option("--threads, -t"s, threadCount,
                          "Thread pool's thread count."s)
        ->expected(1);

    converter->add_option("--scene, -s"s, sceneFile,
                          "Scene to render."s)
        ->check(CLI::ExistingFile)
        ->required()
        ->expected(1);


    converter->add_option("--renderConf, --rConf"s, renderConfigFile,
                          "Renderer to be launched."s)
        ->check(CLI::ExistingFile)
        ->required()
        ->expected(1);

    // TODO: Change this to be a region maybe?
    converter->add_option("--resolution, -r"s, imgRes,
                          "Initial renderer's resolution. "
                          "Requires a renderer to be set."s)
        ->check(CLI::Number)
        ->delimiter('x')
        ->required()
        ->expected(1);

    return converter;
}

CommandI& RunCommand::Instance()
{
    static RunCommand c = {};
    return c;
}