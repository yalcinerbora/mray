#include "RunCommand.h"
#include "Core/Profiling.h"
#include "TracerThread.h"

#include "Core/Timer.h"
#include "Core/MRayDescriptions.h"
#include "Core/Log.h"
#include "Core/ThreadPool.h"

#include "Common/RenderImageStructs.h"
#include "Common/TransferQueue.h"

#include "ImageLoader/EntryPoint.h"

#include <CLI/CLI.hpp>
#include <string_view>
#include <barrier>
#include <immintrin.h>

// Kinda hacky way but w/e.
// fmt does not have a easy way to accept repeating char patterns
// unless it is string etc. (You probably can do it, but we have at most 128 * 2
// char buffer so it is fine, did not bother implementing some internals of fmt)
//
template<char C, uint32_t MAX_SIZE>
struct RepeatingChar
{
    private:
    static constexpr std::array<char, MAX_SIZE> Allocate()
    {
        std::array<char, MAX_SIZE> result;
        result.fill(C);
        return result;
    }
    static constexpr std::array CharBuffer = Allocate();
    static constexpr std::string_view CharSV = std::string_view(CharBuffer.data(), MAX_SIZE);
    //
    public:
    static constexpr std::string_view StringView(uint32_t length)
    {
        return CharSV.substr(0, std::min(length, MAX_SIZE));
    };
};

namespace EyeAnim
{
    // Instead of pulling std::chorno_literals to global space (it is a single
    // translation unit but w/e), using constructors
    using DurationMS = std::chrono::milliseconds;
    using LegolasLookupElem = Pair<std::string_view, DurationMS>;

    static constexpr auto AragornLine = "Legolas! What do your elf eyes see?";
    static constexpr auto LegolasLine = "Uruks turn north-east, they're taking the hobbits to Isengard!";

    static constexpr auto AnimDurationLong = DurationMS(850);
    static constexpr auto AnimDurationShort = DurationMS(450);
    static constexpr auto AnimDurationCommon = DurationMS(50);
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
            offset += uint64_t(keyFrame.second.count());
        }
        result.back() = offset;
        return result;
    }
    static constexpr std::array LegolasAnimDurations = ScanLegolasAnimDurations();

    class SimpleProgressBar
    {
        private:

        static constexpr uint32_t MAX_WIDTH = 128;
        using EqualsBuffer = RepeatingChar<'=', MAX_WIDTH >;
        using SpaceBuffer  = RepeatingChar<' ', MAX_WIDTH >;

        public:
        void Display(Float ratio, uint64_t timeMS, std::string_view postfix);
    };
}

void EyeAnim::SimpleProgressBar::Display(Float ratio, uint64_t timeMS,
                                         std::string_view postfix)
{
    // TODO: Terminal sometimes fails when fast minimization occurs
    // (Residual characters appear at the next line).
    // Investigate.
    //
    // There is a race condition that hard to synchronize
    // since we can't properly lock the terminal resize event when this function runs.
    // (Probably there is a way but I did not investigate fully)
    static constexpr auto FMT_STR = fmt::string_view("\033[2K({:s}) [{:s}{:s}] {:s}\r");

    uint64_t localTime = timeMS % LegolasAnimDurations.back();
    // TODO: "find_if" probably better for perf but did not bother
    auto loc = std::upper_bound(LegolasAnimDurations.cbegin(),
                                LegolasAnimDurations.cend(), localTime);
    assert(loc != LegolasAnimDurations.end());
    std::ptrdiff_t animIndex = std::distance(LegolasAnimDurations.begin(), loc) - 1;
    std::string_view animSprite = LegolasAnimSheet[static_cast<uint32_t>(animIndex)].first;

    // We query this everytime for adjusting the size
    auto terminalSize = GetTerminalSize();
    // Do not bother with progress bar, just print eye and postfix
    if(terminalSize[0] <= postfix.size())
    {
        fmt::print("\033[2K({:s}) {:s}\r", animSprite, postfix);
        std::fflush(stdout);
        return;
    }

    uint32_t leftover = static_cast<uint32_t>(terminalSize[0] - postfix.size());
    leftover -= static_cast<uint32_t>(LegolasAnimSheet.front().first.size());
    leftover -= 20; // Arbitrary 20 character padding
    leftover = Math::Min(leftover, MAX_WIDTH);
    uint32_t eqCount = static_cast<uint32_t>(Math::Round(Float(leftover) * ratio));
    uint32_t spaceCount = leftover - eqCount;

    fmt::print(FMT_STR,
               animSprite,
               EqualsBuffer::StringView(eqCount),
               SpaceBuffer::StringView(spaceCount),
               postfix);
    std::fflush(stdout);
}

namespace Accum
{
    static constexpr size_t RenderBufferAlignment = 4096;
    static_assert(RenderBufferAlignment >= MemAlloc::DefaultSystemAlignment(),
                  "RenderBufferAlignment does not cover "
                  "\"MemAlloc::DefaultSystemAlignment()\"!");
    template<std::floating_point T>
    using RGBWeightSpan = SoASpan<T, T, T, T>;

    static constexpr uint32_t R = 0;
    static constexpr uint32_t G = 1;
    static constexpr uint32_t B = 2;
    static constexpr uint32_t W = 3;

    template<class F>
    requires(std::is_same_v<F, float>)
    std::array<__m256d, 4>
    LoadAVX2(size_t i,
             const F* rInPtrA,
             const F* gInPtrA,
             const F* bInPtrA,
             const F* wInPtrA);

    template<class F>
    requires(std::is_same_v<F, double>)
    std::array<__m256d, 4>
    LoadAVX2(size_t i,
             const F* rInPtrA,
             const F* gInPtrA,
             const F* bInPtrA,
             const F* wInPtrA);

    template<class F>
    requires(std::is_same_v<F, float>)
    std::array<__m512d, 4>
    LoadAVX512(size_t i,
               const F* rInPtrA,
               const F* gInPtrA,
               const F* bInPtrA,
               const F* wInPtrA);

    template<class F>
    requires(std::is_same_v<F, double>)
    std::array<__m512d, 4>
    LoadAVX512(size_t i,
               const F* rInPtrA,
               const F* gInPtrA,
               const F* bInPtrA,
               const F* wInPtrA);


    void AccumulateScanline(RGBWeightSpan<double> output,
                            RGBWeightSpan<const Float> input);
    //
    void AccumulatePortionBulk(double* MRAY_RESTRICT rOutPtr,
                               double* MRAY_RESTRICT gOutPtr,
                               double* MRAY_RESTRICT bOutPtr,
                               double* MRAY_RESTRICT wOutPtr,
                               const Float* MRAY_RESTRICT rInPtr,
                               const Float* MRAY_RESTRICT gInPtr,
                               const Float* MRAY_RESTRICT bInPtr,
                               const Float* MRAY_RESTRICT wInPtr,
                               size_t outputSize);

    MultiFuture<void>
    AccumulateImage(RGBWeightSpan<double> output,
                    TimelineSemaphore& sem,
                    ThreadPool& threadPool,
                    const RenderImageSection&,
                    const RenderBufferInfo&);
}

template<class F>
requires(std::is_same_v<F, float>)
std::array<__m256d, 4>
Accum::LoadAVX2(size_t i,
                const F* rInPtrA,
                const F* gInPtrA,
                const F* bInPtrA,
                const F* wInPtrA)
{
    static constexpr size_t SIMD_WIDTH = MRay::HostArchSIMDWidth<double>();

    __m256d rIn, gIn, bIn, sIn;
    rIn = _mm256_cvtps_pd(_mm_load_ps(rInPtrA + i * SIMD_WIDTH));
    gIn = _mm256_cvtps_pd(_mm_load_ps(gInPtrA + i * SIMD_WIDTH));
    bIn = _mm256_cvtps_pd(_mm_load_ps(bInPtrA + i * SIMD_WIDTH));
    sIn = _mm256_cvtps_pd(_mm_load_ps(wInPtrA + i * SIMD_WIDTH));

    return {rIn, gIn, bIn, sIn};
}

template<class F>
requires(std::is_same_v<F, double>)
std::array<__m256d, 4>
Accum::LoadAVX2(size_t i,
                const F* rInPtrA,
                const F* gInPtrA,
                const F* bInPtrA,
                const F* wInPtrA)
{
    static constexpr size_t SIMD_WIDTH = MRay::HostArchSIMDWidth<double>();
    __m256d rIn, gIn, bIn, sIn;
    rIn = _mm256_load_pd(rInPtrA + i * SIMD_WIDTH);
    gIn = _mm256_load_pd(gInPtrA + i * SIMD_WIDTH);
    bIn = _mm256_load_pd(bInPtrA + i * SIMD_WIDTH);
    sIn = _mm256_load_pd(wInPtrA + i * SIMD_WIDTH);

    return {rIn, gIn, bIn, sIn};
}

template<class F>
requires(std::is_same_v<F, float>)
std::array<__m512d, 4>
Accum::LoadAVX512(size_t i,
                  const F* rInPtrA,
                  const F* gInPtrA,
                  const F* bInPtrA,
                  const F* wInPtrA)
{
    static constexpr size_t SIMD_WIDTH = MRay::HostArchSIMDWidth<double>();

    __m512d rIn, gIn, bIn, sIn;
    rIn = _mm512_cvtps_pd(_mm256_load_ps(rInPtrA + i * SIMD_WIDTH));
    gIn = _mm512_cvtps_pd(_mm256_load_ps(gInPtrA + i * SIMD_WIDTH));
    bIn = _mm512_cvtps_pd(_mm256_load_ps(bInPtrA + i * SIMD_WIDTH));
    sIn = _mm512_cvtps_pd(_mm256_load_ps(wInPtrA + i * SIMD_WIDTH));

    return {rIn, gIn, bIn, sIn};
}

template<class F>
requires(std::is_same_v<F, double>)
std::array<__m512d, 4>
Accum::LoadAVX512(size_t i,
                  const F* rInPtrA,
                  const F* gInPtrA,
                  const F* bInPtrA,
                  const F* wInPtrA)
{
    static constexpr size_t SIMD_WIDTH = MRay::HostArchSIMDWidth<double>();
    __m512d rIn, gIn, bIn, sIn;
    rIn = _mm512_load_pd(rInPtrA + i * SIMD_WIDTH);
    gIn = _mm512_load_pd(gInPtrA + i * SIMD_WIDTH);
    bIn = _mm512_load_pd(bInPtrA + i * SIMD_WIDTH);
    sIn = _mm512_load_pd(wInPtrA + i * SIMD_WIDTH);

    return {rIn, gIn, bIn, sIn};
}

void Accum::AccumulateScanline(RGBWeightSpan<double> output,
                               RGBWeightSpan<const Float> input)
{

    assert(output.Size() == input.Size());
    constexpr uint32_t SIMD_WIDTH = MRay::HostArchSIMDWidth<double>();

    auto Iteration_AVX512 = [&](uint32_t i)
    {
        __m512d rOut = _mm512_loadu_pd(output.Get<R>().data() + i * SIMD_WIDTH);
        __m512d gOut = _mm512_loadu_pd(output.Get<G>().data() + i * SIMD_WIDTH);
        __m512d bOut = _mm512_loadu_pd(output.Get<B>().data() + i * SIMD_WIDTH);
        __m512d sOut = _mm512_loadu_pd(output.Get<W>().data() + i * SIMD_WIDTH);
        //
        auto [rIn, gIn, bIn, sIn] = LoadAVX512(i,
                                               input.Get<R>().data(),
                                               input.Get<G>().data(),
                                               input.Get<B>().data(),
                                               input.Get<W>().data());
        //
        __m512d sTotal = _mm512_add_pd(sOut, sIn);
        __m512d newR = _mm512_fmadd_pd(rOut, sOut, rIn);
        __m512d newG = _mm512_fmadd_pd(gOut, sOut, gIn);
        __m512d newB = _mm512_fmadd_pd(bOut, sOut, bIn);
        //
        __mmask8 notZero = _mm512_cmp_pd_mask(sTotal, _mm512_setzero_pd(),
                                              _CMP_NEQ_UQ);
        static constexpr bool HAS_RECIP = false;
        if constexpr(HAS_RECIP)
        {
            __m512d sRecip = _mm512_rcp14_pd(sTotal);
            newR = _mm512_maskz_mul_pd(notZero, newR, sRecip);
            newG = _mm512_maskz_mul_pd(notZero, newG, sRecip);
            newB = _mm512_maskz_mul_pd(notZero, newB, sRecip);
        }
        else
        {
            newR = _mm512_maskz_div_pd(notZero, newR, sTotal);
            newG = _mm512_maskz_div_pd(notZero, newG, sTotal);
            newB = _mm512_maskz_div_pd(notZero, newB, sTotal);
        }
        //
        _mm512_storeu_pd(output.Get<R>().data() + i * SIMD_WIDTH, newR);
        _mm512_storeu_pd(output.Get<G>().data() + i * SIMD_WIDTH, newG);
        _mm512_storeu_pd(output.Get<B>().data() + i * SIMD_WIDTH, newB);
        _mm512_storeu_pd(output.Get<W>().data() + i * SIMD_WIDTH, sTotal);
    };

    auto Iteration_AVX2 = [&](uint32_t i)
    {
        __m256d rOut = _mm256_loadu_pd(output.Get<R>().data() + i * SIMD_WIDTH);
        __m256d gOut = _mm256_loadu_pd(output.Get<G>().data() + i * SIMD_WIDTH);
        __m256d bOut = _mm256_loadu_pd(output.Get<B>().data() + i * SIMD_WIDTH);
        __m256d sOut = _mm256_loadu_pd(output.Get<W>().data() + i * SIMD_WIDTH);
        //
        auto [rIn, gIn, bIn, sIn] = LoadAVX2(i,
                                             input.Get<R>().data(),
                                             input.Get<G>().data(),
                                             input.Get<B>().data(),
                                             input.Get<W>().data());
        //
        __m256d sTotal = _mm256_add_pd(sOut, sIn);
        //
        __m256d newR = _mm256_fmadd_pd(rOut, sOut, rIn);
        __m256d newG = _mm256_fmadd_pd(gOut, sOut, gIn);
        __m256d newB = _mm256_fmadd_pd(bOut, sOut, bIn);
        //
        __m256d notZero = _mm256_cmp_pd(sTotal, _mm256_setzero_pd(),
                                        _CMP_NEQ_UQ);
        static constexpr bool HAS_RECIP = false;
        if constexpr(HAS_RECIP)
        {
            __m256d sRecip = _mm256_rcp14_pd(sTotal);
            newR = _mm256_mul_pd(newR, sRecip);
            newG = _mm256_mul_pd(newG, sRecip);
            newB = _mm256_mul_pd(newB, sRecip);
        }
        else
        {
            newR = _mm256_div_pd(newR, sTotal);
            newG = _mm256_div_pd(newG, sTotal);
            newB = _mm256_div_pd(newB, sTotal);
        }
        newR = _mm256_blendv_pd(_mm256_setzero_pd(), newR, notZero);
        newG = _mm256_blendv_pd(_mm256_setzero_pd(), newG, notZero);
        newB = _mm256_blendv_pd(_mm256_setzero_pd(), newB, notZero);

        //
        _mm256_storeu_pd(output.Get<R>().data() + i * SIMD_WIDTH, newR);
        _mm256_storeu_pd(output.Get<G>().data() + i * SIMD_WIDTH, newG);
        _mm256_storeu_pd(output.Get<B>().data() + i * SIMD_WIDTH, newB);
        _mm256_storeu_pd(output.Get<W>().data() + i * SIMD_WIDTH, sTotal);
    };

    auto Iteration_Common = [&](uint32_t i)
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
        double newR = (totalSample == 0.0) ? 0.0 : (rOut * sOut + double(rIn)) * recip;
        double newG = (totalSample == 0.0) ? 0.0 : (gOut * sOut + double(gIn)) * recip;
        double newB = (totalSample == 0.0) ? 0.0 : (bOut * sOut + double(bIn)) * recip;
        output.Get<R>()[i] = newR;
        output.Get<G>()[i] = newG;
        output.Get<B>()[i] = newB;
        output.Get<W>()[i] = totalSample;
    };

    // TODO: With scanline, we cannot guarantee memory
    // alignment, (due to image width is not aligned
    // with SIMD width)
    // So the code below should be slower.
    uint32_t loopSize = uint32_t(output.Size() / SIMD_WIDTH);
    uint32_t residual = uint32_t(output.Size() % SIMD_WIDTH);
    for(uint32_t i = 0; i < loopSize; i++)
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
    uint32_t offset = loopSize * SIMD_WIDTH;
    for(uint32_t i = offset; i < offset + residual; i++)
    {
       Iteration_Common(i);
    }
}

void Accum::AccumulatePortionBulk(double* MRAY_RESTRICT rOutPtr,
                                  double* MRAY_RESTRICT gOutPtr,
                                  double* MRAY_RESTRICT bOutPtr,
                                  double* MRAY_RESTRICT wOutPtr,
                                  const Float* MRAY_RESTRICT rInPtr,
                                  const Float* MRAY_RESTRICT gInPtr,
                                  const Float* MRAY_RESTRICT bInPtr,
                                  const Float* MRAY_RESTRICT wInPtr,
                                  size_t size)
{
    static constexpr size_t SIMD_WIDTH = MRay::HostArchSIMDWidth<double>();

    static constexpr auto DSA = MemAlloc::DefaultSystemAlignment();
    double* MRAY_RESTRICT rOutPtrA = std::assume_aligned<DSA>(rOutPtr);
    double* MRAY_RESTRICT gOutPtrA = std::assume_aligned<DSA>(gOutPtr);
    double* MRAY_RESTRICT bOutPtrA = std::assume_aligned<DSA>(bOutPtr);
    double* MRAY_RESTRICT wOutPtrA = std::assume_aligned<DSA>(wOutPtr);
    const Float* MRAY_RESTRICT rInPtrA = std::assume_aligned<DSA>(rInPtr);
    const Float* MRAY_RESTRICT gInPtrA = std::assume_aligned<DSA>(gInPtr);
    const Float* MRAY_RESTRICT bInPtrA = std::assume_aligned<DSA>(bInPtr);
    const Float* MRAY_RESTRICT wInPtrA = std::assume_aligned<DSA>(wInPtr);

    auto Iteration_AVX512 = [&](size_t i)
    {
        __m512d rOut = _mm512_load_pd(rOutPtrA + i * SIMD_WIDTH);
        __m512d gOut = _mm512_load_pd(gOutPtrA + i * SIMD_WIDTH);
        __m512d bOut = _mm512_load_pd(bOutPtrA + i * SIMD_WIDTH);
        __m512d sOut = _mm512_load_pd(wOutPtrA + i * SIMD_WIDTH);
        //
        auto [rIn, gIn, bIn, sIn] = LoadAVX512(i,
                                               rInPtrA, gInPtrA,
                                               bInPtrA, wInPtrA);
        //
        __m512d sTotal = _mm512_add_pd(sOut, sIn);
        __m512d newR = _mm512_fmadd_pd(rOut, sOut, rIn);
        __m512d newG = _mm512_fmadd_pd(gOut, sOut, gIn);
        __m512d newB = _mm512_fmadd_pd(bOut, sOut, bIn);
        //
        __mmask8 notZero = _mm512_cmp_pd_mask(sTotal, _mm512_setzero_pd(),
                                              _CMP_NEQ_UQ);
        static constexpr bool HAS_RECIP = false;
        if constexpr(HAS_RECIP)
        {
            __m512d sRecip = _mm512_rcp14_pd(sTotal);
            newR = _mm512_maskz_mul_pd(notZero, newR, sRecip);
            newG = _mm512_maskz_mul_pd(notZero, newG, sRecip);
            newB = _mm512_maskz_mul_pd(notZero, newB, sRecip);
        }
        else
        {
            newR = _mm512_maskz_div_pd(notZero, newR, sTotal);
            newG = _mm512_maskz_div_pd(notZero, newG, sTotal);
            newB = _mm512_maskz_div_pd(notZero, newB, sTotal);
        }
        //
        _mm512_store_pd(rOutPtrA + i * SIMD_WIDTH, newR);
        _mm512_store_pd(gOutPtrA + i * SIMD_WIDTH, newG);
        _mm512_store_pd(bOutPtrA + i * SIMD_WIDTH, newB);
        _mm512_store_pd(wOutPtrA + i * SIMD_WIDTH, sTotal);
    };

    auto Iteration_AVX2 = [&](size_t i)
    {
        __m256d rOut = _mm256_load_pd(rOutPtrA + i * SIMD_WIDTH);
        __m256d gOut = _mm256_load_pd(gOutPtrA + i * SIMD_WIDTH);
        __m256d bOut = _mm256_load_pd(bOutPtrA + i * SIMD_WIDTH);
        __m256d sOut = _mm256_load_pd(wOutPtrA + i * SIMD_WIDTH);
        //
        auto [rIn, gIn, bIn, sIn] = LoadAVX2(i,
                                             rInPtrA, gInPtrA,
                                             bInPtrA, wInPtrA);
        //
        __m256d sTotal = _mm256_add_pd(sOut, sIn);
        //
        __m256d newR = _mm256_fmadd_pd(rOut, sOut, rIn);
        __m256d newG = _mm256_fmadd_pd(gOut, sOut, gIn);
        __m256d newB = _mm256_fmadd_pd(bOut, sOut, bIn);
        //
        __m256d notZero = _mm256_cmp_pd(sTotal, _mm256_setzero_pd(),
                                        _CMP_NEQ_UQ);
        static constexpr bool HAS_RECIP = false;
        if constexpr(HAS_RECIP)
        {
            __m256d sRecip = _mm256_rcp14_pd(sTotal);
            newR = _mm256_mul_pd(newR, sRecip);
            newG = _mm256_mul_pd(newG, sRecip);
            newB = _mm256_mul_pd(newB, sRecip);
        }
        else
        {
            newR = _mm256_div_pd(newR, sTotal);
            newG = _mm256_div_pd(newG, sTotal);
            newB = _mm256_div_pd(newB, sTotal);
        }
        newR = _mm256_blendv_pd(_mm256_setzero_pd(), newR, notZero);
        newG = _mm256_blendv_pd(_mm256_setzero_pd(), newG, notZero);
        newB = _mm256_blendv_pd(_mm256_setzero_pd(), newB, notZero);
        //
        _mm256_store_pd(rOutPtrA + i * SIMD_WIDTH, newR);
        _mm256_store_pd(gOutPtrA + i * SIMD_WIDTH, newG);
        _mm256_store_pd(bOutPtrA + i * SIMD_WIDTH, newB);
        _mm256_store_pd(wOutPtrA + i * SIMD_WIDTH, sTotal);
    };

    auto Iteration_Common = [&](size_t i)
    {
        double rOut = rOutPtr[i];
        double gOut = gOutPtr[i];
        double bOut = bOutPtr[i];
        double sOut = wOutPtr[i];
        //
        Float rIn = rInPtr[i];
        Float gIn = gInPtr[i];
        Float bIn = bInPtr[i];
        Float sIn = wInPtr[i];
        //
        double totalSample = sOut + double(sIn);
        double recip = double(1) / totalSample;
        double newR = (totalSample == 0.0) ? 0.0 : (rOut * sOut + double(rIn)) * recip;
        double newG = (totalSample == 0.0) ? 0.0 : (gOut * sOut + double(gIn)) * recip;
        double newB = (totalSample == 0.0) ? 0.0 : (bOut * sOut + double(bIn)) * recip;
        rOutPtr[i] = newR;
        gOutPtr[i] = newG;
        bOutPtr[i] = newB;
        wOutPtr[i] = totalSample;
    };

    size_t loopSize = size / SIMD_WIDTH;
    size_t residual = size % SIMD_WIDTH;
    for(size_t i = 0; i < loopSize;)
    {
        using enum MRay::HostArch;
        if constexpr(MRay::MRAY_HOST_ARCH == MRAY_AVX512)
        {
            if(i++ < loopSize) [[likely]] Iteration_AVX512(i - 1);
            if(i++ < loopSize) [[likely]] Iteration_AVX512(i - 1);
        }
        else if constexpr(MRay::MRAY_HOST_ARCH == MRAY_AVX2)
        {
            if(i++ < loopSize) [[likely]] Iteration_AVX2(i - 1);
            if(i++ < loopSize) [[likely]] Iteration_AVX2(i - 1);
            if(i++ < loopSize) [[likely]] Iteration_AVX2(i - 1);
            if(i++ < loopSize) [[likely]] Iteration_AVX2(i - 1);
        }
        else
        {
            if(i++ < loopSize) [[likely]] Iteration_Common(i - 1);
            if(i++ < loopSize) [[likely]] Iteration_Common(i - 1);
            if(i++ < loopSize) [[likely]] Iteration_Common(i - 1);
            if(i++ < loopSize) [[likely]] Iteration_Common(i - 1);
            if(i++ < loopSize) [[likely]] Iteration_Common(i - 1);
            if(i++ < loopSize) [[likely]] Iteration_Common(i - 1);
            if(i++ < loopSize) [[likely]] Iteration_Common(i - 1);
            if(i++ < loopSize) [[likely]] Iteration_Common(i - 1);
        }
    }

    // Early skip, most of the images should be
    // multiple of 2, 4 or 8
    if(residual == 0) [[likely]] return;

    size_t offset = loopSize * SIMD_WIDTH;
    if(residual > 0) Iteration_Common(offset + 0);
    if(residual > 1) Iteration_Common(offset + 1);
    if(residual > 2) Iteration_Common(offset + 2);
    if(residual > 3) Iteration_Common(offset + 3);
    if(residual > 4) Iteration_Common(offset + 4);
    if(residual > 5) Iteration_Common(offset + 5);
    if(residual > 6) Iteration_Common(offset + 6);
    if(residual > 7) Iteration_Common(offset + 7);
    static_assert(SIMD_WIDTH <= 8, "Expand this loop unroll");
}

MultiFuture<void>
Accum::AccumulateImage(RGBWeightSpan<double> output,
                       TimelineSemaphore& sem,
                       ThreadPool& threadPool,
                       const RenderImageSection& rIS,
                       const RenderBufferInfo& rBI)
{
    static const auto issueAnnot = ProfilerAnnotation("AccumPortion-Issue");
    const auto issueScope = issueAnnot.AnnotateScope();

    const Float* rStart = reinterpret_cast<const Float*>(rBI.data + rIS.pixStartOffsets[R]);
    const Float* gStart = reinterpret_cast<const Float*>(rBI.data + rIS.pixStartOffsets[G]);
    const Float* bStart = reinterpret_cast<const Float*>(rBI.data + rIS.pixStartOffsets[B]);
    const Float* wStart = reinterpret_cast<const Float*>(rBI.data + rIS.weightStartOffset);

    // TODO: Move this from shared ptr, at least allocate once instead of every iteration
    auto BarrierFunc = [&]() noexcept { sem.Release();};
    using Barrier = std::barrier<decltype(BarrierFunc)>;
    uint32_t threadCount = threadPool.ThreadCount();
    auto barrier = std::make_shared<Barrier>(threadCount, BarrierFunc);
    if(rBI.resolution == rIS.pixelMax - rIS.pixelMin)
    {
        constexpr size_t SIMD_WIDTH = MRay::HostArchSIMDWidth<double>();
        size_t totalPixels = rBI.resolution.Multiply();
        size_t bulkCount = Math::DivideUp(totalPixels, SIMD_WIDTH);

        auto WorkFuncBulk = [=](uint32_t start, uint32_t end) -> void
        {
            static const auto _ = ProfilerAnnotation("AccumPortion-Bulk");
            const auto scope = _.AnnotateScope();

            size_t offsetInOut = start * SIMD_WIDTH;
            size_t bulkEnd = std::min(end * SIMD_WIDTH, totalPixels);
            size_t bulkWidth = bulkEnd - offsetInOut;

            AccumulatePortionBulk(output.Get<R>().subspan(offsetInOut, bulkWidth).data(),
                                  output.Get<G>().subspan(offsetInOut, bulkWidth).data(),
                                  output.Get<B>().subspan(offsetInOut, bulkWidth).data(),
                                  output.Get<W>().subspan(offsetInOut, bulkWidth).data(),
                                  rStart + offsetInOut,
                                  gStart + offsetInOut,
                                  bStart + offsetInOut,
                                  wStart + offsetInOut,
                                  bulkWidth);

            // TODO: Only arrive gives warning, but we use barrier
            // only for completion function. Check a way later.
            barrier->arrive_and_wait();
        };
        //
        return threadPool.SubmitBlocks(uint32_t(bulkCount),
                                       std::move(WorkFuncBulk), threadCount);
    }
    else
    {
        uint32_t scanlineWidth = rIS.pixelMax[0] - rIS.pixelMin[0];
        uint32_t scanlineCount = rIS.pixelMax[1] - rIS.pixelMin[1];

        auto WorkFuncScanline = [=](uint32_t start, uint32_t end) -> void
        {
            static const auto _ = ProfilerAnnotation("AccumPortion-Scanline");
            const auto scope = _.AnnotateScope();

            for(uint32_t i = start; i < end; i++)
            {
                uint32_t offsetOut = (rIS.pixelMin[1] + i) * rBI.resolution[0];
                offsetOut += rIS.pixelMin[0];
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
                    Span<const Float>(rStart + offsetIn, scanlineWidth),
                    Span<const Float>(gStart + offsetIn, scanlineWidth),
                    Span<const Float>(bStart + offsetIn, scanlineWidth),
                    Span<const Float>(wStart + offsetIn, scanlineWidth)
                );
                //
                AccumulateScanline(scanlineOut, scanlineIn);
            }
            // TODO: Only arrive gives warning, but we use barrier
            // only for completion function. Check a way later.
            barrier->arrive_and_wait();
        };
        //
        return threadPool.SubmitBlocks(scanlineCount,
                                       std::move(WorkFuncScanline), threadCount);
    }
}

namespace MRayCLI::RunNames
{
    using namespace std::literals;
    static constexpr auto Name = "run"sv;
    static constexpr auto Description = "Directly runs a given scene file without GUI"sv;
};

bool RunCommand::EventLoop(TransferQueue& transferQueue,
                           ThreadPool& threadPool)
{
    cmdTimer.Split();

    Optional<RenderBufferInfo>      newRenderBuffer;
    Optional<RenderImageSection>    newImageSection;
    Optional<bool>                  newClearSignal;
    Optional<RenderImageSaveInfo>   newSaveInfo;

    TracerResponse response;
    auto& vView = transferQueue.GetVisorView();
    using enum TimedDequeueResult;
    while(vView.TryDequeue(response, EyeAnim::AnimDurationCommon) == SUCCESS)
    {
        using RespType = typename TracerResponse::Type;
        RespType tp = static_cast<RespType>(response.index());
        using enum TracerResponse::Type;

        // Stop consuming commands if image section
        // related things are in the queue
        // these require to be processed.
        //
        // For other things, the latest value is enough
        // (most of these are analytics etc)
        bool stopConsuming = false;
        switch(tp)
        {
            case CAMERA_INIT_TRANSFORM:
            {
                //MRAY_LOG("[Run]   : Transform received and ignored");
                break;
            }
            case SCENE_ANALYTICS:
            {
                MRAY_LOG("[Run]   : Scene Info received");
                sceneInfo = std::get<SCENE_ANALYTICS>(response);
                break;
            }
            case TRACER_ANALYTICS:
            {
                MRAY_LOG("[Run]   : Tracer Info received");
                tracerInfo = std::get<TRACER_ANALYTICS>(response);
                break;
            }
            case RENDERER_ANALYTICS:
            {
                //MRAY_LOG("[Run]   : Render Info received");
                rendererInfo = std::get<RENDERER_ANALYTICS>(response);
                renderThroughputAverage.FeedValue(Float(rendererInfo.throughput));
                iterationTimeAverage.FeedValue(Float(rendererInfo.iterationTimeMS));

                break;
            }
            case RENDERER_OPTIONS:
            {
                MRAY_LOG("[Run]   :Render Options received and ignored");
                break; // TODO: User may change the render options during runtime
            }
            case RENDER_BUFFER_INFO:
            {
                MRAY_LOG("[Run]   : Render Buffer Info received");
                newRenderBuffer = std::get<RENDER_BUFFER_INFO>(response);
                stopConsuming = true;
                break;
            }
            case CLEAR_IMAGE_SECTION:
            {
                MRAY_LOG("[Run]   : Clear Image received");
                newClearSignal = std::get<CLEAR_IMAGE_SECTION>(response);
                stopConsuming = true;
                break;
            }
            case IMAGE_SECTION:
            {
                //MRAY_LOG("[Run]   : Image section received");
                newImageSection = std::get<IMAGE_SECTION>(response);
                stopConsuming = true;
                break;
            }
            case SAVE_AS_HDR:
            {
                //MRAY_LOG("[Run]   : Save HDR received");
                newSaveInfo = std::get<SAVE_AS_HDR>(response);
                stopConsuming = true;
                break;
            }
            case SAVE_AS_SDR:
            {
                MRAY_WARNING_LOG("[Run]   : Save SDR cannot be processed "
                                 "(No color conversion logic)");
                break;
            }
            case MEMORY_USAGE:
            {
                MRAY_LOG("[Run]   : Memory usage received");
                memUsage = std::get<MEMORY_USAGE>(response);
                break;
            }
            default: MRAY_WARNING_LOG("[Run] Unkown tracer response is ignored!"); break;
        }
        if(stopConsuming) break;
    }

    // Tracer is terminated it closed the queue.
    if(vView.IsTerminated()) return true;
    //
    if(newRenderBuffer)
    {
        if(accumulateFuture.AnyValid())
            accumulateFuture.WaitAll();

        size_t pixelCount = newRenderBuffer->resolution.Multiply();
        MemAlloc::AllocateMultiData(Tie(imageRData, imageGData,
                                        imageBData, imageSData),
                                    imageMem,
                                    {pixelCount, pixelCount,
                                    pixelCount, pixelCount});
        renderBufferInfo = newRenderBuffer.value();

        std::fill(imageRData.begin(), imageRData.end(), 0.0);
        std::fill(imageGData.begin(), imageGData.end(), 0.0);
        std::fill(imageBData.begin(), imageBData.end(), 0.0);
        std::fill(imageSData.begin(), imageSData.end(), 0.0);

        MRAY_LOG("[Run]   : Rendering Started... Resolution [{}x{}]",
                 renderBufferInfo.resolution[0],
                 renderBufferInfo.resolution[1]);

        MRAY_LOG(EyeAnim::AragornLine);
        renderTimer.Start();
        startDisplayProgressBar = true;
    }
    //
    if(newClearSignal)
    {
        std::fill(imageSData.begin(), imageSData.end(), 0.0);
    }
    //
    if(newImageSection)
    {
        renderTimer.Split();
        lastReceiveMS = renderTimer.ElapsedIntMS();

        const auto& section = newImageSection.value();
        // Tracer may abruptly terminated (crash probably),
        // so do not issue anything, return nullopt and
        // let the main render loop to terminate
        if(!syncSemaphore.Acquire(section.waitCounter))
            return true;

        // We may encounter runaway issue here, so only issue new
        // accumulation when previous one is finished
        if(accumulateFuture.AnyValid())
            accumulateFuture.WaitAll();

        Accum::RGBWeightSpan<double> out(imageRData, imageGData,
                                         imageBData, imageSData);
        accumulateFuture = Accum::AccumulateImage(out, syncSemaphore,
                                                  threadPool,
                                                  section, renderBufferInfo);
    }

    if(newSaveInfo)
    {
        const auto& saveInfo = newSaveInfo.value();
        if(accumulateFuture.AnyValid())
            accumulateFuture.WaitAll();

        MRAY_LOG(EyeAnim::LegolasLine);
        transferQueue.Terminate();
        syncSemaphore.Invalidate();

        Vector2ui res = renderBufferInfo.resolution;
        size_t pixCount = res.Multiply();
        // Convert to AoS
        Span<Vector3> rgbData;
        MemAlloc::AlignedMemory saveMem;
        MemAlloc::AllocateMultiData(Tie(rgbData), saveMem, {pixCount});
        for(uint32_t i = 0; i < pixCount; i++)
        {
            rgbData[i] = Vector3(imageRData[i],
                                 imageGData[i],
                                 imageBData[i]);
        }
        const Byte* rgbPtr = reinterpret_cast<Byte*>(rgbData.data());
        using enum MRayPixelEnum;
        auto pixType = MRayPixelTypeRT(MR_RGB_FLOAT);
        size_t paddedImageSize = pixType.PixelSize() * pixCount;
        MRayColorSpaceEnum colorSpace = renderBufferInfo.renderColorSpace;
        WriteImageParams imgInfo =
        {
            .header =
            {
                .dimensions = Vector3ui(res, 1u),
                .mipCount = 1,
                .pixelType = pixType,
                .colorSpace = Pair<Float, MRayColorSpaceEnum>
                (
                    Float(1.0),
                    colorSpace
                ),
                .readMode = MRayTextureReadMode::MR_ENUM_END
            },
            .inputType = pixType,
            .pixels = Span<const Byte>(rgbPtr, paddedImageSize)
        };

        // Load the DLL for saving
        ImageLoaderIPtr imgLoader(CreateImageLoader());
        // TODO: render time prevents overwriting, and render time
        // may be confused with frame count etc. we need to check how
        // other renderers makes their file names.
        std::string filePath = MRAY_FORMAT("{:s}_{:08.0f}wpp",
                                           saveInfo.prefix,
                                           Math::Round(saveInfo.workPerPixel));
        MRAY_LOG("Saving \"{}\"...", filePath);

        MRayError e = imgLoader->WriteImage(imgInfo, filePath,
                                            ImageType::EXR);

        // Log the error
        if(e) MRAY_ERROR_LOG("{}", e.GetError());
        else
        {
            renderTimer.Split();
            // TODO: Accuracy issue maybe when render time is long?
            double rtTotal = renderTimer.Elapsed<Millisecond>();

            FormatTimeDynamic(renderTimer);
            MRAY_LOG("Entire render took {} | {:.2f}ms wpp",
                     FormatTimeDynamic(renderTimer),
                     rtTotal / rendererInfo.wppLimit);
        }
        return true;
    }

    //
    double progressRatio = 0.0;
    std::string displaySuffix;
    if(startDisplayProgressBar)
    {
        progressRatio = rendererInfo.workPerPixel / rendererInfo.wppLimit;
        double sppLeft = rendererInfo.wppLimit - rendererInfo.workPerPixel;
        //
        uint64_t totalMS = lastReceiveMS;
        uint64_t wppCurrent = uint64_t(Math::Round(rendererInfo.workPerPixel));
        if(wppCurrent != 0)
        {
            totalMS /= wppCurrent;
            totalMS *= uint64_t(Math::Round(sppLeft));
        }
        else totalMS = std::numeric_limits<uint64_t>::max();

        //
        uint64_t totalUsedMem = memUsage + rendererInfo.usedGPUMemoryBytes;
        using MemAlloc::ConvertMemSizeToString;
        auto usedGPUMem = ConvertMemSizeToString(totalUsedMem);
        auto totalGPUMem = ConvertMemSizeToString(tracerInfo.totalGPUMemoryBytes);
        //
        // TODO: Every frame we allocate a string... Wasteful
        // Use "format_to" for both time left and the entire display prefix
        std::string timeLeft = FormatTimeDynamic(totalMS);
        displaySuffix = MRAY_FORMAT("mem [{:.1f}{:s}/{:.1f}{:s}] | wpp [{:.1f}/{:.1f}] ~left {}",
                                    usedGPUMem.first, usedGPUMem.second,
                                    totalGPUMem.first, totalGPUMem.second,
                                    rendererInfo.workPerPixel, rendererInfo.wppLimit,
                                    timeLeft);

        using namespace EyeAnim;
        SimpleProgressBar().Display(Float(progressRatio),
                                    cmdTimer.ElapsedIntMS(), displaySuffix);
    }

    return false;
}

RunCommand::RunCommand()
    : threadCount(Math::Max(1u, std::thread::hardware_concurrency() - 1))
    , rendererInfo({})
{
    // To prevent
    rendererInfo.wppLimit = 1;
}

MRayError RunCommand::Invoke()
{
    try
    {
        // Transfer queue, responsible for communication between main thread
        // (window render thread) and tracer thread
        static constexpr size_t CommandBufferSize = 8;
        static_assert(CommandBufferSize >= 4,
                      "Command buffer should at least have a size of two. "
                      "We issue two event before starting the tracer.");
        TransferQueue transferQueue(CommandBufferSize, CommandBufferSize,
                                    [](){});

        ThreadPool threadPool;

        // Get the tracer dll
        TracerThread tracerThread(transferQueue, threadPool);
        MRayError e = tracerThread.MTInitialize(tracerConfigFile);
        if(e) return e;

        // Reset the thread pool and initialize the threads with GPU specific
        // initialization routine, also change the name of the threads.
        // We need to do this somewhere here, if we do it on tracer side
        // due to passing between dll boundaries, it crash on destruction.
        threadPool.RestartThreads(threadCount, [&tracerThread](std::thread::native_handle_type handle,
                                                               uint32_t threadNumber)
        {
            RenameThread(handle, MRAY_FORMAT("{:03d}_[W]MRay", threadNumber));
            auto GPUInit = tracerThread.GetThreadInitFunction();
            GPUInit();
        });

        // Set resolution
        assert(imgRes.has_value());
        Vector2ui resolution((*imgRes)[0], (*imgRes)[1]);
        tracerThread.SetInitialResolution(resolution,
                                        Vector2ui::Zero(),
                                        resolution);
        // TODO: Cleanup this API (why SetInitResolution is a function
        // but these are queue events)
        MRAY_LOG("[Run]   : Sending sync semaphore...");
        transferQueue.GetVisorView().Enqueue(VisorAction
        (
            std::in_place_index<VisorAction::SEND_SYNC_SEMAPHORE>,
            SemaphoreInfo{&syncSemaphore, Accum::RenderBufferAlignment}
        ));
        MRAY_LOG("[Run]   : Sending initial scene...");
        transferQueue.GetVisorView().Enqueue(VisorAction
        (
            std::in_place_index<VisorAction::LOAD_SCENE>,
            sceneFile
        ));
        // Launch the renderer
        MRAY_LOG("[Run]   : Configuring Tracer via initial render config...");
        transferQueue.GetVisorView().Enqueue(VisorAction
        (
            std::in_place_index<VisorAction::KICKSTART_RENDER>,
            renderConfigFile
        ));
        transferQueue.GetVisorView().Enqueue(VisorAction
        (
            std::in_place_index<VisorAction::START_STOP_RENDER>,
            true
        ));
        // Finally start the tracer
        tracerThread.Start("TracerThread");

        // Do the loop
        cmdTimer.Start();
        for(bool isTerminated = false; !isTerminated;)
        {
            isTerminated = EventLoop(transferQueue, threadPool);
        }
        // Order is important here
        // First wait the thread pool
        accumulateFuture.futures.clear();
        threadPool.Wait();
        // Destroy the transfer queue
        // So that the tracer can drop from queue wait
        transferQueue.Terminate();
        // Invalidate the semaphore,
        // If tracer waits to issue next image section
        // it can terminate
        syncSemaphore.Invalidate();
        // First stop the tracer, since tracer commands
        // submit glfw "empty event" to trigger visor rendering
        tracerThread.Stop();
        // All Done!
        return e;
    }
    catch(const MRayError& e)
    {
        return e;
    }
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
                          "Resolution of the output image "
                          "(i.e., 1920x1080.)"s)
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