#include "RunCommand.h"
#include "GFGConverter/GFGConverter.h"
#include "Core/Timer.h"

#include <CLI/CLI.hpp>
#include <string_view>

using DurationMS = std::chrono::milliseconds;
using LegolasLookupElem = Pair<std::string_view, DurationMS>;

// Instead of pulling std::chorno_literals to global space (it is a single
// translation unit but w/e), using constructors
static constexpr auto AnimDurationLong = DurationMS(850);
static constexpr auto AnimDurationShort = DurationMS(475);
static constexpr std::array LegolasAnimSheet =
{
    LegolasLookupElem{"< 0 >\r", AnimDurationLong},
    LegolasLookupElem{"<  0>\r", AnimDurationLong},
    LegolasLookupElem{"< 0 >\r", AnimDurationLong},
    LegolasLookupElem{"< _ >\r", AnimDurationShort},
    LegolasLookupElem{"< 0 >\r", AnimDurationLong},
    LegolasLookupElem{"<0  >\r", AnimDurationLong},
    LegolasLookupElem{"< 0 >\r", AnimDurationLong},
    LegolasLookupElem{"< _ >\r", AnimDurationShort}
};

//
void AccumulateScanline(Span<Vector3d> hPixelsOut,
                        Span<double> hSamplesOut,
                        Span<const Vector3> hPixels,
                        Span<const Float> hSamples)
{
    assert(hPixelsOut.size() == hSamplesOut.size());
    assert(hSamplesOut.size() == hPixels.size());
    assert(hPixels.size() == hSamples.size());

    for(size_t i = 0; i < hPixelsOut.size(); i++)
    {
        Vector3d pixOut = hPixelsOut[i];
        double sampleOut = hSamplesOut[i];
        double totalSample = sampleOut + double(hSamples[i]);

        Vector3d newColor = pixOut * sampleOut + Vector3d(hPixels[i]);
        newColor /= Vector3d(totalSample);

        hPixelsOut[i] = newColor;
        hSamplesOut[i] = totalSample;
    }
}


void Accumulate(Span<Vector3d> dPixelsOut, Span<double> dSamplesOut,
                Span<const Vector3> dPixels, Span<const Float> dSamples,
                Vector2ui canvasSize,
                Vector2ui offset,
                Vector2ui count)
{
    static constexpr size_t PAGE_ALIGN = 4096;
    Vector3d* __restrict dPixelsOutPtr = std::assume_aligned<PAGE_ALIGN>(dPixelsOut.data());
    double* __restrict dSamplesOutPtr = std::assume_aligned<PAGE_ALIGN>(dSamplesOut.data());
    const Vector3* __restrict dPixelsPtr = std::assume_aligned<PAGE_ALIGN>(dPixels.data());
    const Float* __restrict dSamplesPtr = std::assume_aligned<PAGE_ALIGN>(dSamples.data());

    for(size_t j = 0; j < count[1]; j++)
    for(size_t i = 0; i < count[0]; i++)
    {
        size_t inI = j * count[0] + i;

        Vector2ui pixel = Vector2ui(i, j);
        pixel += offset;
        size_t outI = pixel[1] * canvasSize[0] + pixel[0];

        double sampleOut = dSamplesOutPtr[outI];
        double totalSample3D = dSamplesOutPtr[outI] + double(dSamplesPtr[inI]);

        Vector3d newColor = dPixelsOutPtr[outI] * sampleOut + Vector3d(dPixelsPtr[inI]);
        newColor /= totalSample3D;
        dPixelsOutPtr[outI] = newColor;
    }
}



//template<class T>
//class alignas(SimdRegiserAlignment<T>()) SimdRegister
//{
//    std::array<T, N> data;
//
//    friend operator*(SimdRegister, SimdRegister);
//};


namespace MRayCLI::RunNames
{
    using namespace std::literals;
    static constexpr auto Name = "run"sv;
    static constexpr auto Description = "Directly runs a given scene file without GUI"sv;
};

MRayError RunCommand::Invoke()
{
    return MRayError::OK;
}

CLI::App* RunCommand::GenApp(CLI::App& mainApp)
{
    using namespace MRayCLI::RunNames;
    CLI::App* converter = mainApp.add_subcommand(std::string(Name),
                                                 std::string(Description));

    return converter;
}

CommandI& RunCommand::Instance()
{
    static RunCommand c = {};
    return c;
}