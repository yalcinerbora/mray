#include "TextureFilter.h"
#include "RayPartitioner.h"
#include "DistributionFunctions.h"
#include "Filters.h"
#include "GenericTextureRW.h"

#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgBinaryPartition.h"
#include "Device/GPUAlgRadixSort.h"

#include "Core/ColorFunctions.h"
#include "Core/DeviceVisit.h"
#include "Core/GraphicsFunctions.h"

#ifdef MRAY_GPU_BACKEND_CUDA
    #include <cub/block/block_reduce.cuh>
#else
    #error "Only CUDA Version of Reconstruction filter is implemented"
#endif

enum class FilterMode
{
    SAMPLING,
    ACCUMULATE
};

MRAY_HYBRID MRAY_CGPU_INLINE
int32_t FilterRadiusToPixelWH(Float filterRadius)
{
    // At every 0.5 increment, conservative pixel estimate is increasing
    // [0]          = Single Pixel (Special Case)
    // (0, 0.5]     = 2x2
    // (0.5, 1]     = 3x3
    // (1, 1.5]     = 4x4
    // (1.5, 2]     = 5x5
    // etc...
    int32_t result = 1;
    if(filterRadius == Float(0)) return result;
    // Do division
    int32_t quot = static_cast<uint32_t>(filterRadius / Float(0.5));
    float remainder = fmod(filterRadius, Float(0.5));
    // Exact divisions reside on previous segment
    if(remainder == Float(0)) quot -= 1;

    result += (quot + 1);
    return result;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector2i FilterRadiusPixelRange(int32_t wh)
{
    Vector2i range(-(wh - 1) / 2, (wh + 2) / 2);
    return range;
}

template<class Filter, class DataFetcher>
MRAY_GPU MRAY_GPU_INLINE
Vector4 FilterPixel(const Vector2ui& pixelCoord,
                    //
                    const Vector2ui& spp,
                    FilterMode filterMode,
                    const Filter& FilterFunc,
                    const DataFetcher& FetchData)
{
    // We should sample "the peak" of the filter (so we need odd samples
    //Vector2ui oddSPP = Vector2ui((spp[0] & 1u) == 0 ? spp[0] - 1 : spp[0],
    //                             (spp[1] & 1u) == 0 ? spp[1] - 1 : spp[1]);
    Vector2ui oddSPP = spp;

    Vector2 wPixCoord = Vector2(pixelCoord);
    // We use float as a catch-all type
    // It is allocated as a max channel
    Vector4 writePix = Vector4::Zero();
    Float weightSum = Float(0);
    // Stochastically sample the up level via the filter
    // Mini Monte Carlo..
    for(uint32_t sppY = 0; sppY < oddSPP[1]; sppY++)
    for(uint32_t sppX = 0; sppX < oddSPP[0]; sppX++)
    {
        Vector2 dXY = Vector2(1) / Vector2(oddSPP);
        // Create a quasi sampler by perfectly stratifying the
        // sample space
        Vector2 xi = dXY * Float(0.5) + dXY * Vector2(sppX, sppY);

        Vector2 xy;
        Float pdf, totalSampleInv;
        if(filterMode == FilterMode::ACCUMULATE)
        {
            xy = xi * Float(2) * FilterFunc.Radius() - FilterFunc.Radius();
            pdf = totalSampleInv = Float(1);
        }
        else
        {
            auto sample = FilterFunc.Sample(xi);
            xy = sample.value;
            pdf = sample.pdf;
            totalSampleInv = dXY.Multiply();
        }

        // Eval the weight
        Float weight = FilterFunc.Evaluate(xy);
        Vector4 localPix = FetchData(wPixCoord + xy);
        // Actual calculation
        writePix += weight * localPix * totalSampleInv / pdf;
        // Do the ingegration seperately as well
        // we need to compansate
        weightSum += weight * totalSampleInv / pdf;
    }
    writePix /= weightSum;
    return writePix;
}

// TODO: Should we dedicate a warp per pixel?
template<uint32_t TPB, class Filter>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCGenerateMipmaps(// I-O
                       MRAY_GRID_CONSTANT const Span<MipArray<SurfViewVariant>> dSurfaces,
                       // Inputs
                       MRAY_GRID_CONSTANT const Span<const MipGenParams> dMipGenParamsList,
                       // Constants
                       MRAY_GRID_CONSTANT const uint32_t currentMipLevel,
                       MRAY_GRID_CONSTANT const Vector2ui spp,
                       MRAY_GRID_CONSTANT const uint32_t blockPerTexture,
                       MRAY_GRID_CONSTANT const FilterMode filterMode,
                       MRAY_GRID_CONSTANT const Filter FilterFunc)
{
    static constexpr Vector2ui TILE_SIZE = Vector2ui(32, 16);
    static_assert(TILE_SIZE.Multiply() == TPB);
    assert(dSurfaces.size() == dMipGenParamsList.size());

    // Block-stride loop
    KernelCallParams kp;
    uint32_t textureCount = static_cast<uint32_t>(dSurfaces.size());
    uint32_t blockCount = blockPerTexture * textureCount;
    for(uint32_t bI = kp.blockId; bI < blockCount; bI += kp.gridSize)
    {
        uint32_t tI = bI / blockPerTexture;
        uint32_t localBI = bI % blockPerTexture;
        // Load to local space
        MipGenParams curParams = dMipGenParamsList[tI];
        SurfViewVariant writeSurf = dSurfaces[tI][currentMipLevel];
        const SurfViewVariant readSurf = dSurfaces[tI][currentMipLevel - 1];
        //
        Vector2ui mipRes = Graphics::TextureMipSize(curParams.mipZeroRes,
                                                    currentMipLevel);
        Vector2ui parentRes = Graphics::TextureMipSize(curParams.mipZeroRes,
                                                       currentMipLevel - 1);
        Vector2 uvRatio = Vector2(parentRes) / Vector2(mipRes);
        // Skip this mip if it is already loaded.
        // This may happen when a texture has up to x amount of mips
        // but it can support log2(floor(max(res))) amount so we generate
        // these mipmaps.
        // Since we issue mip generation in bulk (meaning for all of the textures
        // mip generate level X will be called regardless of that texture has a valid
        // mip). We need to early exit for a texture that can not support that level of mip.
        // If variant is in monostate we skip this mip level generation
        if(curParams.validMips[currentMipLevel] ||
           std::holds_alternative<std::monostate>(writeSurf)) continue;

        // Loop over the blocks for this tex
        Vector2ui totalTiles = MathFunctions::DivideUp(mipRes, TILE_SIZE);
        for(uint32_t tileI = localBI; tileI < totalTiles.Multiply();
            tileI += blockPerTexture)
        {
            Vector2ui localPI = Vector2ui(kp.threadId % TILE_SIZE[0],
                                          kp.threadId / TILE_SIZE[0]);
            Vector2ui tile2D = Vector2ui(tileI % totalTiles[0],
                                         tileI / totalTiles[0]);
            Vector2ui wPixCoordInt = tile2D * TILE_SIZE + localPI;
            if(wPixCoordInt[0] >= mipRes[0] || wPixCoordInt[1] >= mipRes[1])
                continue;

            // Generic filter, reader can be defined via lambda
            Vector4 writePix = FilterPixel(wPixCoordInt, spp,
                                           filterMode, FilterFunc,
            [&](Vector2 rPixCoord) -> Vector4
            {
                // Find the upper level coordinate
                using Graphics::ConvertPixelIndices;
                rPixCoord = ConvertPixelIndices(rPixCoord,
                                                Vector2(parentRes),
                                                Vector2(mipRes));
                Vector2ui rPixCoordInt = Vector2ui(rPixCoord.RoundSelf());
                return GenericRead(rPixCoordInt, readSurf);
            });
            // Finally write the pixel
            GenericWrite(writeSurf, writePix, wPixCoordInt);
        }
    }
}

// TODO: Should we dedicate a warp per pixel?
template<uint32_t TPB, class Filter>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCClampImage(// Output
                  MRAY_GRID_CONSTANT const SurfViewVariant surfaceOut,
                  // Inputs
                  MRAY_GRID_CONSTANT const Span<const Byte> dBufferImage,
                  // Constants
                  MRAY_GRID_CONSTANT const Vector2ui surfaceImageRes,
                  MRAY_GRID_CONSTANT const Vector2ui bufferImageRes,
                  MRAY_GRID_CONSTANT const Vector2ui spp,
                  MRAY_GRID_CONSTANT const FilterMode filterMode,
                  MRAY_GRID_CONSTANT const Filter FilterFunc)
{
    static constexpr Vector2ui TILE_SIZE = Vector2ui(32, 16);
    static_assert(TILE_SIZE.Multiply() == TPB);

    // Pre-calculation, calculate uv ratio
    Vector2 uvRatio = Vector2(bufferImageRes) / Vector2(surfaceImageRes);

    KernelCallParams kp;
    // Loop over the tiles for this tex, each block is dedicated to a tile
    // (32, 16) pixels
    Vector2ui totalTiles = MathFunctions::DivideUp(surfaceImageRes, TILE_SIZE);
    for(uint32_t tileI = kp.blockId; tileI < totalTiles.Multiply();
        tileI += kp.gridSize)
    {
        Vector2ui localPI = Vector2ui(kp.threadId % TILE_SIZE[0],
                                      kp.threadId / TILE_SIZE[0]);
        Vector2ui tile2D = Vector2ui(tileI % totalTiles[0],
                                     tileI / totalTiles[0]);
        Vector2ui wPixCoordInt = tile2D * TILE_SIZE + localPI;
        //
        if(wPixCoordInt[0] >= surfaceImageRes[0] ||
           wPixCoordInt[1] >= surfaceImageRes[1])
            continue;

        // Generic filter, reader can be defined via lambda
        Vector4 writePix = FilterPixel(wPixCoordInt, spp,
                                       filterMode, FilterFunc,
        [&](Vector2 rPixCoord) -> Vector4
        {
            // Find the upper level coordinate
            using Graphics::ConvertPixelIndices;
            rPixCoord = ConvertPixelIndices(rPixCoord,
                                            Vector2(bufferImageRes),
                                            Vector2(surfaceImageRes));
            Vector2ui rPixCoordInt = Vector2ui(rPixCoord.RoundSelf());
            // Data is tightly packed, we can directly find the lienar index
            uint32_t pixCoordLinear = (rPixCoordInt[1] * bufferImageRes[0] +
                                       rPixCoordInt[0]);

            // Now the type fetch part, utilize surface variant to
            // find the type
            Vector4 outData = GenericReadFromBuffer(dBufferImage, surfaceOut,
                                                    pixCoordLinear);
            return outData;
        });
        // Finally write the pixel
        SurfViewVariant sOut = surfaceOut;
        GenericWrite(sOut, writePix, wPixCoordInt);
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCExpandSamplesToPixels(// Outputs
                             MRAY_GRID_CONSTANT const Span<uint32_t> dPixelIds,
                             MRAY_GRID_CONSTANT const Span<uint32_t> dIndices,
                             // Inputs
                             MRAY_GRID_CONSTANT const Span<const Vector2> dImgCoords,
                             // Constants
                             MRAY_GRID_CONSTANT const Float filterRadius,
                             MRAY_GRID_CONSTANT const uint32_t maxPixelPerSample,
                             MRAY_GRID_CONSTANT const Vector2i extent)
{
    int32_t filterWH = FilterRadiusToPixelWH(filterRadius);
    Vector2i range = FilterRadiusPixelRange(filterWH);
    // Don't use 1.0f exactly here
    // pixel is [0,1)
    Float pixelWidth = MathFunctions::PrevFloat<Float>(1);
    Float radiusSqr = filterRadius * filterRadius;

    KernelCallParams kp;
    uint32_t sampleCount = static_cast<uint32_t>(dImgCoords.size());
    for(uint32_t sampleIndex = kp.GlobalId(); sampleIndex < sampleCount;
        sampleIndex += kp.TotalSize())
    {
        Vector2 imgCoords = dImgCoords[sampleIndex];
        Vector2 relImgCoords;
        Vector2 fractions = Vector2(std::modf(imgCoords[0], &(relImgCoords[0])),
                                    std::modf(imgCoords[1], &(relImgCoords[1])));

        // If fractions is on the left subpixel and radius is odd,
        // shift the filter window
        Vector2i localRangeX = range;
        Vector2i localRangeY = range;
        if(filterWH % 2 == 0)
        {
            if(fractions[0] < 0.5f) localRangeX -= Vector2i(1);
            if(fractions[1] < 0.5f) localRangeY -= Vector2i(1);
        }

        // Actual write
        int32_t stride = 0;
        for(int32_t y = range[0]; y < range[1]; y++)
        for(int32_t x = range[0]; x < range[1]; x++)
        {
            Vector2 pixCoord = relImgCoords + Vector2(x, y);
            Vector2 pixCenter = pixCoord + Float(0.5);
            // TODO: Should we use pixCoord or center coord
            // which one is better?
            Float lengthSqr = (imgCoords - pixCenter).LengthSqr();

            // Skip if this pixel is out of range,
            // Filter WH is a conservative estimate
            // so this can happen
            bool doWrite = true;
            if(lengthSqr > radiusSqr) doWrite = false;

            // Get ready for writing
            Vector2i globalPixCoord = Vector2i(imgCoords) + Vector2i(x, y);

            bool pixOutside = (globalPixCoord[0] < 0            ||
                               globalPixCoord[0] >= extent[0]   ||
                               globalPixCoord[1] < 0            ||
                               globalPixCoord[1] >= extent[1]);
            // Do not write (obviously) if pixel is outside
            if(pixOutside) doWrite = false;

            // Now we can write
            assert(stride != maxPixelPerSample);
            constexpr uint32_t INVALID_INT = std::numeric_limits<uint32_t>::max();

            namespace Morton = Graphics::MortonCode;
            uint32_t pixelLinearId = Morton::Compose2D<uint32_t>(Vector2ui(globalPixCoord[1]));
            pixelLinearId = (!doWrite) ? INVALID_INT : pixelLinearId;
            uint32_t index = (!doWrite) ? INVALID_INT : sampleIndex;

            // Write windows, each sample has "maxPixelPerSample" amount of allocation
            // slots. These are available in a strided fashion, so writes can be coalesced
            // This is why we do not termiante this double loop, we either write INT_MAX
            // or actually write the result.
            uint32_t writeIndex = stride * sampleCount + sampleIndex;
            dPixelIds[writeIndex] = pixelLinearId;
            dIndices[writeIndex] = index;
            stride++;
        }
        // All Done!
    }
}

template <uint32_t TPB, uint32_t LOGICAL_WARP_SIZE, class Filter>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCFilterToImgWarpRGB(MRAY_GRID_CONSTANT const ImageSpan<3> img,
                          // Inputs per segment
                          MRAY_GRID_CONSTANT const Span<const uint32_t> dStartOffsets,
                          MRAY_GRID_CONSTANT const Span<const uint32_t> dPixelIds,
                          // Inputs per thread
                          MRAY_GRID_CONSTANT const Span<const uint32_t> dIndices,
                          // Inputs Accessed by SampleId
                          MRAY_GRID_CONSTANT const Span<const Spectrum> dValues,
                          MRAY_GRID_CONSTANT const Span<const Vector2> dImgCoords,
                          // Constants
                          MRAY_GRID_CONSTANT const Float scalarWeightMultiplier,
                          MRAY_GRID_CONSTANT const Filter filter)
{
    KernelCallParams kp;
    assert(dStartOffsets.size() == dPixelIds.size());
    static_assert(TPB % LOGICAL_WARP_SIZE == 0);

    // Some constants
    static constexpr uint32_t WARP_PER_BLOCK = TPB / LOGICAL_WARP_SIZE;
    const uint32_t totalWarpCount = WARP_PER_BLOCK * kp.gridSize;
    const uint32_t globalWarpId = kp.GlobalId() / LOGICAL_WARP_SIZE;
    const uint32_t localWarpId = kp.threadId / LOGICAL_WARP_SIZE;
    const uint32_t laneId = kp.GlobalId() % LOGICAL_WARP_SIZE;

    using WarpReduceVec4 = cub::WarpReduce<Vector4, LOGICAL_WARP_SIZE>;
    using ReduceShMem = typename WarpReduceVec4::TempStorage;
    // Per-Warp Shared Memory
    MRAY_SHARED_MEMORY Vector2ui    sSegmentRange[WARP_PER_BLOCK];
    MRAY_SHARED_MEMORY uint32_t     sResonsiblePixel[WARP_PER_BLOCK];
    MRAY_SHARED_MEMORY ReduceShMem  sReduceMem[WARP_PER_BLOCK];

    // Warp-stride loop
    uint32_t segmentCount = static_cast<uint32_t>(dStartOffsets.size());
    for(uint32_t segmentIndex = globalWarpId; segmentIndex < segmentCount;
        segmentIndex += totalWarpCount)
    {
        // Load items to warp level
        if(laneId == 0) sSegmentRange[localWarpId][0] = dStartOffsets[segmentIndex + 0];
        if(laneId == 1) sSegmentRange[localWarpId][1] = dStartOffsets[segmentIndex + 1];
        if(laneId == 2) sResonsiblePixel[localWarpId] = dPixelIds[segmentIndex];
        // Wait for these writes to be visible across warp
        WarpSynchronize<LOGICAL_WARP_SIZE>();

        // Locally compute the coordinates
        namespace Morton = Graphics::MortonCode;
        Vector2i pixCoordsInt = Vector2i(Morton::Decompose2D(sResonsiblePixel[localWarpId]));
        Vector2 pixCoords = Vector2(pixCoordsInt) + Float(0.5);

        uint32_t sampleStart = sSegmentRange[localWarpId][0];
        uint32_t sampleCount = sSegmentRange[localWarpId][1] - sampleStart;
        uint32_t iterationCount = MathFunctions::NextMultiple(sampleCount, LOGICAL_WARP_SIZE);

        Vector4 totalValue = Vector4::Zero();
        for(uint32_t i = 0; i < iterationCount; i++)
        {
            // Specifically set the actual index here
            // We want all lanes in warp to be active
            // (due to WarpSync and other stuff)
            uint32_t warpI = i * LOGICAL_WARP_SIZE + laneId;
            uint32_t sampleIndex = sampleStart + warpI;

            Vector4 value = Vector4::Zero();
            if(warpI < sampleCount)
            {
                uint32_t readIndex = dIndices[sampleIndex];
                Vector3 sampleVal = Vector3(dValues[readIndex]);
                Vector2 sampleCoord = dImgCoords[readIndex];
                Float weight = filter.Evaluate(pixCoords - sampleCoord) * scalarWeightMultiplier;
                sampleVal *= weight;
                value = Vector4(sampleVal, weight);
            }

            totalValue += WarpReduceVec4(sReduceMem[localWarpId]).Sum(value);
            WarpSynchronize<LOGICAL_WARP_SIZE>();
        }
        // Now all is reduced, warp leader can write to the img buffer
        // Here we assume we are the sole warp responsible for this
        // pixel. However; other writes would've been occured on
        // previous calls. So we need do an non-atomic add here.
        if(laneId == 0)
        {
            Float weight = img.FetchWeight(pixCoordsInt);
            img.StoreWeight(weight + totalValue[0], pixCoordsInt);

            Vector3 pixValue = img.FetchPixel(pixCoordsInt);
            img.StorePixel(pixValue + Vector3(totalValue),
                               pixCoordsInt);
        }
    }
}

template<class Filter>
void ReconFilterGenericRGB(// Output
                           const ImageSpan<3>& img,
                           // I-O
                           RayPartitioner& partitioner,
                           // Input
                           const Span<const Spectrum>& dValues,
                           const Span<const Vector2>& dImgCoords,
                           // Constants
                           Float scalarWeightMultiplier,
                           Float filterRadius,
                           Filter filter,
                           const GPUSystem& gpuSystem)
{
    // Get algo temp buffers
    assert(dValues.size() == dImgCoords.size());
    uint32_t elementCount = static_cast<uint32_t>(dValues.size());

    uint32_t wh = FilterRadiusToPixelWH(filterRadius);
    uint32_t maxPixelPerSample = wh * wh;
    uint32_t totalPPS = elementCount * maxPixelPerSample;

    // Maximum partition count is relative tot he image resolution
    Vector2i maxPixels = img.Extent() + Vector2i(wh);
    uint32_t maxPartitionCount = static_cast<uint32_t>(maxPixels.Multiply());

    auto [dIndices, dKeys] = partitioner.Start(totalPPS, maxPartitionCount, false);

    using namespace std::string_view_literals;
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    queue.IssueSaturatingKernel<KCExpandSamplesToPixels>
    (
        "KCExpandSamplesToPixels"sv,
        KernelIssueParams{.workCount = elementCount},
        // Outputs
        dKeys,
        dIndices,
        // Inputs
        dImgCoords,
        // Constants
        filterRadius,
        maxPixelPerSample,
        img.Extent()
    );

    using namespace Bit;
    Vector2ui sortRange = Vector2ui(0, RequiredBitsToRepresent(maxPartitionCount));
    auto
    [
        hPartitionCount,
        //
        isHostVisible,
        dPartitionStartOffsets,
        dPartitionPixelIds,
        //
        dPartitionIndices,
        dPartitionKeys
    ] = partitioner.MultiPartition(dKeys, dIndices,
                                    Vector2ui::Zero(),
                                    sortRange, queue, true);
    assert(isHostVisible == false);

    // We've partitioned now determine the kernel with a basic
    // heuristic. Just find the average spp and use it.
    // This heuristic assumes samples are uniformly distributed
    // on the pixel range.
    uint32_t averageSPP = elementCount / hPartitionCount[0];

    // TODO: Currently, only warp-dedicated reduction.
    // If tracer pounds towards a specific region on the scene
    // (i.e. 4x4 pixel wide), Each warp will have ~2M / 16
    // amount of work to do. This is not optimal.
    // We need to create block and device variants of this reduction.
    //
    // According to the simple test dedicating nearest amount of warps (rounded down)
    // was slower. Interestingly, for a 5x5 kernel, logical warp size of 1
    // (a thread) was the fastest. So dividing spp with this.
    static constexpr uint32_t WORK_PER_THREAD = 16;
    uint32_t logicalWarpSize = MathFunctions::DivideUp(averageSPP, WORK_PER_THREAD);
    logicalWarpSize = MathFunctions::PrevPowerOfTwo(logicalWarpSize);
    logicalWarpSize = std::min(logicalWarpSize, WarpSize());

    // Some boilerplate to make the code more readable
    auto KernelCall = [&]<auto Kernel>(std::string_view Name)
    {
        Span<const Spectrum> dX = dValues;
        Span<const Vector2> dY = dImgCoords;

        uint32_t blockCount =
            queue.RecommendedBlockCountDevice(Kernel,
                                              StaticThreadPerBlock1D(), 0);
        queue.IssueExactKernel<Kernel>
        (
            Name,
            KernelExactIssueParams
            {
                .gridSize = blockCount,
                .blockSize = StaticThreadPerBlock1D()
            },
            img,
            // Inputs per segment
            dPartitionStartOffsets,
            dPartitionPixelIds,
            // Inputs per thread
            dPartitionIndices,
            // Inputs Accessed by SampleId
            dX, dY,
            // Constants
            scalarWeightMultiplier,
            filter
        );
    };
    constexpr std::array WK =
    {
        KCFilterToImgWarpRGB<StaticThreadPerBlock1D(), 1, Filter>,
        KCFilterToImgWarpRGB<StaticThreadPerBlock1D(), 2, Filter>,
        KCFilterToImgWarpRGB<StaticThreadPerBlock1D(), 4, Filter>,
        KCFilterToImgWarpRGB<StaticThreadPerBlock1D(), 8, Filter>,
        KCFilterToImgWarpRGB<StaticThreadPerBlock1D(), 16, Filter>,
        KCFilterToImgWarpRGB<StaticThreadPerBlock1D(), 32, Filter>
    };
    switch(logicalWarpSize)
    {
        case 0: KernelCall.template operator()<WK[0]>("KCFilterToImgWarpRGB<1>"sv); break;
        case 1: KernelCall.template operator()<WK[1]>("KCFilterToImgWarpRGB<2>"sv); break;
        case 2: KernelCall.template operator()<WK[2]>("KCFilterToImgWarpRGB<4>"sv); break;
        case 3: KernelCall.template operator()<WK[3]>("KCFilterToImgWarpRGB<8>"sv); break;
        case 4: KernelCall.template operator()<WK[4]>("KCFilterToImgWarpRGB<16>"sv); break;
        case 5: KernelCall.template operator()<WK[5]>("KCFilterToImgWarpRGB<32>"sv); break;
        default: throw MRayError("Unknown logical warp size!");
    }
    // All Done!
}

template<class Filter>
void MultiPassReconFilterGenericRGB(// Output
                                    const ImageSpan<3>& img,
                                    // I-O
                                    RayPartitioner& partitioner,
                                    // Input
                                    const Span<const Spectrum>& dValues,
                                    const Span<const Vector2>& dImgCoords,
                                    // Constants
                                    uint32_t parallelHint,
                                    Float scalarWeightMultiplier,
                                    Float filterRadius,
                                    Filter filter,
                                    const GPUSystem& gpuSystem)
{
    // This partition-based design uses too much memory
    // ~(filter_width * filter_height * sampleCount * 2 * sizeof(uint32_t))
    // By default, renderers launch ~2M samples
    // For 3x3 filter size,
    //    - 2_MiB * 2 * 4 * 9 = 72MiB of temporary storage
    // For 5x5 filter size (maximum case probably),
    //    - 2_MiB * 2 * 4 * 25 = 400MiB of temporary storage (!)
    // Most of this temporary memory is shared by the partitioner,
    // however, partitioner's memory sticks throughout the runtime.
    //
    // All in all, we do multiple passes over the samples to
    // reduce memory usage.
    assert(dValues.size() == dImgCoords.size());
    uint32_t totalWork = static_cast<uint32_t>(dValues.size());
    uint32_t wh = FilterRadiusToPixelWH(filterRadius);
    uint32_t maxPixelPerSample = wh * wh;
    uint32_t totalPPS = totalWork * maxPixelPerSample;

    // Try to comply the parallelization hint
    // Divide the work equally
    uint32_t iterations = MathFunctions::DivideUp(totalPPS, parallelHint);
    uint32_t workPerIter = MathFunctions::DivideUp(totalPPS, iterations);
    for(uint32_t i = 0; i < iterations; i++)
    {
        uint32_t start = workPerIter * i;
        uint32_t end = std::min(workPerIter * i + 1, totalWork);
        uint32_t count = end - start;

        Span<const Spectrum> dLocalValues = dValues.subspan(start, count);
        Span<const Vector2> dLocalImgCoords = dImgCoords.subspan(start, count);
        ReconFilterGenericRGB(img, partitioner,
                              dLocalValues, dLocalImgCoords,
                              scalarWeightMultiplier, filterRadius,
                              filter, gpuSystem);
    }
}

template<class Filter>
void GenerateMipsGeneric(const std::vector<MipArray<SurfRefVariant>>& textures,
                         const std::vector<MipGenParams>& mipGenParams,
                         const GPUSystem& gpuSystem, Filter filter)
{
    assert(textures.size() == mipGenParams.size());
    // TODO: Textures should be partitioned with respect to
    // devices, so that we can launch kernel from those devices
    const GPUDevice& bestDevice = gpuSystem.BestDevice();
    const GPUQueue& queue = bestDevice.GetComputeQueue(0);

    // We will dedicate N blocks for each texture.
    static constexpr uint32_t THREAD_PER_BLOCK = 512;
    static constexpr uint32_t BLOCK_PER_TEXTURE = 256;
    static constexpr auto Kernel = KCGenerateMipmaps<THREAD_PER_BLOCK, Filter>;

    // Find maximum block count for state allocation
    // TODO: Change this so that it is relative to the
    // filter radius.
    static constexpr Vector2ui SPP = Vector2ui(8, 8);
    uint32_t blockCount = queue.RecommendedBlockCountDevice(Kernel,
                                                            THREAD_PER_BLOCK, 0);

    // We can temporarily allocate here. This will be done at
    // initialization time.
    DeviceLocalMemory mem(gpuSystem.BestDevice());
    Span<MipArray<SurfViewVariant>> dSufViews;
    Span<MipGenParams> dMipGenParams;
    MemAlloc::AllocateMultiData(std::tie(dSufViews, dMipGenParams),
                                mem, {textures.size(), textures.size()});

    // Copy references
    std::vector<MipArray<SurfViewVariant>> hSurfViews;
    hSurfViews.reserve(textures.size());
    for(const MipArray<SurfRefVariant>& surfRefs : textures)
    {
        MipArray<SurfViewVariant> mipViews;
        for(size_t i = 0; i < TracerConstants::MaxTextureMipCount; i++)
        {
            const SurfRefVariant& surf = surfRefs[i];
            mipViews[i] = std::visit([](auto&& v) -> SurfViewVariant
            {
                using T = std::remove_cvref_t<decltype(v)>;
                if constexpr(std::is_same_v<T, std::monostate>)
                    return std::monostate{};
                else return v.View();
            }, surf);
        }
        hSurfViews.push_back(mipViews);
    }

    auto hSurfViewSpan = Span<MipArray<SurfViewVariant>>(hSurfViews.begin(),
                                                         hSurfViews.end());
    auto hMipGenParams = Span<const MipGenParams>(mipGenParams.cbegin(),
                                                  mipGenParams.cend());
    queue.MemcpyAsync(dSufViews, ToConstSpan(hSurfViewSpan));
    queue.MemcpyAsync(dMipGenParams, hMipGenParams);

    // Since texture writes are not coherent,
    // we need to call one kernel for each level of mips
    uint16_t maxMipCount = std::transform_reduce(mipGenParams.cbegin(),
                                                 mipGenParams.cend(),
                                                 std::numeric_limits<uint16_t>::min(),
    [](uint16_t l, uint16_t r)
    {
        return std::max(l, r);
    },
    [](const MipGenParams& p) -> uint16_t
    {
        return p.mipCount;
    });

    // Start from 1, we assume miplevel zero is already available
    for(uint16_t i = 1; i < maxMipCount; i++)
    {
        uint32_t BlockPerTexture = std::max(1u, BLOCK_PER_TEXTURE >> 1);
        using namespace std::string_view_literals;
        queue.IssueExactKernel<Kernel>
        (
            "KCGenerateMipmaps"sv,
            KernelExactIssueParams
            {
                .gridSize = blockCount,
                .blockSize = THREAD_PER_BLOCK
            },
            // I-O
            dSufViews,
            // Inputs
            dMipGenParams,
            // Constants
            i,
            SPP,
            BlockPerTexture,
            FilterMode::ACCUMULATE,
            filter
        );
    }
    queue.Barrier().Wait();
}

template<class Filter>
void ClampImageFromBufferGeneric(// Output
                                 const SurfRefVariant& surf,
                                 // Input
                                 const Span<const Byte>& dDataBuffer,
                                 // Constants
                                 const Vector2ui& surfImageDims,
                                 const Vector2ui& bufferImageDims,
                                 Filter filter,
                                 const GPUQueue& queue)
{
    using MathFunctions::DivideUp;
    static constexpr Vector2ui TILE_SIZE = Vector2ui(32, 16);
    static constexpr uint32_t THREAD_PER_BLOCK = TILE_SIZE.Multiply();
    static constexpr auto Kernel = KCClampImage<THREAD_PER_BLOCK, Filter>;
    // Find maximum block count for state allocation
    // TODO: Change this so that it is relative to the
    // filter radius.
    static constexpr Vector2ui SPP = Vector2ui(8, 8);
    uint32_t blockCount = queue.RecommendedBlockCountDevice(Kernel, THREAD_PER_BLOCK, 0);
    uint32_t blockPerTexture = DivideUp(surfImageDims, TILE_SIZE).Multiply();
    blockCount = std::min(blockPerTexture, blockCount);

    SurfViewVariant surfRef = std::visit([](auto&& v) -> SurfViewVariant
    {
        using T = std::remove_cvref_t<decltype(v)>;
        if constexpr(std::is_same_v<T, std::monostate>)
            return std::monostate{};
        else return v.View();
    }, surf);

    using namespace std::string_view_literals;
    queue.IssueExactKernel<Kernel>
    (
        "KCClampImage"sv,
        KernelExactIssueParams
        {
            .gridSize = blockCount,
            .blockSize = THREAD_PER_BLOCK
        },
        // Output
        surfRef,
        // Inputs
        dDataBuffer,
        // Constants
        surfImageDims,
        bufferImageDims,
        SPP,
        // Use sampling here, quality is not that important
        // (We are clamping textures)
        FilterMode::ACCUMULATE,
        filter
    );
}

template<FilterType::E E, class FF>
TextureFilterT<E, FF>::TextureFilterT(const GPUSystem& system,
                                      Float fR)
    : gpuSystem(system)
    , filterRadius(fR)
{}

template<FilterType::E E, class FF>
void TextureFilterT<E, FF>::GenerateMips(const std::vector<MipArray<SurfRefVariant>>& textures,
                                         const std::vector<MipGenParams>& params) const
{
    using namespace std::string_literals;
    static const std::string Name = ("ConvertColor"s + std::string(FilterType::ToString(E)));
    static const auto annotation = gpuSystem.CreateAnnotation(Name);
    const auto _ = annotation.AnnotateScope();

    GenerateMipsGeneric(textures, params, gpuSystem, FF(filterRadius));
}

template<FilterType::E E, class FF>
void TextureFilterT<E, FF>::ClampImageFromBuffer(// Output
                                                 const SurfRefVariant& surf,
                                                 // Input
                                                 const Span<const Byte>& dDataBuffer,
                                                 // Constants
                                                 const Vector2ui& surfImageDims,
                                                 const Vector2ui& bufferImageDims,
                                                 const GPUQueue& queue) const
{
    ClampImageFromBufferGeneric(surf, dDataBuffer,
                                surfImageDims, bufferImageDims,
                                FF(filterRadius), queue);
}

template<FilterType::E E, class FF>
void TextureFilterT<E, FF>::ReconstructionFilterRGB(// Output
                                                    const ImageSpan<3>& img,
                                                    // I-O
                                                    RayPartitioner& partitioner,
                                                    // Input
                                                    const Span<const Spectrum>& dValues,
                                                    const Span<const Vector2>& dImgCoords,
                                                    // Constants
                                                    uint32_t parallelHint,
                                                    Float scalarWeightMultiplier) const
{
    MultiPassReconFilterGenericRGB(img, partitioner, dValues, dImgCoords,
                                   parallelHint, scalarWeightMultiplier,
                                   filterRadius, FF(filterRadius),
                                   gpuSystem);
}

template<FilterType::E E, class FF>
Vector2ui TextureFilterT<E, FF>::FilterExtent() const
{
    return Vector2ui(FilterRadiusToPixelWH(filterRadius));
}

template TextureFilterT<FilterType::BOX, BoxFilter>;
template TextureFilterT<FilterType::TENT, TentFilter>;
template TextureFilterT<FilterType::GAUSSIAN, GaussianFilter>;
template TextureFilterT<FilterType::MITCHELL_NETRAVALI, MitchellNetravaliFilter>;