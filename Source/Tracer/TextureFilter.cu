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

static constexpr uint32_t INVALID_MORTON = std::numeric_limits<uint32_t>::max();

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

MRAY_HYBRID MRAY_GPU_INLINE
Tuple<Vector3, Float> ConvertNaNsToColor(Spectrum value, Float weight)
{
    if(value.HasNaN())
        return Tuple(BIG_MAGENTA(), weight * Float(128.0));
    else
        return Tuple(Vector3(value), weight);
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
        Vector2ui totalTiles = Math::DivideUp(mipRes, TILE_SIZE);
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
    Vector2ui totalTiles = Math::DivideUp(surfaceImageRes, TILE_SIZE);
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
                             MRAY_GRID_CONSTANT const Span<CommonKey> dPixelIds,
                             MRAY_GRID_CONSTANT const Span<CommonIndex> dIndices,
                             // Inputs
                             MRAY_GRID_CONSTANT const Span<const ImageCoordinate> dImgCoords,
                             // Constants
                             MRAY_GRID_CONSTANT const Float filterRadius,
                             MRAY_GRID_CONSTANT const uint32_t maxPixelPerSample,
                             MRAY_GRID_CONSTANT const Vector2i extent)
{
    int32_t filterWH = FilterRadiusToPixelWH(filterRadius);
    Vector2i range = FilterRadiusPixelRange(filterWH);
    // Don't use 1.0f exactly here
    // pixel is [0,1)
    Float pixelWidth = Math::PrevFloat<Float>(1);
    Float radiusSqr = filterRadius * filterRadius;

    KernelCallParams kp;
    uint32_t sampleCount = static_cast<uint32_t>(dImgCoords.size());
    for(uint32_t sampleIndex = kp.GlobalId(); sampleIndex < sampleCount;
        sampleIndex += kp.TotalSize())
    {
        Vector2 imgCoords = dImgCoords[sampleIndex].GetPixelIndex();
        imgCoords += Vector2(0.5);
        Vector2 relImgCoords;
        Vector2 fractions = Vector2(std::modf(imgCoords[0], &(relImgCoords[0])),
                                    std::modf(imgCoords[1], &(relImgCoords[1])));

        // If fractions is on the left subpixel and radius is odd,
        // shift the filter window
        Vector2i localRangeX = range;
        Vector2i localRangeY = range;
        if(filterWH % 2 == 0)
        {
            if(fractions[0] < Float(0.5)) localRangeX -= Vector2i(1);
            if(fractions[1] < Float(0.5)) localRangeY -= Vector2i(1);
        }

        // Actual write
        int32_t stride = 0;
        for(int32_t y = localRangeX[0]; y < localRangeX[1]; y++)
        for(int32_t x = localRangeX[0]; x < localRangeX[1]; x++)
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
            if(radiusSqr != Float(0) && lengthSqr > radiusSqr) doWrite = false;

            // Get ready for writing
            Vector2i globalPixCoord = Vector2i(imgCoords) + Vector2i(x, y);
            bool pixOutside = (globalPixCoord[0] < 0            ||
                               globalPixCoord[0] >= extent[0]   ||
                               globalPixCoord[1] < 0            ||
                               globalPixCoord[1] >= extent[1]);
            // Do not write (obviously) if pixel is outside
            if(pixOutside) doWrite = false;

            // Now we can write
            assert(stride < maxPixelPerSample);
            namespace Morton = Graphics::MortonCode;
            uint32_t pixelLinearId = Morton::Compose2D<uint32_t>(Vector2ui(globalPixCoord));
            pixelLinearId = (!doWrite) ? INVALID_MORTON : pixelLinearId;
            uint32_t index = (!doWrite) ? INVALID_MORTON : sampleIndex;

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
void KCFilterToImgWarpRGB(MRAY_GRID_CONSTANT const ImageSpan img,
                          // Inputs per segment
                          MRAY_GRID_CONSTANT const Span<const uint32_t> dStartOffsets,
                          MRAY_GRID_CONSTANT const Span<const CommonKey> dPixelIds,
                          // Inputs per thread
                          MRAY_GRID_CONSTANT const Span<const CommonIndex> dIndices,
                          // Inputs Accessed by SampleId
                          MRAY_GRID_CONSTANT const Span<const Spectrum> dValues,
                          MRAY_GRID_CONSTANT const Span<const ImageCoordinate> dImgCoords,
                          // Constants
                          MRAY_GRID_CONSTANT const Span<const uint32_t, 1u> hPartitionCount,
                          MRAY_GRID_CONSTANT const Float scalarWeightMultiplier,
                          MRAY_GRID_CONSTANT const Filter filter)
{
    KernelCallParams kp;
    assert(dStartOffsets.size() == (dPixelIds.size() + 1));
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
    MRAY_SHARED_MEMORY CommonKey    sResonsiblePixel[WARP_PER_BLOCK];
    MRAY_SHARED_MEMORY ReduceShMem  sReduceMem[WARP_PER_BLOCK];

    // Warp-stride loop
    uint32_t segmentCount = hPartitionCount[0];
    for(uint32_t segmentIndex = globalWarpId; segmentIndex < segmentCount;
        segmentIndex += totalWarpCount)
    {
        static constexpr uint32_t LOAD_0 = (0 % LOGICAL_WARP_SIZE);
        static constexpr uint32_t LOAD_1 = (1 % LOGICAL_WARP_SIZE);
        static constexpr uint32_t LOAD_2 = (2 % LOGICAL_WARP_SIZE);

        // Load items to warp level
        if(laneId == LOAD_0) sSegmentRange[localWarpId][0] = dStartOffsets[segmentIndex + 0];
        if(laneId == LOAD_1) sSegmentRange[localWarpId][1] = dStartOffsets[segmentIndex + 1];
        if(laneId == LOAD_2) sResonsiblePixel[localWarpId] = dPixelIds[segmentIndex];
        // Wait for these writes to be visible across warp
        WarpSynchronize<LOGICAL_WARP_SIZE>();

        // This partition is residuals, we conservatively allocated but not every
        // potential filter slot is not filled. Skip this partition
        if(sResonsiblePixel[localWarpId] == INVALID_MORTON)
            continue;

        // Locally compute the coordinates
        namespace Morton = Graphics::MortonCode;
        Vector2i pixCoordsInt = Vector2i(Morton::Decompose2D(sResonsiblePixel[localWarpId]));
        Vector2 pixCoords = Vector2(pixCoordsInt) + Float(0.5);

        uint32_t sampleStart = sSegmentRange[localWarpId][0];
        uint32_t sampleCount = sSegmentRange[localWarpId][1] - sampleStart;
        uint32_t iterationCount = Math::NextMultiple(sampleCount, LOGICAL_WARP_SIZE);

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
                Vector2 sampleCoord = dImgCoords[readIndex].GetPixelIndex();
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
            img.StoreWeight(weight + totalValue[3], pixCoordsInt);

            Vector3 pixValue = img.FetchPixel(pixCoordsInt);
            img.StorePixel(pixValue + Vector3(totalValue),
                           pixCoordsInt);
        }
    }
}

template <class Filter>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCFilterToImgAtomicRGB(MRAY_GRID_CONSTANT const ImageSpan img,
                            // Input
                            MRAY_GRID_CONSTANT const Span<const Spectrum> dValues,
                            MRAY_GRID_CONSTANT const Span<const ImageCoordinate> dImgCoords,
                            // Constants
                            MRAY_GRID_CONSTANT const Float scalarWeightMultiplier,
                            MRAY_GRID_CONSTANT const Filter filter)
{
    Float filterRadius = filter.Radius();
    Vector2i extent = img.Extent();
    int32_t filterWH = FilterRadiusToPixelWH(filterRadius);
    Vector2i range = FilterRadiusPixelRange(filterWH);
    // Don't use 1.0f exactly here
    // pixel is [0,1)
    Float pixelWidth = Math::PrevFloat<Float>(1);
    Float radiusSqr = filterRadius * filterRadius;

    assert(dValues.size() == dImgCoords.size());
    uint32_t sampleCount = static_cast<uint32_t>(dImgCoords.size());

    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < sampleCount; i += kp.TotalSize())
    {
        Vector2 imgCoords = dImgCoords[i].GetPixelIndex();
        imgCoords += Vector2(0.5);
        Vector2 relImgCoords;
        Vector2 fractions = Vector2(std::modf(imgCoords[0], &(relImgCoords[0])),
                                    std::modf(imgCoords[1], &(relImgCoords[1])));

        // If fractions is on the left subpixel and radius is odd,
        // shift the filter window
        Vector2i localRangeX = range;
        Vector2i localRangeY = range;
        if(filterWH % 2 == 0)
        {
            if(fractions[0] < Float(0.5)) localRangeX -= Vector2i(1);
            if(fractions[1] < Float(0.5)) localRangeY -= Vector2i(1);
        }

        // Actual write
        for(int32_t y = localRangeX[0]; y < localRangeX[1]; y++)
        for(int32_t x = localRangeX[0]; x < localRangeX[1]; x++)
        {
            Vector2 pixCoord = relImgCoords + Vector2(x, y);
            Vector2 pixCenter = pixCoord + Float(0.5);
            // TODO: Should we use pixCoord or center coord
            // which one is better?
            Float lengthSqr = (imgCoords - pixCenter).LengthSqr();

            // Skip if this pixel is out of range,
            // Filter WH is a conservative estimate
            // so this can happen
            if(radiusSqr != Float(0) && lengthSqr > radiusSqr)
                continue;

            // Get ready for writing
            Vector2i globalPixCoord = Vector2i(imgCoords) + Vector2i(x, y);
            bool pixOutside = (globalPixCoord[0] < 0            ||
                               globalPixCoord[0] >= extent[0]   ||
                               globalPixCoord[1] < 0            ||
                               globalPixCoord[1] >= extent[1]);
            // Do not write (obviously) if pixel is outside
            if(pixOutside) continue;

            // globalPixCoord is index
            Float weight = filter.Evaluate(imgCoords - pixCenter) * scalarWeightMultiplier;
            Vector3 value = Vector3(dValues[i]) * weight;
            if(std::abs(weight) < MathConstants::Epsilon<Float>()) continue;

            img.AddToPixelAtomic(value, globalPixCoord);
            img.AddToWeightAtomic(weight, globalPixCoord);
        }
        // All Done!
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCSetImagePixels(MRAY_GRID_CONSTANT const ImageSpan img,
                      // Input
                      MRAY_GRID_CONSTANT const Span<const Spectrum> dValues,
                      MRAY_GRID_CONSTANT const Span<const Float> dFilterWeights,
                      MRAY_GRID_CONSTANT const Span<const ImageCoordinate> dImgCoords,
                      // Constants
                      MRAY_GRID_CONSTANT const Float scalarWeightMultiplier)
{
    uint32_t sampleCount = static_cast<uint32_t>(dImgCoords.size());

    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < sampleCount; i += kp.TotalSize())
    {
        Vector2i pixCoords = Vector2i(dImgCoords[i].pixelIndex);

        Vector3 val = img.FetchPixel(pixCoords);
        Float weight = img.FetchWeight(pixCoords);

        auto [valueIn, sampleIn] = ConvertNaNsToColor(dValues[i],
                                                      dFilterWeights[i]);
        val += valueIn;
        weight += sampleIn;

        img.StorePixel(val, pixCoords);
        img.StoreWeight(weight, pixCoords);
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCSetImagePixelsIndirect(MRAY_GRID_CONSTANT const ImageSpan img,
                              // Input
                              MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                              MRAY_GRID_CONSTANT const Span<const Spectrum> dValues,
                              MRAY_GRID_CONSTANT const Span<const Float> dFilterWeights,
                              MRAY_GRID_CONSTANT const Span<const ImageCoordinate> dImgCoords,
                              // Constants
                              MRAY_GRID_CONSTANT const Float scalarWeightMultiplier)
{
    uint32_t sampleCount = static_cast<uint32_t>(dIndices.size());

    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < sampleCount; i += kp.TotalSize())
    {
        uint32_t index = dIndices[i];
        Vector2i pixCoords = Vector2i(dImgCoords[index].pixelIndex);

        Vector3 val = img.FetchPixel(pixCoords);
        Float weight = img.FetchWeight(pixCoords);
        auto [valueIn, weightIn] = ConvertNaNsToColor(dValues[index],
                                                      dFilterWeights[index]);
        val += valueIn;
        weight += weightIn * scalarWeightMultiplier;

        img.StorePixel(val, pixCoords);
        img.StoreWeight(weight, pixCoords);
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCSetImagePixelsIndirectAtomic(MRAY_GRID_CONSTANT const ImageSpan img,
                                    // Input
                                    MRAY_GRID_CONSTANT const Span<const RayIndex> dIndices,
                                    MRAY_GRID_CONSTANT const Span<const Spectrum> dValues,
                                    MRAY_GRID_CONSTANT const Span<const Float> dFilterWeights,
                                    MRAY_GRID_CONSTANT const Span<const ImageCoordinate> dImgCoords,
                                    // Constants
                                    MRAY_GRID_CONSTANT const Float scalarWeightMultiplier)
{
    uint32_t sampleCount = static_cast<uint32_t>(dIndices.size());

    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < sampleCount; i += kp.TotalSize())
    {
        uint32_t index = dIndices[i];
        Vector2i pixCoords = Vector2i(dImgCoords[index].pixelIndex);
        auto [valueIn, weightIn] = ConvertNaNsToColor(dValues[index],
                                                      dFilterWeights[index]);
        Vector3 addVal = valueIn;
        Float addWeight = weightIn * scalarWeightMultiplier;

        img.AddToPixelAtomic(addVal, pixCoords);
        img.AddToWeightAtomic(addWeight, pixCoords);
    }
}

void SetImagePixelsIndirect(// Output
                            const ImageSpan& img,
                            // Input
                            const Span<const RayIndex>& dIndices,
                            const Span<const Spectrum>& dValues,
                            const Span<const Float>& dFilterWeights,
                            const Span<const ImageCoordinate>& dImgCoords,
                            // Constants
                            Float scalarWeightMultiplier,
                            const GPUQueue& queue)
{
    assert(dValues.size() == dFilterWeights.size());
    assert(dFilterWeights.size() == dImgCoords.size());
    using namespace std::string_view_literals;
    queue.IssueSaturatingKernel<KCSetImagePixelsIndirect>
    (
        "KCSetImagePixelsIndirect",
        KernelIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
        img,
        dIndices,
        dValues,
        dFilterWeights,
        dImgCoords,
        scalarWeightMultiplier
    );
}

void SetImagePixelsIndirectAtomic(// Output
                                  const ImageSpan& img,
                                  // Input
                                  const Span<const RayIndex>& dIndices,
                                  const Span<const Spectrum>& dValues,
                                  const Span<const Float>& dFilterWeights,
                                  const Span<const ImageCoordinate>& dImgCoords,
                                  // Constants
                                  Float scalarWeightMultiplier,
                                  const GPUQueue& queue)
{
    assert(dValues.size() == dFilterWeights.size());
    assert(dFilterWeights.size() == dImgCoords.size());
    using namespace std::string_view_literals;
    queue.IssueSaturatingKernel<KCSetImagePixelsIndirectAtomic>
    (
        "KCSetImagePixelsIndirectAtomic",
        KernelIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
        img,
        dIndices,
        dValues,
        dFilterWeights,
        dImgCoords,
        scalarWeightMultiplier
    );
}

void SetImagePixels(// Output
                    const ImageSpan& img,
                    // Input
                    const Span<const Spectrum>& dValues,
                    const Span<const Float>& dFilterWeights,
                    const Span<const ImageCoordinate>& dImgCoords,
                    // Constants
                    Float scalarWeightMultiplier,
                    const GPUQueue& queue)
{
    assert(dValues.size() == dFilterWeights.size());
    assert(dFilterWeights.size() == dImgCoords.size());
    using namespace std::string_view_literals;
    queue.IssueSaturatingKernel<KCSetImagePixels>
    (
        "KCSetImagePixels",
        KernelIssueParams{.workCount = static_cast<uint32_t>(dValues.size())},
        img,
        dValues,
        dFilterWeights,
        dImgCoords,
        scalarWeightMultiplier
    );
}

template<class Filter>
void ReconFilterGenericRGB(// Output
                           const ImageSpan& img,
                           // I-O
                           RayPartitioner& partitioner,
                           // Input
                           const Span<const Spectrum>& dValues,
                           const Span<const ImageCoordinate>& dImgCoords,
                           // Constants
                           Float scalarWeightMultiplier,
                           Float filterRadius,
                           Filter filter,
                           const GPUQueue& queue)
{
    // We use 32-bit morton code but last value 0xFFF..FF is reserved.
    // If image is 64k x 64k this system will not work (it will be rare but..)
    // Throw it that is the case
    if(img.Extent() == Vector2i(std::numeric_limits<uint16_t>::max()))
        throw MRayError("Unable to filter image size of 64k x 64k");

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
    // In order to not to wait the host result, we estimate SPP
    // as PPS.
    uint32_t averageSPP = maxPixelPerSample;

    // TODO: Currently, only warp-dedicated reduction.
    // If tracer pounds towards a specific region on the scene
    // (i.e. 4x4 pixel wide), Each warp will have ~2M / 16
    // amount of work to do. This is not optimal.
    // We need to create block and device variants of this reduction.
    //
    // According to a simple test, dedicating nearest amount of warps (rounded down)
    // was slower. Interestingly, for a 5x5 kernel, logical warp size of 1
    // (a thread) was the fastest. So dividing spp with this.
    static constexpr uint32_t WORK_PER_THREAD = 16;
    uint32_t logicalWarpSize = Math::DivideUp(averageSPP, WORK_PER_THREAD);
    logicalWarpSize = Math::PrevPowerOfTwo(logicalWarpSize);
    logicalWarpSize = std::min(logicalWarpSize, WarpSize());

    // Some boilerplate to make the code more readable
    auto KernelCall = [&]<auto Kernel>(std::string_view Name)
    {
        // Forgot why this is here? But probably to capture by copy?
        // TODO: Investigate
        Span<const Spectrum> dValuesIn = dValues;
        Span<const ImageCoordinate> dImgCoordsIn = dImgCoords;

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
            dValuesIn,
            dImgCoordsIn,
            // Constants
            Span<const uint32_t, 1>(hPartitionCount.data(), 1u),
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
    switch(logicalWarpSize - 1)
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
void ReconFilterGenericRGBAtomic(// Output
                                 const ImageSpan& img,
                                 // Input
                                 const Span<const Spectrum>& dValues,
                                 const Span<const ImageCoordinate>& dImgCoords,
                                 // Constants
                                 Float scalarWeightMultiplier,
                                 Float,
                                 Filter filter,
                                 const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    queue.IssueSaturatingKernel<KCFilterToImgAtomicRGB<Filter>>
    (
        "KCFilterToImgAtomicRGB"sv,
        KernelIssueParams{.workCount = static_cast<uint32_t>(dValues.size())},
        //
        img,
        //
        dValues,
        dImgCoords,
        //
        scalarWeightMultiplier,
        filter
    );
}

template<class Filter>
void MultiPassReconFilterGenericRGB(// Output
                                    const ImageSpan& img,
                                    // I-O
                                    RayPartitioner& partitioner,
                                    // Input
                                    const Span<const Spectrum>& dValues,
                                    const Span<const ImageCoordinate>& dImgCoords,
                                    // Constants
                                    uint32_t parallelHint,
                                    Float scalarWeightMultiplier,
                                    Float filterRadius,
                                    Filter filter,
                                    const GPUQueue& queue)
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
    uint32_t iterations = Math::DivideUp(totalPPS, parallelHint);
    uint32_t workPerIter = Math::DivideUp(totalWork, iterations);
    for(uint32_t i = 0; i < iterations; i++)
    {
        uint32_t start = workPerIter * i;
        uint32_t end = std::min(workPerIter * (i + 1), totalWork);
        uint32_t count = end - start;

        Span<const Spectrum> dLocalValues = dValues.subspan(start, count);
        Span<const ImageCoordinate> dLocalImgCoords = dImgCoords.subspan(start, count);
        ReconFilterGenericRGB(img, partitioner,
                              dLocalValues, dLocalImgCoords,
                              scalarWeightMultiplier, filterRadius,
                              filter, queue);
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
    using Math::DivideUp;
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
    static const std::string Name = ("GenerateMips"s + std::string(FilterType::ToString(E)));
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
                                                    const ImageSpan& img,
                                                    // I-O
                                                    RayPartitioner& partitioner,
                                                    // Input
                                                    const Span<const Spectrum>& dValues,
                                                    const Span<const ImageCoordinate>& dImgCoords,
                                                    // Constants
                                                    uint32_t parallelHint,
                                                    Float scalarWeightMultiplier,
                                                    const GPUQueue& queue) const
{
    static const auto annotation = queue.CreateAnnotation("Reconstruction Filter");
    const auto _ = annotation.AnnotateScope();

    MultiPassReconFilterGenericRGB(img, partitioner, dValues, dImgCoords,
                                   parallelHint, scalarWeightMultiplier,
                                   filterRadius, FF(filterRadius),
                                   queue);
}

template<FilterType::E E, class FF>
void TextureFilterT<E, FF>::ReconstructionFilterAtomicRGB(// Output
                                                          const ImageSpan& img,
                                                          // Input
                                                          const Span<const Spectrum>& dValues,
                                                          const Span<const ImageCoordinate>& dImgCoords,
                                                          // Constants
                                                          Float scalarWeightMultiplier,
                                                          const GPUQueue& queue) const
{
    static const auto annotation = queue.CreateAnnotation("Atomic Reconstruction Filter");
    const auto _ = annotation.AnnotateScope();

    ReconFilterGenericRGBAtomic(img, dValues, dImgCoords,
                                scalarWeightMultiplier,
                                filterRadius, FF(filterRadius),
                                queue);
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