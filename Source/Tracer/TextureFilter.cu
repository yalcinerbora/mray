#include "TextureFilter.h"
#include "RayPartitioner.h"
#include "DistributionFunctions.h"
#include "Filters.h"

#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgBinaryPartition.h"
#include "Device/GPUAlgRadixSort.h"

#include "Core/DeviceVisit.h"
#include "Core/GraphicsFunctions.h"

#ifdef MRAY_GPU_BACKEND_CUDA
    #include <cub/block/block_reduce.cuh>
#else
    #error "Only CUDA Version of Reconstruction filter is implemented"
#endif


#include "ColorFunctions.h"

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

// TODO: Should we dedicate a warp per pixel?
//template <uint32_t TPB, uint32_t LOGICAL_WARP_SIZE, class Filter>
//MRAY_KERNEL //MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)

template<class Filter>
MRAY_KERNEL //MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenerateMipmaps(// I-O
                       MRAY_GRID_CONSTANT const Span<MipArray<SurfViewVariant>> dSurfaces,
                       // Inputs
                       MRAY_GRID_CONSTANT const Span<const MipGenParams> dMipGenParamsList,
                       // Constants
                       MRAY_GRID_CONSTANT const uint32_t currentMipLevel,
                       MRAY_GRID_CONSTANT const Vector2ui spp,
                       MRAY_GRID_CONSTANT const Filter FilterFunc)
{
    assert(dSurfaces.size() == dMipGenParamsList.size());
    Vector2 dXY = Vector2(1) / Vector2(spp);

    KernelCallParams kp;
    MRAY_SHARED_MEMORY MipGenParams curParams;
    MRAY_SHARED_MEMORY SurfViewVariant curSurfs[2];
    // Block-stride loop
    uint32_t textureCount = static_cast<uint32_t>(dSurfaces.size());
    for(uint32_t tI = kp.blockId; tI < textureCount; tI += kp.gridSize)
    {
        if(kp.threadId == 0) curParams.validMips = dMipGenParamsList[tI].validMips;
        if(kp.threadId == 1) curParams.mipCount = dMipGenParamsList[tI].mipCount;
        if(kp.threadId == 2) curParams.mipZeroRes[0] = dMipGenParamsList[tI].mipZeroRes[0];
        if(kp.threadId == 3) curParams.mipZeroRes[1] = dMipGenParamsList[tI].mipZeroRes[1];
        // Load the surfaces to shared memory (maybe faster to access?)
        if(kp.threadId == 4) curSurfs[0] = dSurfaces[tI][currentMipLevel];
        if(kp.threadId == 5) curSurfs[1] = dSurfaces[tI][currentMipLevel - 1];

        BlockSynchronize();

        SurfViewVariant curT = curSurfs[0];
        SurfViewVariant prevT = curSurfs[1];
        Vector2ui mipRes = Graphics::TextureMipSize(curParams.mipZeroRes,
                                                    currentMipLevel);
        Vector2ui parentRes = Graphics::TextureMipSize(curParams.mipZeroRes,
                                                       currentMipLevel - 1);
        uint32_t totalPix = mipRes.Multiply();

        // Skip this mip if it is already loaded.
        // This may happen when a texture has up to x amount of mips
        // but it can support log2(floor(max(res))) amount so we generate
        // these mipmaps.
        // Since we issue mip generation in bulk (meaning for all of the textures
        // mip generate level X will be called regardless of that texture has a valid
        // mip). We need to early exit for a texture that can not support that level of mip.
        // If variant is in monostate we skip this mip level generation
        if(curParams.validMips[currentMipLevel] ||
           std::holds_alternative<std::monostate>(curT)) continue;

        // Loop over the pixel by block
        for(uint32_t pixI = kp.threadId; pixI < totalPix; pixI += kp.blockSize)
        {
            Vector2ui pixCoordInt = Vector2ui(pixI % mipRes[0], pixI / mipRes[0]);
            Vector2 pixCoord = (Vector2(pixCoordInt[0], pixCoordInt[1]) +
                                Vector2(0.5));
            // We use float as a catch-all type
            // It is allocated as a max channel
            Vector4 writePix = Vector4::Zero();
            // Stochastically sample the up level via the filter
            // Mini Monte Carlo..
            for(uint32_t sppY = 0; sppY < spp[1]; sppY++)
            for(uint32_t sppX = 0; sppX < spp[0]; sppX++)
            {
                // Create a quasi sampler by perfectly stratifying the
                // sample space
                Vector2 xi = dXY * Vector2(sppX, sppY);
                SampleT<Vector2> sample = FilterFunc.Sample(xi);
                //SampleT<Vector2> sample = {Vector2::Zero(), Float(1)};
                // Find the upper level coordinate
                Vector2 readPixCoord = (pixCoord + sample.value).RoundSelf();
                readPixCoord *= Vector2(2);
                readPixCoord.ClampSelf(Vector2::Zero(), Vector2(parentRes - 1));
                Vector2ui readPixCoordInt = Vector2ui(readPixCoord);
                Vector4 localPix = DeviceVisit(std::as_const(prevT),
                [readPixCoordInt](auto&& readSurf) -> Vector4
                {
                    Vector4 result = Vector4::Zero();
                    using VariantType = std::remove_cvref_t<decltype(readSurf)>;
                    if constexpr(!std::is_same_v<VariantType, std::monostate>)
                    {
                        using ReadType = typename VariantType::Type;
                        constexpr uint32_t C = VariantType::Channels;
                        ReadType rPix = readSurf(readPixCoordInt);
                        //ReadType rPix = readSurf(pixCoordInt);

                        // We need to manually loop over the channels here
                        if constexpr(C != 1)
                        {
                            UNROLL_LOOP
                            for(uint32_t c = 0; c < C; c++)
                                result[c] = static_cast<Float>(rPix[c]);
                        }
                        else result[0] = static_cast<Float>(rPix);
                    }
                    return result;
                });
                writePix += localPix / sample.pdf;
            }
            // Divide by the sample count
            writePix *= dXY.Multiply();

            //writePix = Vector4(Color::RandomColorRGB(pixI), 1);


            // Finally write the pixel
            DeviceVisit(curT,
            [writePix, pixCoordInt](auto&& writeSurf) -> void
            {
                using VariantType = std::remove_cvref_t<decltype(writeSurf)>;
                if constexpr(!std::is_same_v<VariantType, std::monostate>)
                {
                    using WriteType = typename VariantType::Type;
                    constexpr uint32_t C = VariantType::Channels;
                    WriteType writeVal;
                    if constexpr(C != 1)
                    {
                        using InnerT = typename WriteType::InnerType;
                        UNROLL_LOOP
                        for(uint32_t c = 0; c < C; c++)
                            writeVal[c] = static_cast<InnerT>(writePix[c]);
                    }
                    else writeVal = static_cast<WriteType>(writePix[0]);
                    writeSurf(pixCoordInt) = writeVal;
                }
            });
        }
        // Wait for next issue,
        // This block synchronization is about surface data dependency
        BlockSynchronize();
    }
}


MRAY_KERNEL //MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCExpandSamplesToPixels(// Outputs
                             MRAY_GRID_CONSTANT const Span<uint32_t> dPixelIds,
                             MRAY_GRID_CONSTANT const Span<uint32_t> dIndices,
                             // Inputs
                             MRAY_GRID_CONSTANT const Span<const Vector2> dImgCoords,
                             // Constants
                             MRAY_GRID_CONSTANT const Float filterRadius,
                             MRAY_GRID_CONSTANT const uint32_t maxPixelPerSample,
                             MRAY_GRID_CONSTANT const Vector2i imgResolution)
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

            bool pixOutside = (globalPixCoord[0] < 0 ||
                               globalPixCoord[0] >= imgResolution[0] ||
                               globalPixCoord[1] < 0 ||
                               globalPixCoord[1] >= imgResolution[1]);
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
MRAY_KERNEL //MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCFilterToImgWarpRGB(MRAY_GRID_CONSTANT const SubImageSpan<3> img,
                          // Inputs per segment
                          MRAY_GRID_CONSTANT const Span<const uint32_t> dStartOffsets,
                          MRAY_GRID_CONSTANT const Span<const uint32_t> dPixelIds,
                          // Inputs per thread
                          MRAY_GRID_CONSTANT const Span<const uint32_t> dIndices,
                          // Inputs Accessed by SampleId
                          MRAY_GRID_CONSTANT const Span<const Vector3> dValues,
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
                Vector3 sampleVal = dValues[readIndex];
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

            Vector3 pixValue = img.FetchPixelBulk(pixCoordsInt);
            img.StorePixelBulk(pixValue + Vector3(totalValue),
                               pixCoordsInt);
        }
    }
}

template<class Filter>
void ReconFilterGenericRGB(// Output
                           const SubImageSpan<3>& img,
                           // I-O
                           RayPartitioner& partitioner,
                           // Input
                           const Span<const Vector3>& dValues,
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
        img.Resolution()
    );

    using namespace BitFunctions;
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
        Span<const Vector3> dX = dValues;
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
                                    const SubImageSpan<3>& img,
                                    // I-O
                                    RayPartitioner& partitioner,
                                    // Input
                                    const Span<const Vector3>& dValues,
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

        Span<const Vector3> dLocalValues = dValues.subspan(start, count);
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

    // TODO: First we need RNG, we are generating multiple mipmaps of multiple
    // textures in bulk so quality RNG will be too costly. Currently
    // this is initialization time tool, when we provide this as cmd app
    // we need to change this.
    //
    // We will dedicate a single large block(1024) for each texture.
    // The reason for large texture is to allocate a single block on an SM,
    // and provide the entire L1 cache (L1 caches are per SM afaik) to the
    // block. We assume the mip generation is a memory bound operation.
    constexpr uint32_t THREAD_PER_BLOCK = 512;
    // To combine the both, we will dedicate a single RNG per thread.
    // This is why space-filling RNG (Sobol etc.) are not used because
    // those will not make sense when each thread will be responsible for
    // multiple pixels. We can reset use multiple dimensions whatnot but
    // even still generating a sobol sample is costly
    // (for a single number 32xLUT).

    // Find maximum block count for state allocation
    static constexpr Vector2ui SPP = Vector2ui(8, 8);
    uint32_t blockCount = queue.RecommendedBlockCountDevice(&KCGenerateMipmaps<Filter>,
                                                            THREAD_PER_BLOCK, 0);
    // We can temporarily allocate here since this will be done at
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
        using namespace std::string_view_literals;
        queue.IssueExactKernel<KCGenerateMipmaps<Filter>>
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
            filter
        );
        //break;
    }
    queue.Barrier().Wait();
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
    GenerateMipsGeneric(textures, params, gpuSystem, FF(filterRadius));
}

template<FilterType::E E, class FF>
void TextureFilterT<E, FF>::ReconstructionFilterRGB(// Output
                                                    const SubImageSpan<3>& img,
                                                    // I-O
                                                    RayPartitioner& partitioner,
                                                    // Input
                                                    const Span<const Vector3>& dValues,
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

template TextureFilterT<FilterType::BOX, BoxFilter>;
template TextureFilterT<FilterType::TENT, TentFilter>;
template TextureFilterT<FilterType::GAUSSIAN, GaussianFilter>;
template TextureFilterT<FilterType::MITCHELL_NETRAVALI, MitchellNetravaliFilter>;