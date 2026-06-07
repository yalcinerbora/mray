#include "TextureFilter.h"
#include "RayPartitioner.h"
#include "DistributionFunctions.h"
#include "Filters.h"
#include "GenericTextureRW.h"

#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgBinaryPartition.h"
#include "Device/GPUAlgRadixSort.h"

#include "Core/ColorFunctions.h"
#include "Core/GraphicsFunctions.h"

#include <numeric>

#ifdef MRAY_GPU_BACKEND_CUDA
    #include <cub/warp/warp_reduce.cuh>
#elif defined MRAY_GPU_BACKEND_HIP
    #include <hipcub/warp/warp_reduce.hpp>
#endif

#include "Core/DataStructures.h"

// We have multiple options filter images (too many in fact)
// We even have a stochastic filterting option to filtering.
//
// Generic options below does not seperate the 2D kernels
// since most of the filtering code was about filtering the
// resulting image, not input texture filtering.
//
// So all the options below is slow for mip generation etc.
//
// On my laptop, downsampling 104 textures (all 4096x4096, half R8, half RGB8)
// did took 70 seconds with all threads active (20 threads). Which is was not proper
// (GPU eats it like cany though it was negligable, amortized during loading).
//
// So instead we do proper seperable kernel computation for mipmaps.
// for downsampling we do the same pyramid scheme but in local memory.

//
enum class FilterMode
{
    SAMPLING,
    ACCUMULATE
};

// Static kernel sized and pre-generated
// downsampling filters.
//
// All the data is compile time generated.
//
// We hope these data will be embedded to immediate values
// if possible (similar to the color conversion routines).
//
// Filter is the filter type
template<class Filter2D>
struct KernelWeightsStatic1D
{
    static constexpr Float RADIUS = Filter2D::IDEAL_RADIUS;
    static constexpr uint32_t DIAMETER = uint32_t(Math::Ceil(2 * RADIUS));
    //
    static constexpr uint32_t PADDING = (DIAMETER) - 1;
    static constexpr uint32_t TOTAL_WIDTH = 2 * DIAMETER;

    static constexpr Float DELTA = Float(2) * RADIUS / Float(TOTAL_WIDTH);
    static constexpr Float SAMPLE_START = -RADIUS + DELTA / Float(2);

    using FloatArray = Array<Float, TOTAL_WIDTH>;
    using Filter = typename Filter2D::Filter1D;

    // First time properly using constexpr math functionality
    // of mray (thanks to njuffa!), whish me luck!
    static constexpr FloatArray GenerateWeights()
    {
        FloatArray offsets = {};
        for(uint32_t i = 0; i < TOTAL_WIDTH; i++)
            offsets[i] = SAMPLE_START + Float(i) * DELTA;

        FloatArray result = {};
        Filter filter = Filter(RADIUS);
        Float sum = Float(0);
        for(uint32_t i = 0; i < TOTAL_WIDTH; i++)
        {
            result[i] = filter.Evaluate(offsets[i]);
            sum += result[i];
        }
        for(uint32_t i = 0; i < TOTAL_WIDTH; i++)
            result[i] /= sum;
        return result;
    }
};

//using YY = KernelWeightsStatic1D<BoxFilter>;
//static constexpr auto P = YY::PADDING;
//static constexpr auto D = YY::DIAMETER;
//static constexpr auto R = YY::RADIUS;
//
//static constexpr auto UUUU = std::ceil(2 * 0.5);
//static constexpr auto YYY = Math::Ceil(2 * 0.5);
//static constexpr auto YYYSSS = 1 + Math::PrevFloat(1.0f);
//
//constexpr Float XAX(Float num)
//{
//    Float f = Float(uint32_t(num));
//    return f + (f < num);
//}
//
//static constexpr Float F = XAX(1e8);



// We do dirt cheap (kinda) sampling over the filtering kernel
// for image clamping. Since it is a temporary method
// to reduce resolution of the frame during load time.
//
// Unlike mipmap generation, it is not respected as much
// so we stochastically sample the filter kernel (SX x SY)
// over the radius R (R is in downsampled image coordinates).
//
// Compile time kernels
// size and since we can't use temp memory (if user set this parameter
// scene do not fit to the memory so...
//
template<class Filter2D, uint32_t SPP>
struct KernelWeightsStochasticStatic1D
{
    static constexpr Float RADIUS = Filter2D::IDEAL_RADIUS;
    static constexpr uint32_t TOTAL_WIDTH = 2 * SPP;

    static constexpr Float DELTA = Float(2) * RADIUS / Float(TOTAL_WIDTH);
    static constexpr Float SAMPLE_START = -RADIUS + DELTA / Float(2);

    using FloatArray = Array<Float, TOTAL_WIDTH>;
    using Filter = typename Filter2D::Filter1D;

    //static constexpr Vector2 GenerateSampleAndXi(uint32_t i)
    //{
    //    Filter filter = Filter(RADIUS);
    //    // "o" is radius relative equally spaced locations
    //    Float o = SAMPLE_START + Float(i) * DELTA;
    //    // Move to sample space [0, 1)
    //    Float xi = (o + (RADIUS)) / (2 * RADIUS);
    //    //
    //    //return Vector2(xi, filter.Sample(xi).value);
    //    return Vector2(xi, xi);
    //}

    static constexpr FloatArray GenerateSamples()
    {
        Filter filter = Filter(RADIUS);
        FloatArray samples = {};
        for(uint32_t i = 0; i < TOTAL_WIDTH; i++)
        {
            // "o" is radius relative equally spaced locations
            Float o = SAMPLE_START + Float(i) * DELTA;
            // Move to sample space [0, 1)
            Float xi = (o + (RADIUS)) / (2 * RADIUS);
            //
            samples[i] = filter.Sample(xi).value;

            // For box filter this sampling etc should be wasteful
            // so check for mistake
            if constexpr(std::is_same_v<Filter, BoxFilter1D>)
                assert(o == samples[i]);
        }

        return samples;
    }

    static constexpr uint32_t FindPadding()
    {
        FloatArray samples = GenerateSamples();
        // It is symmetric so just find max
        // Not optimal code but it is constexpr
        uint32_t result = uint32_t(0);
        for(uint32_t i = 0; i < TOTAL_WIDTH; i++)
        {
            Float s = samples[i];
            if(s < 0.0f) continue;
            result = Math::Max(result, uint32_t(Math::Ceil(s)));
        }
        return result;
    }


    // First time properly using constexpr math functionality
    // of mray (thanks to njuffa!), whish me luck!
    static constexpr FloatArray GenerateWeights()
    {
        FloatArray offsets = GenerateSamples();

        FloatArray result = {};
        Filter filter = Filter(RADIUS);
        Float sum = Float(0);
        for(uint32_t i = 0; i < TOTAL_WIDTH; i++)
        {
            result[i] = filter.Evaluate(offsets[i]);
            result[i] /= filter.Pdf(offsets[i]);
            sum += result[i];
        }
        for(uint32_t i = 0; i < TOTAL_WIDTH; i++)
            result[i] /= sum;
        return result;
    }


    static constexpr FloatArray RelativeRange(FloatArray x, uint32_t factor)
    {
        for(uint32_t i = 0; i < TOTAL_WIDTH; i++)
        {
            x[i] *= Float(factor);
        }
        return x;
    }

    static constexpr FloatArray ClampRange(FloatArray x)
    {

        for(uint32_t i = 0; i < TOTAL_WIDTH; i++)
        {
            Float val = x[i];
            x[i] = (val < Float(0)) ? Math::Floor(val) : Math::Ceil(val);
        }
        return x;
    }
};

////using KK = KernelWeightsStochasticStatic1D<TentFilter, 4>;
////using KK = KernelWeightsStochasticStatic1D<MitchellNetravaliFilter, 8>;
//using KK = KernelWeightsStochasticStatic1D<GaussianFilter, 8>;
////using KK = KernelWeightsStochasticStatic1D<TentFilter, 128>;
//static constexpr auto PAD = KK::FindPadding();
//static constexpr auto SAMPLES = KK::GenerateSamples();
//static constexpr auto RANGE = KK::RelativeRange(SAMPLES, 3);
//static constexpr auto PIXES = KK::ClampRange(RANGE);
//static constexpr auto WEIGHTS = KK::GenerateWeights();



static constexpr uint32_t INVALID_MORTON = std::numeric_limits<uint32_t>::max();
// Be careful changing these, these may have bank conflict implications
// on some of the kernels below.
static constexpr Vector2ui KC_CLAMP_IMAGE_TILE_SIZE = Vector2ui(32, 16);
static constexpr Vector2ui KC_MIPMAP_GEN_TILE_SIZE = Vector2ui(32, 16);

MR_HF_DECL
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
    int32_t quot = static_cast<int32_t>(filterRadius / Float(0.5));
    Float remainder = Math::FMod(filterRadius, Float(0.5));
    // Exact divisions reside on previous segment
    if(remainder == Float(0)) quot -= 1;

    result += (quot + 1);
    return result;
}

MR_HF_DECL
Vector2i FilterRadiusPixelRange(int32_t wh)
{
    Vector2i range(-(wh - 1) / 2, (wh + 2) / 2);
    return range;
}

template<class Filter, class DataFetcher>
MR_GF_DECL
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
        // Do the integration separately as well
        // we need to compensate
        weightSum += weight * totalSampleInv / pdf;
    }
    writePix /= weightSum;
    return writePix;
}

MR_GF_DECL
Tuple<Vector3, Float> ConvertNaNsToColor(Spectrum value, Float weight)
{
    // Clang-HIP bug on CTAD maybe (or probably my bug on deduction guide)?
    using RetT = Tuple<Vector3, Float>;

    if(!Math::IsFinite(value))
        return RetT(BIG_MAGENTA(), weight * Float(128.0));
    else
        return RetT(Vector3(value), weight);
}

// We do refain from using "ifdef" to segregate CPU and GPU code
// inside a function block. Or we try to eliminiate some discrapencies
// between CPU-GPU directly via code. For example:
//
//  - CPU do not have BlockSynchronize() so it is just an empty function.
//    we hope compiler to eliminiate that code (it certainly will).
//
//  - The code base heavily relies on grid-stride block-stride loops
//    so that you can customize the kernel bounds and try to dedicate
//    a single SM for a certain kernel so that you can concurrently
//    launch multiple kernels that does different things.
//
//    Such loops are logically eliminated for CPU via giving large enough
//    grid size to do an iteration over that loop just once. These are
//    definately will not be eliminated (so we lose two comparisons
//    and a potential branch if compiler is dumb). Since main focus
//    of this renderer is not CPU rendering, we did not change the all
//    of the kernels to remove the loops and rely on the driver.
//    That refactoring work is not worth for solo development atm.
//    Hopefully, in future the design of the kernel calls would change to
//    express GPU/CPU code.
//
//  - And lastly these kind of functions (in this case these and the color converter
//    kernels) have static load balancing which does not require the texture sizes
//    to be known ahead of time, so you can launch a single kernel that can handle
//    many texture types (via switch case generation) and many texture dimensions
//    (via statically allcating blocks per texture).
//
// However; this comes with a cost in terms of wasted loops over nothing in block-wide
// scale that we choose to accept due to simplicity/maintainability of the code.
//
// For CPU, case 2 and case 3 (this case) clashes and we essentially doing a heavy
// strided access[1] which tanks the performance.
//
// All in all, in this rare case we use #ifdef over the loops and completely change
// the control flow of the code wrt. CPU/GPU. Sorry...
//
//
// [1] Loops in question is:
//
//  for(uint32_t tileI = localBI; tileI < totalTiles.Multiply();
//      tileI += blockPerTexture)
//
// ========================================= //
//   Statically Weighted Mipmap Generation   //
// ========================================= //
// Hightly performant mipmap gen with caveats:
//  - Statically generated 4x4 taps
//  - No phase adjustment (aka. image may crawl towards "top right" due to
//                         odd number of width / height on a given mip).
//  - Added phase adjustment as a slow path but it does not work atm :(
//    Could not find the bug.
//
// It is generic single pass kernel for all textures in the scene.
// Kernel is called for N times where N is the highest mip level of a texture.
// Also it does not use transient memory besides the shared memory, so has constant
// memory complexity (Except for CPU-side, it uses 32 x 16 x 4 32-bit floats as a
// transient memory).
template<uint32_t TPB, class FilterKernel>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCGenerateMipmapsStatic(// I-O
                             MRAY_GRID_CONSTANT const Span<MipArray<TracerSurfView>> dSurfaces,
                             // Inputs
                             MRAY_GRID_CONSTANT const Span<const MipGenParams> dMipGenParamsList,
                             // Constants
                             MRAY_GRID_CONSTANT const uint32_t currentMipLevel,
                             MRAY_GRID_CONSTANT const uint32_t blockPerTexture)
{
    using Kernel = FilterKernel;
    using Filter = Kernel::Filter;
    Filter runtimeFilter = Filter(Kernel::RADIUS);

    static constexpr auto KERNEL_WEIGHTS = Kernel::GenerateWeights();
    static constexpr Vector2ui TILE_SIZE = KC_MIPMAP_GEN_TILE_SIZE;
    static_assert(TILE_SIZE.Multiply() == TPB);
    assert(dSurfaces.size() == dMipGenParamsList.size());

    // This is expanded since we need 2x more pixels
    static constexpr uint32_t SHMEM_Y_SIZE   = 2 * (Kernel::PADDING + KC_MIPMAP_GEN_TILE_SIZE[1]);
    static constexpr Vector2ui SHMEM_SIZE_2D = Vector2ui(TILE_SIZE[0], SHMEM_Y_SIZE);
    static constexpr uint32_t SHMEM_SIZE_1D  = SHMEM_SIZE_2D.Multiply();
    // TODO: Bank conflicts
    // Currently there are no bank conflicts, because TPB = 512
    // and tile size is 32x16. Thus we allocate 32x34 sized shmem.
    // Each thread will write the results of X convolve in row-major order.
    // each warp will be responsible for a single row.
    //
    // On Y convolve case, again a warp will handle a single row.
    // And each warp will read from single row.
    //
    // Actual TODO part is to generate offsets from "TILE_SIZE"
    // comptime and add here as offset if needed.
    MRAY_SHARED_MEMORY Float sLocalPixelsR[SHMEM_SIZE_1D];
    MRAY_SHARED_MEMORY Float sLocalPixelsG[SHMEM_SIZE_1D];
    MRAY_SHARED_MEMORY Float sLocalPixelsB[SHMEM_SIZE_1D];
    MRAY_SHARED_MEMORY Float sLocalPixelsA[SHMEM_SIZE_1D];

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
        TracerSurfView writeSurf = dSurfaces[tI][currentMipLevel];
        const TracerSurfView readSurf = dSurfaces[tI][currentMipLevel - 1];
        //
        Vector2ui mipRes = Graphics::TextureMipSize(curParams.mipZeroRes,
                                                    currentMipLevel);
        Vector2ui parentRes = Graphics::TextureMipSize(curParams.mipZeroRes,
                                                       currentMipLevel - 1);
        Vector2 scale = Vector2(parentRes) / Vector2(mipRes);
        auto KernelPhaseOffset = [](uint32_t globalXOrY, Float scale)
        {
            Float texel = (Float(globalXOrY) + Float(0.5)) * scale;
            Float result = texel - Math::Floor(texel);
            // TODO: Hack, we shift the phase offset towards center
            // since the phase period is most 1 pixel wide.
            // (Maybe two if both image sizes are odd?)
            //
            // So instead of drifting towards "top-right"
            // we drift "inside-out"
            //
            // Could not fix this unfortunately :(
            result -= Float(0.5);
            return result;
        };
        // Check if we require to use unaligned filters
        // (meanin we will caclulate them on the fly
        bool isEvenX = ((mipRes[0] & 0x1) == 0 && (parentRes[0] & 0x1) == 0);
        bool isEvenY = ((mipRes[1] & 0x1) == 0 && (parentRes[1] & 0x1) == 0);
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
            if(tileI != localBI) BlockSynchronize();

            Vector2ui tileI2D = Vector2ui(tileI % totalTiles[0],
                                          tileI / totalTiles[0]);
            // Assuming tile size of is 32x16, we need to access 66x34 pixels
            // Given the kernel is 4x4. This is too much memory when texture is
            // 4 channel (We could've written a separate kernel for each channel
            // here we use single kernel for compile times and maintainability).
            //
            // So we duplicate the reads of x-axis and directly convolve the
            // pixels to the shared memory. So we only need 32x34 pixel region.
            // Which is comfortable to fit to shared memory
            //
            // Iteration is per shared-memory pixel and it is stored in
            // row-major order.
            #ifdef MRAY_GPU_BACKEND_CPU
                // CPU thread loads all on first iteration. No notion of shared memory.
                if(kp.threadId == 0)
                for(uint32_t i = 0; i < SHMEM_SIZE_1D; i++)
            #else
                // Load cooperatively
                for(uint32_t i = kp.threadId; i < SHMEM_SIZE_1D; i += kp.blockSize)
            #endif
            {
                // 2D Version of our buffer
                Vector2i parentLocalTexel = Vector2i(i % SHMEM_SIZE_2D[0],
                                                     i / SHMEM_SIZE_2D[0]);
                // We need to calculate couple pixels extra so
                parentLocalTexel[1] -= Kernel::PADDING;
                // Now convert it to it be
                // X dimension is in mip's coordinates, convert it
                parentLocalTexel[0] *= 2;
                //
                Vector2i parentGlobalTexel = parentLocalTexel;
                parentGlobalTexel += Vector2i(tileI2D * TILE_SIZE * 2);

                //
                // If there are size mismatch, we need to align the kernel weights
                int32_t mipLocalTexelX = (i % TILE_SIZE[0]);
                int32_t mipGlobalTexelX = mipLocalTexelX;
                mipGlobalTexelX += int32_t(tileI2D[0] * TILE_SIZE[0]);
                Float phaseOffsetX = Float(0);
                if(!isEvenX) phaseOffsetX = KernelPhaseOffset(mipGlobalTexelX, scale[0]);

                // This is the top left pixel of the region we try to convolve
                // "parentGlobalTexel + 0.5" is the exact location that we will
                // convolve upon.
                // Convolve X
                Vector4 pixOut = Vector4::Zero();
                Float filterSumX = Float(0);
                MRAY_UNROLL_LOOP
                for(int32_t j = 0; j < int32_t(Kernel::TOTAL_WIDTH); j++)
                {
                    int32_t offset = j - int32_t(Kernel::PADDING);
                    Vector2i t = Vector2i(parentGlobalTexel[0] + offset, parentGlobalTexel[1]);
                    // TODO: Expose other out of bound methods later
                    Vector2i cT = Math::Clamp(t, Vector2i::Zero(), Vector2i(parentRes - 1));
                    Vector4 pixIn = GenericRead(Vector2ui(cT), readSurf);
                    // Fast path
                    if(isEvenX)
                    {
                        using Math::FMA;
                        pixOut[0] = FMA(pixIn[0], KERNEL_WEIGHTS[j], pixOut[0]);
                        pixOut[1] = FMA(pixIn[1], KERNEL_WEIGHTS[j], pixOut[1]);
                        pixOut[2] = FMA(pixIn[2], KERNEL_WEIGHTS[j], pixOut[2]);
                        pixOut[3] = FMA(pixIn[3], KERNEL_WEIGHTS[j], pixOut[3]);
                    }
                    else
                    {
                        Float o = (Float(j) + phaseOffsetX) * Kernel::DELTA;
                        o += Kernel::SAMPLE_START;
                        //
                        Float weight = runtimeFilter.Evaluate(o);
                        filterSumX += weight;
                        using Math::FMA;
                        pixOut[0] = FMA(pixIn[0], weight, pixOut[0]);
                        pixOut[1] = FMA(pixIn[1], weight, pixOut[1]);
                        pixOut[2] = FMA(pixIn[2], weight, pixOut[2]);
                        pixOut[3] = FMA(pixIn[3], weight, pixOut[3]);
                    }
                }
                if(!isEvenX)
                {
                    Float sumRecip = Float(1) / filterSumX;
                    pixOut[0] *= sumRecip;
                    pixOut[1] *= sumRecip;
                    pixOut[2] *= sumRecip;
                    pixOut[3] *= sumRecip;
                }
                // Write!
                sLocalPixelsR[i] = pixOut[0];
                sLocalPixelsG[i] = pixOut[1];
                sLocalPixelsB[i] = pixOut[2];
                sLocalPixelsA[i] = pixOut[3];
            }

            //
            BlockSynchronize();

            #ifdef MRAY_GPU_BACKEND_CPU
                // All the work is done by the first thread
                // For CPU kernel call, we will do 1 thread per block
                //
                // These threads are logical each actual OS thread
                // works on block, the kernel functions called inside
                // a loop for each "thread".
                if(kp.threadId == 0)
                for(uint32_t i = 0; i < TILE_SIZE.Multiply(); i++)
            #else
                // Same thing but we exactly have enough threads in a block
                // so no loops
                uint32_t i = kp.threadId;
            #endif
            {
                Vector2i mipLocalTexel = Vector2i(i % TILE_SIZE[0],
                                                  i / TILE_SIZE[0]);
                Vector2i mipGlobalTexel = mipLocalTexel;
                mipGlobalTexel += Vector2i(tileI2D * TILE_SIZE);

                // Do not bother convolution if pixel is out of range
                if(mipGlobalTexel[0] >= int32_t(mipRes[0]) || mipGlobalTexel[1] >= int32_t(mipRes[1]))
                    continue;

                Float phaseOffsetY = Float(0);
                if(!isEvenY) phaseOffsetY = KernelPhaseOffset(mipGlobalTexel[1], scale[1]);

                // Convert mip local texel to shared memory texel
                Vector2i shMemTexel = mipLocalTexel;
                shMemTexel[1] *= int32_t(2);
                shMemTexel[1] += Kernel::PADDING;

                Vector4 pixOut = Vector4::Zero();
                Float filterSumY = Float(0);
                MRAY_UNROLL_LOOP
                for(int32_t j = 0; j < int32_t(Kernel::TOTAL_WIDTH); j++)
                {
                    int32_t offset = j - int32_t(Kernel::PADDING);
                    Vector2i localReadTexel = Vector2i(shMemTexel[0], shMemTexel[1] + offset);
                    int32_t sI = localReadTexel[1] * SHMEM_SIZE_2D[0] + localReadTexel[0];
                    Float r = sLocalPixelsR[sI];
                    Float g = sLocalPixelsG[sI];
                    Float b = sLocalPixelsB[sI];
                    Float a = sLocalPixelsA[sI];
                    // Fast Path
                    if(isEvenY)
                    {
                        using Math::FMA;
                        pixOut[0] = FMA(r, KERNEL_WEIGHTS[j], pixOut[0]);
                        pixOut[1] = FMA(g, KERNEL_WEIGHTS[j], pixOut[1]);
                        pixOut[2] = FMA(b, KERNEL_WEIGHTS[j], pixOut[2]);
                        pixOut[3] = FMA(a, KERNEL_WEIGHTS[j], pixOut[3]);
                    }
                    else
                    {
                        Float o = (Float(j) + phaseOffsetY) * Kernel::DELTA;
                        o += Kernel::SAMPLE_START;
                        //
                        Float weight = runtimeFilter.Evaluate(o);
                        filterSumY += weight;
                        using Math::FMA;
                        pixOut[0] = FMA(r, weight, pixOut[0]);
                        pixOut[1] = FMA(g, weight, pixOut[1]);
                        pixOut[2] = FMA(b, weight, pixOut[2]);
                        pixOut[3] = FMA(a, weight, pixOut[3]);
                    }
                }
                if(!isEvenY)
                {
                    Float sumRecip = Float(1) / filterSumY;
                    pixOut[0] *= sumRecip;
                    pixOut[1] *= sumRecip;
                    pixOut[2] *= sumRecip;
                    pixOut[3] *= sumRecip;
                }
                // Finally write the pixel
                GenericWrite(writeSurf, pixOut, Vector2ui(mipGlobalTexel));
            }
        }
    }
}

// ========================================= //
//       Statically-Weighted Stochastic      //
//              Image Clamping               //
// ========================================= //
//
// Similar to static mipmap gen kernel, this kernel has its weights
// potentially embeded to the instructions (if compiler prefers).
//
// Since we want simple / fast (unfortunately quite bad quality
// when downsample ratio is quite high) downsample during load time
// and this routine is a dirty fix for scene which high amount of textures
// to fit to the memory, we stochastically sample the image NxN samples and
// apply the filter.
//
// This is quite fast for high downsample ratios but is **slower** if you
// downsample 1/2. Since we do quadratic operations. Sample pattern is generated
// via stratification and filters are importance sampled with themselves (except for
// Mitchell-Netravali where we use 3x guassians with MIS, later I'd like to change it
// to an approximate to itself via a polynomial, since its sampling also used as recon
// filter of the output images of the renderer.)
template<uint32_t TPB, class StochasticFilterKernel>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCClampImageStatic(// Output
                        MRAY_GRID_CONSTANT const TracerSurfView surfaceOut,
                        // Inputs
                        MRAY_GRID_CONSTANT const Span<const Byte> dBufferImage,
                        // Constants
                        MRAY_GRID_CONSTANT const Vector2ui outputImageRes,
                        MRAY_GRID_CONSTANT const Vector2ui inputImageRes)
{
    using Kernel = StochasticFilterKernel;
    static constexpr Vector2ui TILE_SIZE = KC_CLAMP_IMAGE_TILE_SIZE;
    static_assert(TILE_SIZE.Multiply() == TPB);

    auto ReadPixel = [&](Vector2 rPixCoord) -> Vector4
    {
        // Find the upper level coordinate
        using Graphics::ConvertPixelIndices;
        rPixCoord = ConvertPixelIndices(rPixCoord,
                                        Vector2(inputImageRes),
                                        Vector2(outputImageRes));
        Vector2ui rPixCoordInt = Vector2ui(Math::Round(rPixCoord));
        // Data is tightly packed, we can directly find the lienar index
        uint32_t pixCoordLinear = (rPixCoordInt[1] * inputImageRes[0] +
                                   rPixCoordInt[0]);

        // Now the type fetch part, utilize surface variant to
        // find the type
        Vector4 outData = GenericReadFromBuffer(dBufferImage, surfaceOut,
                                                pixCoordLinear);
        return outData;
    };

    KernelCallParams kp;
    // Loop over the tiles for this tex, each block is dedicated to a tile
    Vector2ui totalTiles = Math::DivideUp(outputImageRes, TILE_SIZE);
    for(uint32_t tileI = kp.blockId; tileI < totalTiles.Multiply();
        tileI += kp.gridSize)
    {

        #ifdef MRAY_GPU_BACKEND_CPU
            // This code is slow lets try this
            if(kp.threadId == 0)
            for(uint32_t pI = 0; pI < TILE_SIZE.Multiply(); pI++)
            #else
            // Load cooperatively
            uint32_t pI = kp.threadId;
        #endif
        {
            Vector2ui localPI = Vector2ui(pI % TILE_SIZE[0],
                                          pI / TILE_SIZE[0]);
            Vector2ui tile2D = Vector2ui(tileI % totalTiles[0],
                                         tileI / totalTiles[0]);
            Vector2ui wPixCoordInt = tile2D * TILE_SIZE + localPI;
            //
            if(wPixCoordInt[0] >= outputImageRes[0] ||
               wPixCoordInt[1] >= outputImageRes[1])
                continue;

            // Here read the data
            Vector4 writePix = Vector4::Zero();
            Vector2 wPixCoord = Vector2(wPixCoordInt);
            for(uint32_t j = 0; j < Kernel::Dim[1]; j++)
            for(uint32_t i = 0; i < Kernel::Dim[0]; i++)
            {
                Vector2ui ij = Vector2ui(i, j);
                Vector2 o = Kernel::Offsets[j][i];
                Float w = Kernel::Weights[j][i].Multiply();
                Vector4 pix = ReadPixel(wPixCoord + o);
                writePix[0] = Math::FMA(writePix[0], w, pix[0]);
                writePix[1] = Math::FMA(writePix[1], w, pix[1]);
                writePix[2] = Math::FMA(writePix[2], w, pix[2]);
                writePix[3] = Math::FMA(writePix[3], w, pix[3]);
            }
            writePix[0] /= Float(Kernel::Dim.Multiply());
            writePix[1] /= Float(Kernel::Dim.Multiply());
            writePix[2] /= Float(Kernel::Dim.Multiply());
            writePix[3] /= Float(Kernel::Dim.Multiply());
            // Finally write the pixel
            TracerSurfView sOut = surfaceOut;
            GenericWrite(sOut, writePix, wPixCoordInt);
        }
    }
}

// ========================================= //
//       Stochastic Mipmap Generation        //
// ========================================= //
//
// Fully dynamic stochastic mipmap generation,
// is not used anymore. Same as the code above, but
// for multiple textures.
//
// This technique is not used because it is not good.
// However for lower mip levels it should be quite good
// since we accumulate quite a bit of samples for these pixels.
//
// No phasing issue but it technically have Monte Carlo noise.
template<uint32_t TPB, class Filter>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCGenerateMipmaps(// I-O
                       MRAY_GRID_CONSTANT const Span<MipArray<TracerSurfView>> dSurfaces,
                       // Inputs
                       MRAY_GRID_CONSTANT const Span<const MipGenParams> dMipGenParamsList,
                       // Constants
                       MRAY_GRID_CONSTANT const uint32_t currentMipLevel,
                       MRAY_GRID_CONSTANT const Vector2ui spp,
                       MRAY_GRID_CONSTANT const uint32_t blockPerTexture,
                       MRAY_GRID_CONSTANT const FilterMode filterMode,
                       MRAY_GRID_CONSTANT const Filter FilterFunc)
{
    static constexpr Vector2ui TILE_SIZE = KC_MIPMAP_GEN_TILE_SIZE;
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
        TracerSurfView writeSurf = dSurfaces[tI][currentMipLevel];
        const TracerSurfView readSurf = dSurfaces[tI][currentMipLevel - 1];
        //
        Vector2ui mipRes = Graphics::TextureMipSize(curParams.mipZeroRes,
                                                    currentMipLevel);
        Vector2ui parentRes = Graphics::TextureMipSize(curParams.mipZeroRes,
                                                       currentMipLevel - 1);
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

            #ifdef MRAY_GPU_BACKEND_CPU
                // The code above for CPU emulation will destroy
                // cache locality so we do all the work of a tile
                // on a single thread.
                if(kp.threadId == 0)
                for(uint32_t i = 0; i < TILE_SIZE.Multiply(); i++)
            #else
                // For GPU this is cooperative so no need
                uint32_t i = kp.threadId;
            #endif
            {
                Vector2ui localPI = Vector2ui(i % TILE_SIZE[0],
                                              i / TILE_SIZE[0]);
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
                    Vector2ui rPixCoordInt = Vector2ui(Math::RoundInt(rPixCoord));
                    return GenericRead(rPixCoordInt, readSurf);
                });
                // Finally write the pixel
                GenericWrite(writeSurf, writePix, wPixCoordInt);
            }
        }
    }
}

// ========================================= //
//        Stochastic Image Clamping          //
// ========================================= //
//
// Exactly the same code of stochastic version
// but kernel evaluations are in runtime.
template<uint32_t TPB, class Filter>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(TPB)
void KCClampImage(// Output
                  MRAY_GRID_CONSTANT const TracerSurfView surfaceOut,
                  // Inputs
                  MRAY_GRID_CONSTANT const Span<const Byte> dBufferImage,
                  // Constants
                  MRAY_GRID_CONSTANT const Vector2ui surfaceImageRes,
                  MRAY_GRID_CONSTANT const Vector2ui bufferImageRes,
                  MRAY_GRID_CONSTANT const Vector2ui spp,
                  MRAY_GRID_CONSTANT const FilterMode filterMode,
                  MRAY_GRID_CONSTANT const Filter FilterFunc)
{
    static constexpr Vector2ui TILE_SIZE = KC_CLAMP_IMAGE_TILE_SIZE;
    static_assert(TILE_SIZE.Multiply() == TPB);

    KernelCallParams kp;
    // Loop over the tiles for this tex, each block is dedicated to a tile
    Vector2ui totalTiles = Math::DivideUp(surfaceImageRes, TILE_SIZE);
    for(uint32_t tileI = kp.blockId; tileI < totalTiles.Multiply();
        tileI += kp.gridSize)
    {
        #ifdef MRAY_GPU_BACKEND_CPU
            // This code is slow lets try this
            if(kp.threadId == 0)
            for(uint32_t pI = 0; pI < TILE_SIZE.Multiply(); pI++)
        #else
            // Load cooperatively
            uint32_t pI = kp.threadId;
        #endif
        {
            Vector2ui localPI = Vector2ui(pI % TILE_SIZE[0],
                                          pI / TILE_SIZE[0]);
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
                Vector2ui rPixCoordInt = Vector2ui(Math::Round(rPixCoord));
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
            TracerSurfView sOut = surfaceOut;
            GenericWrite(sOut, writePix, wPixCoordInt);

            //TracerSurfView sOut = surfaceOut;
            //GenericWrite(sOut, Vector4::Zero(), wPixCoordInt);
        }
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
                             [[maybe_unused]]
                             MRAY_GRID_CONSTANT const uint32_t maxPixelPerSample,
                             MRAY_GRID_CONSTANT const Vector2i extent)
{
    int32_t filterWH = FilterRadiusToPixelWH(filterRadius);
    Vector2i range = FilterRadiusPixelRange(filterWH);
    // Don't use 1.0f exactly here
    // pixel is [0,1)
    //Float pixelWidth = Math::PrevFloat<Float>(1);
    Float radiusSqr = filterRadius * filterRadius;

    KernelCallParams kp;
    uint32_t sampleCount = static_cast<uint32_t>(dImgCoords.size());
    for(uint32_t sampleIndex = kp.GlobalId(); sampleIndex < sampleCount;
        sampleIndex += kp.TotalSize())
    {
        Vector2 imgCoords = dImgCoords[sampleIndex].GetPixelIndex();
        imgCoords += Vector2(0.5);

        auto [iCoordX, fX] = Math::ModF(imgCoords[0]);
        auto [iCoordY, fY] = Math::ModF(imgCoords[1]);
        Vector2 relImgCoords = Vector2(iCoordX, iCoordY);
        Vector2 fractions = Vector2(fX, fY);

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
        uint32_t stride = 0;
        for(int32_t y = localRangeX[0]; y < localRangeX[1]; y++)
        for(int32_t x = localRangeX[0]; x < localRangeX[1]; x++)
        {
            Vector2 pixCoord = relImgCoords + Vector2(x, y);
            Vector2 pixCenter = pixCoord + Float(0.5);
            // TODO: Should we use pixCoord or center coord
            // which one is better?
            Float lengthSqr = Math::LengthSqr(imgCoords - pixCenter);

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
            // This is why we do not terminate this double loop, we either write INT_MAX
            // or actually write the result.
            uint32_t writeIndex = stride * sampleCount + sampleIndex;
            dPixelIds[writeIndex] = pixelLinearId;
            dIndices[writeIndex] = index;
            stride++;
        }
        // All Done!
    }
}

#if (defined MRAY_GPU_BACKEND_CUDA) || (defined MRAY_GPU_BACKEND_HIP)

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
    #ifdef MRAY_GPU_BACKEND_CUDA
        namespace cub_or_hip = cub;
    #else
        namespace cub_or_hip = hipcub;
    #endif

    KernelCallParams kp;
    assert(dStartOffsets.size() == (dPixelIds.size() + 1));
    static_assert(TPB % LOGICAL_WARP_SIZE == 0);

    // Some constants
    static constexpr uint32_t WARP_PER_BLOCK = TPB / LOGICAL_WARP_SIZE;
    const uint32_t totalWarpCount = WARP_PER_BLOCK * kp.gridSize;
    const uint32_t globalWarpId = kp.GlobalId() / LOGICAL_WARP_SIZE;
    const uint32_t localWarpId = kp.threadId / LOGICAL_WARP_SIZE;
    const uint32_t laneId = kp.GlobalId() % LOGICAL_WARP_SIZE;

    using WarpReduceVec4 = cub_or_hip::WarpReduce<Vector4, LOGICAL_WARP_SIZE>;
    using ReduceShMem = typename WarpReduceVec4::TempStorage;
    // Per-Warp Shared Memory
    MRAY_SHARED_MEMORY Vector2ui    sSegmentRange[WARP_PER_BLOCK];
    MRAY_SHARED_MEMORY CommonKey    sResponsiblePixel[WARP_PER_BLOCK];
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
        if(laneId == LOAD_2) sResponsiblePixel[localWarpId] = dPixelIds[segmentIndex];
        // Wait for these writes to be visible across warp
        WarpSynchronize<LOGICAL_WARP_SIZE>();

        // This partition is residuals, we conservatively allocated but not every
        // potential filter slot is not filled. Skip this partition
        if(sResponsiblePixel[localWarpId] == INVALID_MORTON)
            continue;

        // Locally compute the coordinates
        namespace Morton = Graphics::MortonCode;
        Vector2i pixCoordsInt = Vector2i(Morton::Decompose2D(sResponsiblePixel[localWarpId]));
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
        // pixel. However; other writes would've been occurred on
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

#elif defined MRAY_GPU_BACKEND_CPU

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
    // Each thread is responsible for a segment
    for(uint32_t pixId = kp.GlobalId(); pixId < hPartitionCount[0]; pixId += kp.TotalSize())
    {
        //
        Vector2ui segmentRange = Vector2ui(dStartOffsets[pixId + 0], dStartOffsets[pixId + 1]);
        CommonKey responsiblePixel = dPixelIds[pixId];

        // This partition is for residuals, we conservatively allocated but not every
        // potential filter slot is not filled. Skip this partition
        if(responsiblePixel == INVALID_MORTON) continue;

        //
        namespace Morton = Graphics::MortonCode;
        Vector2i pixCoordsInt = Vector2i(Morton::Decompose2D(responsiblePixel));
        Vector2 pixCoords = Vector2(pixCoordsInt) + Float(0.5);
        Vector4 value = Vector4::Zero();
        for(uint32_t i = segmentRange[0]; i < segmentRange[1]; i++)
        {
            uint32_t readIndex = dIndices[i];
            Vector3 sampleVal = Vector3(dValues[readIndex]);
            Vector2 sampleCoord = dImgCoords[readIndex].GetPixelIndex();
            Float weight = filter.Evaluate(pixCoords - sampleCoord) * scalarWeightMultiplier;
            sampleVal *= weight;
            value += Vector4(sampleVal, weight);
        }

        Float weight = img.FetchWeight(pixCoordsInt);
        img.StoreWeight(weight + value[3], pixCoordsInt);

        Vector3 pixValue = img.FetchPixel(pixCoordsInt);
        img.StorePixel(pixValue + Vector3(value), pixCoordsInt);
    }
}

#else

#error "Generic version of reconstruction filter is not implemented!"

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
{}

#endif

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
    //Float pixelWidth = Math::PrevFloat<Float>(1);
    Float radiusSqr = filterRadius * filterRadius;

    assert(dValues.size() == dImgCoords.size());
    uint32_t sampleCount = static_cast<uint32_t>(dImgCoords.size());

    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < sampleCount; i += kp.TotalSize())
    {
        Vector2 imgCoords = dImgCoords[i].GetPixelIndex();
        imgCoords += Vector2(0.5);

        auto [iCoordX, fX] = Math::ModF(imgCoords[0]);
        auto [iCoordY, fY] = Math::ModF(imgCoords[1]);
        Vector2 relImgCoords = Vector2(iCoordX, iCoordY);
        Vector2 fractions = Vector2(fX, fY);

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
            Float lengthSqr = Math::LengthSqr(imgCoords - pixCenter);

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
            if(Math::Abs(weight) < MathConstants::Epsilon<Float>()) continue;

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
                      MRAY_GRID_CONSTANT const Float)
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
    queue.IssueWorkKernel<KCSetImagePixelsIndirect>
    (
        "KCSetImagePixelsIndirect",
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
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
    assert(dFilterWeights.size() == dImgCoords.size());;
    queue.IssueWorkKernel<KCSetImagePixelsIndirectAtomic>
    (
        "KCSetImagePixelsIndirectAtomic",
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dIndices.size())},
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
    queue.IssueWorkKernel<KCSetImagePixels>
    (
        "KCSetImagePixels",
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dValues.size())},
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

    uint32_t wh = uint32_t(FilterRadiusToPixelWH(filterRadius));
    uint32_t maxPixelPerSample = wh * wh;
    uint32_t totalPPS = elementCount * maxPixelPerSample;

    // Maximum partition count is relative tot he image resolution
    Vector2i maxPixels = img.Extent() + Vector2i(wh);
    uint32_t maxPartitionCount = static_cast<uint32_t>(maxPixels.Multiply());

    auto [dIndices, dKeys] = partitioner.Start(totalPPS, maxPartitionCount, queue, false);

    queue.IssueWorkKernel<KCExpandSamplesToPixels>
    (
        "KCExpandSamplesToPixels",
        DeviceWorkIssueParams{.workCount = elementCount},
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
    logicalWarpSize = Math::Min(logicalWarpSize, WarpSize());

    // Some boilerplate to make the code more readable
    auto KernelCall = [&]<auto Kernel>(std::string_view Name)
    {
        // Forgot why this is here? But probably to capture by copy?
        // TODO: Investigate
        Span<const Spectrum> dValuesIn = dValues;
        Span<const ImageCoordinate> dImgCoordsIn = dImgCoords;
        uint32_t blockCount = Math::DivideUp(hPartitionCount[0], logicalWarpSize);
        queue.IssueBlockKernel<Kernel>
        (
            Name,
            DeviceBlockIssueParams
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

    if constexpr(MRAY_GPU_BACKEND_IS_CPU)
    {
        KernelCall.template operator()<WK[0]>("KCFilterToImgWarpRGB<1>");
    }
    else switch(logicalWarpSize - 1)
    {
        case 0: KernelCall.template operator()<WK[0]>("KCFilterToImgWarpRGB<1>"); break;
        case 1: KernelCall.template operator()<WK[1]>("KCFilterToImgWarpRGB<2>"); break;
        case 2: KernelCall.template operator()<WK[2]>("KCFilterToImgWarpRGB<4>"); break;
        case 3: KernelCall.template operator()<WK[3]>("KCFilterToImgWarpRGB<8>"); break;
        case 4: KernelCall.template operator()<WK[4]>("KCFilterToImgWarpRGB<16>"); break;
        case 5: KernelCall.template operator()<WK[5]>("KCFilterToImgWarpRGB<32>"); break;
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
    queue.IssueWorkKernel<KCFilterToImgAtomicRGB<Filter>>
    (
        "KCFilterToImgAtomicRGB",
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(dValues.size())},
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
    uint32_t wh = uint32_t(FilterRadiusToPixelWH(filterRadius));
    uint32_t maxPixelPerSample = wh * wh;
    uint32_t totalPPS = totalWork * maxPixelPerSample;

    // Try to comply the parallelization hint
    // Divide the work equally
    uint32_t iterations = Math::DivideUp(totalPPS, parallelHint);
    uint32_t workPerIter = Math::DivideUp(totalWork, iterations);
    for(uint32_t i = 0; i < iterations; i++)
    {
        uint32_t start = workPerIter * i;
        uint32_t end = Math::Min(workPerIter * (i + 1), totalWork);
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
void GenerateMipsGeneric(const std::vector<MipArray<TracerSurfRef>>& textures,
                         const std::vector<MipGenParams>& mipGenParams,
                         const GPUSystem& gpuSystem, Filter)
{
    assert(textures.size() == mipGenParams.size());
    // TODO: Textures should be partitioned with respect to
    // devices, so that we can launch kernel from those devices
    const GPUDevice& bestDevice = gpuSystem.BestDevice();
    const GPUQueue& queue = bestDevice.GetComputeQueue(0);

    // We can temporarily allocate here. This will be done at
    // initialization time.
    DeviceLocalMemory mem(gpuSystem.BestDevice());
    Span<MipArray<TracerSurfView>> dSufViews;
    Span<MipGenParams> dMipGenParams;
    MemAlloc::AllocateMultiData(Tie(dSufViews, dMipGenParams),
                                mem, {textures.size(), textures.size()});

    // Copy references
    std::vector<MipArray<TracerSurfView>> hSurfViews;
    hSurfViews.reserve(textures.size());
    for(const MipArray<TracerSurfRef>& surfRefs : textures)
    {
        MipArray<TracerSurfView> mipViews;
        for(uint32_t i = 0; i < TracerConstants::MaxTextureMipCount; i++)
        {
            const TracerSurfRef& surf = surfRefs[i];
            mipViews[i] = std::visit([](auto&& v) -> TracerSurfView
            {
                using T = std::remove_cvref_t<decltype(v)>;
                if constexpr(std::is_same_v<T, std::monostate>)
                    return std::monostate{};
                else return v.View();
            }, surf);
        }
        hSurfViews.push_back(mipViews);
    }

    auto hSurfViewSpan = Span<MipArray<TracerSurfView>>(hSurfViews);
    auto hMipGenParams = Span<const MipGenParams>(mipGenParams);
    queue.MemcpyAsync(dSufViews, ToConstSpan(hSurfViewSpan));
    queue.MemcpyAsync(dMipGenParams, hMipGenParams);

    // Since texture writes are not coherent,
    // we need to call one kernel for each level of mips
    uint16_t maxMipCount = std::transform_reduce(mipGenParams.cbegin(),
                                                 mipGenParams.cend(),
                                                 std::numeric_limits<uint16_t>::min(),
    [](uint16_t l, uint16_t r)
    {
        return Math::Max(l, r);
    },
    [](const MipGenParams& p) -> uint16_t
    {
        return p.mipCount;
    });

    // Start from 1, we assume miplevel zero is already available
    for(uint16_t i = 1; i < maxMipCount; i++)
    {
        // Find maximum block count for state allocation
        // TODO: Change this so that it is relative to the
        // filter radius.
        // We will dedicate N blocks for each texture.
        static constexpr uint32_t THREAD_PER_BLOCK = 512;
        static constexpr uint32_t BLOCK_PER_TEXTURE = 256;
        static constexpr Vector2ui SPP = (MRAY_IS_DEBUG) ? Vector2ui(2, 2) : Vector2ui(8, 8);
        static constexpr uint32_t BlockPerTexture = Math::Max(1u, BLOCK_PER_TEXTURE >> 1);
        uint32_t textureCount = static_cast<uint32_t>(dSufViews.size());
        uint32_t blockCount = BlockPerTexture * textureCount;
        //
        #ifdef MRAY_GPU_BACKEND_CPU
            uint32_t blockSize = 1;
        #else
            uint32_t blockSize = THREAD_PER_BLOCK;
        #endif
        using Kernel = KernelWeightsStatic1D<Filter>;
        queue.IssueBlockKernel<KCGenerateMipmapsStatic<THREAD_PER_BLOCK, Kernel>>
        (
            "KCGenerateMipmaps",
            DeviceBlockIssueParams
            {
                .gridSize = blockCount,
                .blockSize = blockSize
            },
            // I-O
            dSufViews,
            // Inputs
            dMipGenParams,
            // Constants
            i,
            BlockPerTexture
        );
    }
    queue.Barrier().Wait();
}

template<class Filter>
void ClampImageFromBufferGeneric(// Output
                                 const TracerSurfRef& surf,
                                 // Input
                                 const Span<const Byte>& dDataBuffer,
                                 // Constants
                                 const Vector2ui& surfImageDims,
                                 const Vector2ui& bufferImageDims,
                                 Filter filter,
                                 const GPUQueue& queue)
{
    using Math::DivideUp;
    // Find maximum block count for state allocation
    // TODO: Change this so that it is relative to the
    // filter radius.
    //static constexpr Vector2ui SPP = Vector2ui(4, 4);
    static constexpr Vector2ui SPP = Vector2ui(8, 8);
    //static constexpr Vector2ui SPP = Vector2ui(2, 2);
    static constexpr Vector2ui TILE_SIZE = KC_CLAMP_IMAGE_TILE_SIZE;
    static constexpr uint32_t THREAD_PER_BLOCK = TILE_SIZE.Multiply();
    uint32_t blockCount = DivideUp(surfImageDims, TILE_SIZE).Multiply();

    TracerSurfView surfRef = Visit(surf, [](auto&& v) -> TracerSurfView
    {
        using T = std::remove_cvref_t<decltype(v)>;
        if constexpr(std::is_same_v<T, std::monostate>)
            return std::monostate{};
        else return v.View();
    });

    #ifdef MRAY_GPU_BACKEND_CPU
        uint32_t blockSize = 1;
    #else
        uint32_t blockSize = THREAD_PER_BLOCK;
    #endif

    queue.IssueBlockKernel<KCClampImage<THREAD_PER_BLOCK, Filter>>
    (
        "KCClampImage",
        DeviceBlockIssueParams
        {
            .gridSize = blockCount,
            .blockSize = blockSize
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
        FilterMode::SAMPLING,
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
void TextureFilterT<E, FF>::GenerateMips(const std::vector<MipArray<TracerSurfRef>>& textures,
                                         const std::vector<MipGenParams>& params) const
{
    using namespace std::string_literals;
    static const std::string Name = ("GenerateMips"s + std::string(FilterType::ToString(E)));
    static const auto annotation = gpuSystem.CreateAnnotation(Name);
    const auto _ = annotation.AnnotateScope();

    if(Math::RoundInt(filterRadius) != 2)
        MRAY_WARNING_LOG("For mipmap generation, only filter radius of \"2\" is supported");

    GenerateMipsGeneric(textures, params, gpuSystem, FF(filterRadius));
}

template<FilterType::E E, class FF>
void TextureFilterT<E, FF>::ClampImageFromBuffer(// Output
                                                 const TracerSurfRef& surf,
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

template class TextureFilterT<FilterType::BOX, BoxFilter>;
template class TextureFilterT<FilterType::TENT, TentFilter>;
template class TextureFilterT<FilterType::GAUSSIAN, GaussianFilter>;
template class TextureFilterT<FilterType::MITCHELL_NETRAVALI, MitchellNetravaliFilter>;