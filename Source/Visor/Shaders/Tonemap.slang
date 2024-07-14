module Tonemap;

__include "TonemapFunctions";

static const uint32_t STATIC_THREAD_PER_BLOCK_1D = 256;
static const uint32_t STATIC_THREAD_PER_BLOCK_2D_X = 16;
static const uint32_t STATIC_THREAD_PER_BLOCK_2D_Y = 16;

extern struct Tonemapper : ITonemapper;

//======================================//
//              FUNCTORS                //
//======================================//

interface IBinaryOp { static float op(in float v0, in float v1); }

struct AddOp : IBinaryOp
{
    static float op(in float v0, in float v1)
    {
        return v0 + v1;
    }
}

struct MaxOp : IBinaryOp
{
    static float op(in float v0, in float v1)
    {
        return max(v0, v1);
    }
}

struct MinOp : IBinaryOp
{
    static float op(in float v0, in float v1)
    {
        return min(v0, v1);
    }
}

//======================================//
//      SHADER STORAGE BUFFERS          //
//======================================//

struct MinMaxAndAvg
{
    float minLum;
    float maxLum;
    float avgLum;
    float logAvgLum;

    float ReadMinLum() { return minLum;}
    float ReadMaxLum() { return maxLum;}
    float ReadAvgLum(uint size) { return exp(avgLum / size);}
    float ReadLogAvgLum(uint size) { return exp(logAvgLum / size);}
}

struct TonemapParameters
{
    typedef Tonemapper.TMParams TMParams;
    typedef ColorConverterOut.GammaParams GammaParams;

    // We could not utilize multi-binding here
    // I think it is because you cannot have
    // generic (templated) UBO's?
    StructuredBuffer<TMParams>          tonemapParams;
    StructuredBuffer<GammaParams>       outGammaParams;
    Texture2D<float4>                   inputTex;
    RWTexture2D<float4>                 outputTex;
    RWStructuredBuffer<MinMaxAndAvg>    maxAndAvg;
}

ParameterBlock<TonemapParameters> params;

[[vk::push_constant]]
uint reduceCount;

groupshared float sReduceBuffer[STATIC_THREAD_PER_BLOCK_1D];

float BlockLocalReduction<T : IBinaryOp>(in float value, in uint localId, in T op)
{
    // Unrolled parallel reduction using shared mem
    sReduceBuffer[localId] = value;
    AllMemoryBarrierWithGroupSync();

    // Hierarhical decay of block local reduction using shared memory
    // This is a more generic approach then warp local intrinsics.
    static const uint START = (STATIC_THREAD_PER_BLOCK_1D >> 1);
    [ForceUnroll]
    for(uint i = START; i != 0; i >>= 1)
    {
        if (localId < i) sReduceBuffer[localId] = T::op(sReduceBuffer[localId], sReduceBuffer[localId + i]);
        AllMemoryBarrierWithGroupSync();
    }
    return sReduceBuffer[localId];
}

// We dropped using this, this was simpler but slower
// (both on Intel and NVIDIA). Commented this for future use.
//void AtomicFloatOp<T : IBinaryOp>(in uint outLoc, float val, in T)
//{
//    // CAS Atomic operation for Atomic Max for floats
//    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
//    uint assumed;
//    uint old = outLoc;

//    [allow_uav_condition]
//    do
//    {
//        assumed = old;
//        float assumedF = asfloat(assumed);
//        float newValF = T::op(assumedF, val);
//        uint newVal = asuint(newValF);
//        // TODO: outLoc was directly refers to the params.maxAndAvg[0]...
//        // but slang couldn't (or I did something wrong) compile it
//        // in a way that atomic compare and exachange actions on the
//        // "outLoc" local variable (at least Vulkan validatior said so)
//        // So we directly refer the SSBO variables here via a literal
//        // Old version!
//        // InterlockedCompareExchange(outLoc, assumed, newVal, old);
//        //
//        if(outLoc == 0) InterlockedCompareExchange(params.maxAndAvg[0].minLum,
//                                                   assumed, newVal, old);
//        if(outLoc == 1) InterlockedCompareExchange(params.maxAndAvg[0].maxLum,
//                                                   assumed, newVal, old);
//        if(outLoc == 2) InterlockedCompareExchange(params.maxAndAvg[0].logAvgLum,
//                                                   assumed, newVal, old);
//        if(outLoc == 3) InterlockedCompareExchange(params.maxAndAvg[0].avgLum,
//                                                   assumed, newVal, old);
//    } while(assumed != old);
//}

//======================================//
//     LUMINANCE REDUCTION SHADER       //
//======================================//
[shader("compute")]
[numthreads(STATIC_THREAD_PER_BLOCK_1D, 1, 1)]
void KCFindAvgMaxLumTex(uint3 globalId: SV_DispatchThreadID,
                        uint3 blockId: SV_GroupID,
                        uint3 localId: SV_GroupThreadID)
{
    uint2 dim;
    params.inputTex.GetDimensions(dim[0], dim[1]);
    uint2 pixId = uint2(globalId.x % dim.x,
                        globalId.x / dim.x);
    float readMin, readMax, readAvg, readLogAvg;
    if(globalId.x < (dim[0] * dim[1]))
    {
        float3 inColor = params.inputTex.Load(uint3(pixId, 0)).xyz;
        float luminance = ColorConverterIn::ToYxy(inColor)[0];
        readMin = luminance;
        readMax = luminance;
        readAvg = luminance;
        readLogAvg = log(luminance + 1e-3f);
    }
    else
    {
        // Put operation invariants here
        // we can't cull threads since we will use block synchronization
        readMin = 1e20f;
        readMax = -1e20f;
        readAvg = 0.0f;
        readLogAvg = 0.0f;
    }
    // The reduction
    float localMin = BlockLocalReduction(readMin, localId.x, MinOp());
    float localMax = BlockLocalReduction(readMax, localId.x, MaxOp());
    float localAvg = BlockLocalReduction(readAvg, localId.x, AddOp());
    float localLogAvg = BlockLocalReduction(readLogAvg, localId.x, AddOp());

    if(localId.x == 0)
    {
        params.maxAndAvg[blockId.x].minLum = localMin;
        params.maxAndAvg[blockId.x].maxLum = localMax;
        params.maxAndAvg[blockId.x].avgLum = localAvg;
        params.maxAndAvg[blockId.x].logAvgLum = localLogAvg;
    }
}

[shader("compute")]
[numthreads(STATIC_THREAD_PER_BLOCK_1D, 1, 1)]
void KCFindAvgMaxLumBuff(uint3 globalId: SV_DispatchThreadID,
                         uint3 blockId: SV_GroupID,
                         uint3 localId: SV_GroupThreadID)
{
    float readMin, readMax, readAvg, readLogAvg;
    if(globalId.x >= reduceCount)
    {
        // Put operation invariants here
        // we can't cull threads since we will use block synchronization
        readMin = 1e20f;
        readMax = -1e20f;
        readAvg = 0.0f;
        readLogAvg = 0.0f;
    }
    else
    {
        readMin = params.maxAndAvg[globalId.x].minLum;
        readMax = params.maxAndAvg[globalId.x].maxLum;
        readAvg = params.maxAndAvg[globalId.x].avgLum;
        readLogAvg = params.maxAndAvg[globalId.x].logAvgLum;
    }

    // The reduction
    float localMin = BlockLocalReduction(readMin, localId.x, MinOp());
    float localMax = BlockLocalReduction(readMax, localId.x, MaxOp());
    float localAvg = BlockLocalReduction(readAvg, localId.x, AddOp());
    float localLogAvg = BlockLocalReduction(readLogAvg, localId.x, AddOp());

    if(localId.x == 0)
    {
        params.maxAndAvg[blockId.x].minLum = localMin;
        params.maxAndAvg[blockId.x].maxLum = localMax;
        params.maxAndAvg[blockId.x].avgLum = localAvg;
        params.maxAndAvg[blockId.x].logAvgLum = localLogAvg;
    }
}

//======================================//
//           TONEMAP SHADER             //
//======================================//
[shader("compute")]
[numthreads(STATIC_THREAD_PER_BLOCK_2D_X,
            STATIC_THREAD_PER_BLOCK_2D_Y, 1)]
void KCTonemapImage(uint3 globalId: SV_DispatchThreadID)
{
    // Get dimensions from the texture
    // TODO: Is this faster or slower compared to
    // uniform buffer fetch
    uint2 dim = uint2(0);
    params.inputTex.GetDimensions(dim[0], dim[1]);
    uint pixCount = dim[0] * dim[1];

    if(any(globalId.xy >= dim)) return;
    uint2 pixId = globalId.xy;

    float2 minmaxLuminance = float2(params.maxAndAvg[0].ReadMinLum(),
                                    params.maxAndAvg[0].ReadMaxLum());
    float2 avgLuminance = float2(exp(params.maxAndAvg[0].ReadLogAvgLum(pixCount)),
                                 params.maxAndAvg[0].ReadAvgLum(pixCount));

    float3 inColorHDR = params.inputTex.Load(uint3(pixId, 0)).xyz;
    ColorConverterIn  inColorSpace;
    ColorConverterOut outColorSpace;
    float3 outColorSDR = Tonemapper.ToneMap(inColorHDR,
                                            inColorSpace,
                                            outColorSpace,
                                            params.outGammaParams[0],
                                            params.tonemapParams[0],
                                            minmaxLuminance,
                                            avgLuminance);
    params.outputTex[pixId] = float4(outColorSDR, 1.0f);
}