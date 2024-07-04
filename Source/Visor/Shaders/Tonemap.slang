module Tonemap;

__include "TonemapFunctions";

static const uint32_t STATIC_THREAD_PER_BLOCK_1D = 256;

extern struct Tonemapper : ITonemapper;

//======================================//
//              FUNCTORS                //
//======================================//

interface IBinaryOp { static float op(float v0, float v1); }

struct AddOp : IBinaryOp
{
    static float op(float v0, float v1)
    {
        return v0 + v1;
    }
}

struct MaxOp : IBinaryOp
{
    static float op(float v0, float v1)
    {
        return max(v0, v1);
    }
}

//======================================//
//      SHADER STORAGE BUFFERS          //
//======================================//

struct MaxAndAvg
{
    uint maxLum;
    uint avgLum;

    float ReadMaxLum() { return asfloat(maxLum);}
    float ReadAvgLum() { return asfloat(avgLum);}
}

struct TonemapParameters
{
    typedef Tonemapper.TMParams TMParams;
    typedef ColorConverterOut.GammaParams GammaParams;

    // We could not utilize multi-binding here
    // I think it is because you cannot have
    // generic (templated) UBO's?
    StructuredBuffer<TMParams>      tonemapParams;
    StructuredBuffer<GammaParams>   outGammaParams;
    Texture2D<float4>               inputTex;
    RWTexture2D<float4>             outputTex;
    RWStructuredBuffer<MaxAndAvg>   maxAndAvg;
}

ParameterBlock<TonemapParameters> params;

groupshared float sReduceBuffer[STATIC_THREAD_PER_BLOCK_1D];

float BlockLocalReduction<T : IBinaryOp>(in float value, in uint localId, in T op)
{
    // Unrolled parallel reduction using shared mem
    sReduceBuffer[localId] = value;
    AllMemoryBarrierWithGroupSync();

    // This failed due to exceeding a threshold
    // so hand unrolling
    //static const uint START = (STATIC_THREAD_PER_BLOCK_1D >> 1);
    //[ForceUnroll]
    // for (uint i = START; i != 0; i >> 1)
    //{
    //    if (localId < i) sReduceBuffer[localId] = T::op(sReduceBuffer[localId], sReduceBuffer[localId + i]);
    //    AllMemoryBarrierWithGroupSync();
    //}
    if (localId < 128) sReduceBuffer[localId] = T::op(sReduceBuffer[localId], sReduceBuffer[localId + 128]);
    AllMemoryBarrierWithGroupSync();
    if (localId < 64) sReduceBuffer[localId] = T::op(sReduceBuffer[localId], sReduceBuffer[localId + 64]);
    AllMemoryBarrierWithGroupSync();
    if (localId < 32) sReduceBuffer[localId] = T::op(sReduceBuffer[localId], sReduceBuffer[localId + 32]);
    AllMemoryBarrierWithGroupSync();
    if (localId < 16) sReduceBuffer[localId] = T::op(sReduceBuffer[localId], sReduceBuffer[localId + 16]);
    AllMemoryBarrierWithGroupSync();
    if (localId <  8) sReduceBuffer[localId] = T::op(sReduceBuffer[localId], sReduceBuffer[localId + 8]);
    AllMemoryBarrierWithGroupSync();
    if (localId <  4) sReduceBuffer[localId] = T::op(sReduceBuffer[localId], sReduceBuffer[localId + 4]);
    AllMemoryBarrierWithGroupSync();
    if (localId <  2) sReduceBuffer[localId] = T::op(sReduceBuffer[localId], sReduceBuffer[localId + 2]);
    AllMemoryBarrierWithGroupSync();
    if (localId <  1) sReduceBuffer[localId] = T::op(sReduceBuffer[localId], sReduceBuffer[localId + 1]);
    AllMemoryBarrierWithGroupSync();

    return sReduceBuffer[localId];
}

void AtomicFloatOp<T : IBinaryOp>(in uint outLoc, float val, in T)
{
    // CAS Atomic operation for Atomic Max for floats
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
    uint assumed;
    uint old = outLoc;

    [allow_uav_condition]
    do
    {
        assumed = old;
        float assumedF = asfloat(assumed);
        float newValF = T::op(assumedF, val);
        uint newVal = asuint(newValF);
        // TODO: outLoc was directly refers to the params.maxAndAvg[0]...
        // but slang couldn't (o I did something wrong) compile it
        // in a way that atomic compare and exachange actions on the
        // "outLoc" local variable (at least Vulkan validatior said so)
        // So we directly refer the SSBO variables here via a literal
        // Old version!
        // InterlockedCompareExchange(outLoc, assumed, newVal, old);
        //
        if(outLoc == 0) InterlockedCompareExchange(params.maxAndAvg[0].maxLum,
                                                   assumed, newVal, old);
        if(outLoc == 1) InterlockedCompareExchange(params.maxAndAvg[0].avgLum,
                                                   assumed, newVal, old);
    } while(assumed != old);
}

//======================================//
//     LUMINANCE REDUCTION SHADER       //
//======================================//
[shader("compute")]
[numthreads(STATIC_THREAD_PER_BLOCK_1D, 1, 1)]
void KCFindAvgMaxLum(uint3 globalId: SV_DispatchThreadID,
                     uint3 localId: SV_GroupThreadID)
{
    uint2 dim;
    params.inputTex.GetDimensions(dim[0], dim[1]);
    uint2 pixId = uint2(globalId.x % dim.x,
                        globalId.x / dim.x);

    float3 inColor = params.inputTex.Load(uint3(pixId, 0), int2(0, 0)).xyz;
    float luminance = ColorConverterIn::ToYxy(inColor)[0];

    // The reduction
    float localAvg = BlockLocalReduction(luminance, localId.x, AddOp());
    float localMax = BlockLocalReduction(luminance, localId.x, MaxOp());

    // Now atomics, (still we need to do comp-sawp idiom for floats...)
    // TODO: If have free time, check the full down reduction
    // (one extra call for this compute shader on an intermediate buffer, not the texture)
    // is faster or slower (probably faster but setting that up is a hassle)
    if(localId.x == 0)
    {
        float old;
        float totalRecip = 1.0f / float(dim.x + dim.y);
        localAvg *= totalRecip;

        AtomicFloatOp(0, localMax, MaxOp());
        AtomicFloatOp(1, localAvg, AddOp());
    }
}

//======================================//
//           TONEMAP SHADER             //
//======================================//
[shader("compute")]
[numthreads(STATIC_THREAD_PER_BLOCK_1D, 1, 1)]
void KCTonemapImage(uint3 globalId : SV_DispatchThreadID)
{
    uint2 dim;
    params.inputTex.GetDimensions(dim[0], dim[1]);
    uint2 pixId = uint2(globalId.x % dim.x,
                        globalId.x / dim.x);
    //
    float3 inColorHDR = params.inputTex.Load(uint3(pixId, 0), int2(0, 0)).xyz;

    ColorConverterIn  inColorSpace;
    ColorConverterOut outColorSpace;
    float3 outColorSDR = Tonemapper.ToneMap(inColorHDR,
                                            inColorSpace, outColorSpace,
                                            params.outGammaParams[0],
                                            params.tonemapParams[0],
                                            params.maxAndAvg[0].ReadAvgLum(),
                                            params.maxAndAvg[0].ReadMaxLum());
    params.outputTex[pixId] = float4(outColorSDR, 0.0f);
}