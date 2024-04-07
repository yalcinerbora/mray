module Tonemap;
import ColorFunctions;
import TonemapFunctions;

static const uint32_t STATIC_THREAD_PER_BLOCK_1D = 256;

extern struct Tonemapper : ITonemapper;

//======================================//
//              FUNCTORS                //
//======================================//

interface IBinaryOp { float op(in float v0, float v1); }

struct AddOp : IBinaryOp
{
    float op(in float v0, float v1)
    {
        return v0 + v1;
    }
}

struct MaxOp : IBinaryOp
{
    float op(in float v0, float v1)
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

    Texture2D<float3>               inputTex;
    RWTexture2D<float3>             outputTex;
    StructuredBuffer<TMParams>      tonemapParams;
    RWStructuredBuffer<MaxAndAvg>   maxAndAvg;
}

ParameterBlock<TonemapParameters> tonemapParameters;
ParameterBlock<ColorConverterOut::GammaParams> gammaParamsOut;

groupshared float sReduceBuffer[STATIC_THREAD_PER_BLOCK_1D];
float BlockLocalReduction(in float value, in uint localId, in IBinaryOp binaryOp)
{
    // Unrolled parallel reduction using shared mem
    sReduceBuffer[localId] = value;
    AllMemoryBarrierWithGroupSync();

    static const uint START = (STATIC_THREAD_PER_BLOCK_1D >> 1);
    [ForceUnroll]
    for (uint i = START; i != 0; i >> 1)
    {
        if (localId < i) sReduceBuffer[localId] = binaryOp.op(sReduceBuffer[localId],
                                                              sReduceBuffer[localId + i]);
        AllMemoryBarrierWithGroupSync();
    }
    return sReduceBuffer[localId];
}

void AtomicFloatOp(out uint outLoc, float val, IBinaryOp binaryOp)
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
        float newValF = binaryOp.op(assumedF, val);
        uint newVal = asuint(newValF);
        InterlockedCompareExchange(outLoc, assumed, newVal, old);

    } while(assumed != old);
}

//======================================//
//     LUMINANCE REDUCTION SHADER       //
//======================================//
[shader("compute")]
[numthreads(STATIC_THREAD_PER_BLOCK_1D, 1, 1)]
void KCFindAvgMaxLum(uint3 globalId: SV_DispatchThreadID,
                     uint3 localId: SV_GroupThreadID,
                     TonemapParameters params)
{
    uint2 dim;
    params.inputTex.GetDimensions(dim[0], dim[1]);
    uint2 pixId = uint2(globalId.x % dim.x,
                        globalId.x / dim.x);

    float3 inColor = params.inputTex.Load(uint3(pixId, 0), int2(0, 0));
    float luminance = ColorConverterIn::ToYxy(inColor)[0];

    // The reduction
    float localAvg = BlockLocalReduction(luminance, localId.x, AddOp());
    float localMax = BlockLocalReduction(luminance, localId.x, MaxOp());

    // Now atomics, (still we need to do comp-sawp idiom for floats...)
    // TODO: If have free time, check the full down reduction
    // (one extra call for this compute shader on an intermediate buffer, not the texture)
    // is faster or slower (probably faster but setting that up is a hassle)
    if (localId.x == 0)
    {
        float old;
        float totalRecip = 1.0f / float(dim.x + dim.y);
        localAvg *= totalRecip;
        AtomicFloatOp(params.maxAndAvg[0].avgLum, localAvg, AddOp());
        AtomicFloatOp(params.maxAndAvg[0].maxLum, localAvg, MaxOp());
    }
}

//======================================//
//           TONEMAP SHADER             //
//======================================//
[shader("compute")]
[numthreads(STATIC_THREAD_PER_BLOCK_1D, 1, 1)]
void KCTonemapImage(uint3 globalId : SV_DispatchThreadID,
                    TonemapParameters params)
{
    uint2 dim;
    params.inputTex.GetDimensions(dim[0], dim[1]);
    uint2 pixId = uint2(globalId.x % dim.x,
                        globalId.x / dim.x);

    //
    float3 inColorHDR = params.inputTex.Load(uint3(pixId, 0), int2(0, 0));

    ColorConverterIn  inColorSpace;
    ColorConverterOut outColorSpace;

    float3 outColorSDR = Tonemapper.ToneMap(inColorHDR,
                                            inColorSpace, outColorSpace,
                                            gammaParamsOut,
                                            params.maxAndAvg[0].ReadAvgLum(),
                                            params.maxAndAvg[0].ReadMaxLum());

    params.outputTex[globalId] = outColorSDR;
}