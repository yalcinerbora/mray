module AccumInput;

// TODO: Parametrize this shader
// For arrayed textures maybe? (For spectral)
static const uint32_t STATIC_THREAD_PER_BLOCK_2D_X = 16;
static const uint32_t STATIC_THREAD_PER_BLOCK_2D_Y = 16;

struct AccumulateUniforms
{
    int2    imgResolution;
    int2    pixStart;
    int2    pixEnd;
    float   globalSampleWeight;
};

struct AccumulateParameters
{
    AccumulateUniforms          U;
    RWTexture2D<float3>         outPixels;
    RWTexture2D<float>          outSamples;
    // Input from host-backed memory
    StructuredBuffer<float3>    inPixels;
    StructuredBuffer<float>     inSamples;
}

ParameterBlock<AccumulateParameters> params;

[shader("compute")]
[numthreads(STATIC_THREAD_PER_BLOCK_2D_X,
            STATIC_THREAD_PER_BLOCK_2D_Y, 1)]
void KCAccumulateInputs(uint3 globalId: SV_DispatchThreadID,
                        uint3 localId: SV_GroupThreadID)
{
    int2 pixIn = int2(globalId.xy);
    int2 pixOut = int2(globalId.xy) + params.U.pixStart;
    int2 pixCount = params.U.pixEnd - params.U.pixStart;

    // Cull threads that is out of range due to block size
    if(any(pixIn >= pixCount)) return;

    // Reads
    int pixLinear = pixIn[1] * pixCount.x + pixIn[0];
    float3 colorIn = params.inPixels[pixLinear].xyz;
    float sampleIn = params.inSamples[pixLinear];
    //
    float3 colorOut = params.outPixels.Load(pixOut);
    float sampleOut = params.outSamples.Load(pixOut);

    // Accumulation
    float inWeight = params.U.globalSampleWeight;
    float totalSample = sampleIn + sampleOut;
    float3 newColor = (sampleOut * colorOut + inWeight * colorIn);
    newColor = newColor / totalSample;

    // Write
    params.outPixels[pixOut] = newColor;
    params.outSamples[pixOut] = totalSample;
}

//StructuredBuffer<float> buffer0;
//StructuredBuffer<float> buffer1;
//RWStructuredBuffer<float> result;

//[shader("compute")]
//[numthreads(1, 1, 1)]
//void KCAccumulateInputs(uint3 threadId: SV_DispatchThreadID)
//{
//    uint index = threadId.x;
//    result[index] = buffer0[index] + buffer1[index];
//}