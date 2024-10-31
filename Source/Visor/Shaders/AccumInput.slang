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
    RWTexture2D<float4>         outPixels;
    RWTexture2D<float>          outSamples;
    // Input from host-backed memory
    StructuredBuffer<float>     inPixelsR;
    StructuredBuffer<float>     inPixelsG;
    StructuredBuffer<float>     inPixelsB;
    StructuredBuffer<float>     inSamples;
}

ParameterBlock<AccumulateParameters> params;

[shader("compute")]
[numthreads(STATIC_THREAD_PER_BLOCK_2D_X,
            STATIC_THREAD_PER_BLOCK_2D_Y, 1)]
void KCAccumulateInputs(uint3 globalId: SV_DispatchThreadID,
                        uint3 localId: SV_GroupThreadID)
{
    int2 pixCount = params.U.pixEnd - params.U.pixStart;
    int2 pixIn = int2(globalId.xy);
    int2 pixOut = int2(globalId.xy) + params.U.pixStart;

    // MRay has bottom-left coordinate system (uv),
    // but Vulkan has top-left.
    // (OpenGL has bottom-left, how does GLSL or Slang
    // handle these)
    uint2 resolution;
    params.outPixels.GetDimensions(resolution[0], resolution[1]);
    pixOut.y = resolution.y - 1 - pixOut.y;

    // Cull threads that is out of range due to block size
    if(any(pixIn >= pixCount)) return;

    // Reads
    int pixLinear = pixIn[1] * pixCount.x + pixIn[0];
    float3 colorIn = float3(params.inPixelsR[pixLinear],
                            params.inPixelsG[pixLinear],
                            params.inPixelsB[pixLinear]);
    float sampleIn = (params.inSamples[pixLinear] *
                      params.U.globalSampleWeight);
    //
    float3 colorOut = params.outPixels.Load(pixOut).rgb;
    float sampleOut = params.outSamples.Load(pixOut);
    float totalSample = sampleOut + sampleIn;
    // Accumulation
    // If sample out is exactly zero, this means we probably
    // cleared the image. However; colorOut is not cleared,
    // and may contain NaN's which will corrupt the mathematical
    // clearing ([0 * colorOut + colorIn * sIn] = colorIn * sIn, will
    // not be satisfied due to NaN)
    float3 newColor = colorOut * sampleOut + colorIn;
    if(totalSample == 0)
        newColor = float3(0);
    else
        newColor /= float3(totalSample);

    // Write
    params.outPixels[pixOut] = float4(newColor, 1.0f);
    params.outSamples[pixOut] = totalSample;
}