#include "SurfaceRenderer.h"
#include "Core/Error.hpp"

SurfaceRenderer::SurfaceRenderer(const RenderImagePtr& rb,
                                 TracerView tv, const GPUSystem& s)
    : RendererT(rb, tv, s)
{}

MRayError SurfaceRenderer::Commit()
{
    if(rendering)
        currentOptions = newOptions;
    return MRayError::OK;
}

typename SurfaceRenderer::AttribInfoList
SurfaceRenderer::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeIsArray;
    using enum AttributeOptionality;
    return AttribInfoList
    {
        {"totalSPP", MRayDataType<MR_UINT32>{}, IS_SCALAR, MR_MANDATORY}
    };
}

RendererOptionPack SurfaceRenderer::CurrentAttributes() const
{
    RendererOptionPack result;
    result.paramTypes = AttributeInfo();

    result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    result.attributes.back().Push(Span<const uint32_t>(&currentOptions.totalSPP, 1));

    if constexpr(MRAY_IS_DEBUG)
    {
        for(const auto& d: result.attributes)
            assert(d.IsFull());
    }
    return result;
}

void SurfaceRenderer::PushAttribute(uint32_t attributeIndex,
                                    TransientData data, const GPUQueue&)
{
    if(attributeIndex != 0)
        throw MRayError("{} Unkown attribute index {}",
                        TypeName(), attributeIndex);
    newOptions.totalSPP = data.AccessAs<uint32_t>()[0];
}

RenderBufferInfo SurfaceRenderer::StartRender(const RenderImageParams& rip,
                                              CamSurfaceId camSurfId,
                                              Optional<CameraTransform> optTransform,
                                              uint32_t customLogicIndex0,
                                              uint32_t)
{
    transOverride = optTransform;
    curCamSurfaceId = camSurfId;
    rIParams = rip;
    //
    // Find the texture index
    using MathFunctions::Roll;
    uint32_t newMode = uint32_t(Roll(int32_t(customLogicIndex0), 0,
                                             int32_t(Mode::END)));
    currentOptions.mode = Mode(newMode);

    // Calculate tile size according to the parallelization hint
    uint32_t parallelHint = tracerView.tracerParams.parallelizationHint;
    Vector2ui imgRegion = rIParams.regionMax - rIParams.regionMin;
    Vector2ui tileSize = FindOptimumTile(imgRegion, parallelHint);
    renderBuffer->Resize(tileSize, 1, 3);
    tileCount = MathFunctions::DivideUp(imgRegion, tileSize);

    curColorSpace = tracerView.tracerParams.globalTextureColorSpace;
    curFramebufferSize = rIParams.resolution;
    curFBMin = rIParams.regionMin;
    curFBMax = rIParams.regionMax;
    RenderBufferInfo rbI = renderBuffer->GetBufferInfo(curColorSpace,
                                                       curFramebufferSize, 1);
    rbI.curRenderLogic0 = newMode;
    rbI.curRenderLogic1 = 0;

    // Generate works per Material/Primitive/Transform
    // triplet
    Mode m = currentOptions.mode;
    if(m >= Mode::END)
        throw MRayError("[(R)Surface]: Unkown render mode \"{}\"");

    uint32_t packIndex = static_cast<uint32_t>(m);
    GenerateWorkMappings(packIndex);
    GenerateLightWorkMappings(packIndex);
    GenerateCameraWorkMappings(packIndex);


    return rbI;

}

RendererOutput SurfaceRenderer::DoRender()
{
    // Each iteration do one tile fully,
    // so we can send it directly

    // Generate rays
    // Cast rays
    // Select work





    //...


    return RendererOutput
    {
        //.analytics = RendererAnalyticData
        //{
        //    samplePerSec,
        //    "M samples/s",
        //    spp,
        //    "spp",
        //    float(timer.Elapsed<Millisecond>()),
        //    mipSize,
        //    MRayColorSpaceEnum::MR_ACES_CG,
        //    static_cast<uint32_t>(textures.size()),
        //    static_cast<uint32_t>(curTex->MipCount())
        //},
        //.imageOut = renderOut
    };
}

void SurfaceRenderer::StopRender()
{}