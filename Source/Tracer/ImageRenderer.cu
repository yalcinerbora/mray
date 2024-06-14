#include "ImageRenderer.h"
#include "TracerBase.h"

ImageRenderer::ImageRenderer(RenderImagePtr rb, TracerView tv,
                             const GPUSystem& s)
    : RendererT(rb, tv, s)
{}

MRayError ImageRenderer::Commit()
{
    if(rendering)
    currentOptions = newOptions;
    return MRayError::OK;
}

typename ImageRenderer::AttribInfoList
ImageRenderer::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeIsArray;
    using enum AttributeOptionality;
    return AttribInfoList
    {
        {"totalSPP", MRayDataType<MR_UINT32>{}, IS_SCALAR, MR_MANDATORY}
    };
}

RendererOptionPack ImageRenderer::CurrentAttributes() const
{
    RendererOptionPack result;
    result.paramTypes = AttributeInfo();

    result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    result.attributes.back().Push(Span<const uint32_t>(&currentOptions.totalSPP, 1));
    return result;
}

void ImageRenderer::PushAttribute(uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& q)
{
    if(attributeIndex != 0)
        throw MRayError("{} Unkown attribute index {}",
                        TypeName(), attributeIndex);
    newOptions.totalSPP = data.AccessAs<uint32_t>()[0];
}

RenderBufferInfo ImageRenderer::StartRender(const RenderImageParams& params,
                                            const CameraKey&)
{
    // Reset renderbuffer
    using enum MRayColorSpaceEnum;
    renderBuffer = std::make_shared<RenderImage>(params, 1,
                                                 MR_ACES_CG,
                                                 gpuSystem);

    return renderBuffer->GetBufferInfo();
}

RendererOutput ImageRenderer::DoRender()
{
    return RendererOutput{};
}

void ImageRenderer::StopRender()
{}