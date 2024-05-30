#include "ImageRenderer.h"

ImageRenderer::ImageRenderer(const GPUSystem& s)
    : RendererT(s)
{}

MRayError ImageRenderer::Commit()
{
    if(rendering)
    currentOptions = newOptions;
    return MRayError::OK;
}

bool ImageRenderer::IsInCommitState() const
{
    return false;
}

typename ImageRenderer::AttribInfoList
ImageRenderer::AttributeInfo() const
{
    return AttribInfoList{};
}

void ImageRenderer::PushAttribute(uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& q)
{

}

RenderBufferInfo ImageRenderer::StartRender(const RenderImageParams& params,
                                            const CameraKey&)
{
    renderBuffer.reset(new RenderImageBuffer(params.semaphore, params.initialSemCounter,
                                             gpuSystem));
    renderBuffer->Resize(params.resolution, params.regionMin,
                         params.regionMax, 1, MRayColorSpaceEnum::MR_ACES_CG);

    return RenderBufferInfo{};
}

RendererOutput ImageRenderer::DoRender()
{
    return RendererOutput{};
}