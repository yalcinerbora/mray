#include "TexViewRenderer.h"
#include "TracerBase.h"

#include "Device/GPUSystem.hpp"

TexViewRenderer::TexViewRenderer(RenderImagePtr rb, TracerView tv,
                             const GPUSystem& s)
    : RendererT(rb, tv, s)
{}

MRayError TexViewRenderer::Commit()
{
    if(rendering)
    currentOptions = newOptions;
    return MRayError::OK;
}

typename TexViewRenderer::AttribInfoList
TexViewRenderer::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeIsArray;
    using enum AttributeOptionality;
    return AttribInfoList
    {
        {"totalSPP", MRayDataType<MR_UINT32>{}, IS_SCALAR, MR_MANDATORY}
    };
}

RendererOptionPack TexViewRenderer::CurrentAttributes() const
{
    RendererOptionPack result;
    result.paramTypes = AttributeInfo();

    result.attributes.push_back(TransientData(std::in_place_type_t<uint32_t>{}, 1));
    result.attributes.back().Push(Span<const uint32_t>(&currentOptions.totalSPP, 1));
    return result;
}

void TexViewRenderer::PushAttribute(uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& q)
{
    if(attributeIndex != 0)
        throw MRayError("{} Unkown attribute index {}",
                        TypeName(), attributeIndex);
    newOptions.totalSPP = data.AccessAs<uint32_t>()[0];
}

RenderBufferInfo TexViewRenderer::StartRender(const RenderImageParams& params,
                                              const CameraKey&)
{
    // Reset renderbuffer

    using enum MRayColorSpaceEnum;
    renderBuffer = std::make_shared<RenderImage>(params, 1,
                                                 MR_ACES_CG,
                                                 gpuSystem);
    return renderBuffer->GetBufferInfo();
}

MRAY_KERNEL
void TestWrite(MRAY_GRID_CONSTANT const Span<Float> pixels,
               MRAY_GRID_CONSTANT const Span<Float> samples,
               MRAY_GRID_CONSTANT const uint32_t colorIndex)
{
    KernelCallParams kp;
    uint32_t pixelChannels = static_cast<uint32_t>(pixels.size());
    for(uint32_t i = kp.GlobalId(); i < pixelChannels; i += kp.TotalSize())
    {
        if(i < samples.size())
            samples[i] = 1.0;
        pixels[i] = (colorIndex % 2 == 0) ? 1.0f : 0.0f;
    }
}

RendererOutput TexViewRenderer::DoRender()
{
    using namespace std::literals;
    //std::this_thread::sleep_for(200ms);

    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    Span<Float> pixels = renderBuffer->Pixels();
    Span<Float> samples = renderBuffer->Samples();
    uint32_t colorIndex = pixelIndex;


    renderBuffer->AcquireImage(queue);
    queue.IssueSaturatingKernel<TestWrite>
    (
        "TexTest"sv,
        KernelIssueParams{.workCount = static_cast<uint32_t>(pixels.size())},
        //
        pixels,
        samples,
        colorIndex
    );
    RenderImageSection renderOut = renderBuffer->ReleaseImage(queue);

    //queue.Barrier().Wait();

    //t.Split();
    //MRAY_LOG("MEMCPY {}", t.Elapsed<Millisecond>());

    pixelIndex++;
    return RendererOutput
    {
        .analytics = RendererAnalyticData
        {
            3.2,
            "M paths/s",
            0.0,
            "spp",
            0.0,
            renderBuffer->Resolution(),
            MRayColorSpaceEnum::MR_ACES_CG
        },
        .imageOut = renderOut
    };
}

void TexViewRenderer::StopRender()
{}