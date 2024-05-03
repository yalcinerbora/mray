#pragma once

#include "RendererC.h"

enum class SamplerType : uint32_t
{
    INDEPENDENT
};

class RenderImage
{
    private:
    HostLocalMemory                 imgMemory;
    size_t                          sampleOffset;
    size_t                          pixelOffset;
    Span<Float>                     imgPixels;
    Span<uint32_t>                  imgSamples;

    RenderImageParams               params;
    std::unique_ptr<GPUSemaphore*>  semaphore;

    //
    public:
    // Constructors & Destructor
                    RenderImage();
                    RenderImage(const RenderImage&) = delete;
    RenderImage&    operator=(const RenderImage&) = delete;
    //
    void            Realloc(const RenderImageParams&,
                            bool shrinkToFit = false);

    const RenderImageParams&    Params();
    Span<Float>                 Pixels();
    Span<uint32_t>              Samples();

    void ClearImage(const GPUQueue& queue);
    void AttachSemaphore(SystemSemaphoreHandle s);

    void AcquireAccess(const GPUQueue& queue, uint64_t value);
    void ReleaseAccess(const GPUQueue& queue, uint64_t value);

};

class ImageRenderer : public RendererI
{
    public:
    struct Options
    {
        SamplerType samplerType = SamplerType ::INDEPENDENT;
    };

    private:
    GPUSystem&      gpuSystem;

    public:
    //
                            ImageRenderer(GPUSystem& s);
                            ImageRenderer(const ImageRenderer&) = delete;
                            ImageRenderer(ImageRenderer&&) = delete;
    ImageRenderer&          operator=(const ImageRenderer&) = delete;
    ImageRenderer&          operator=(ImageRenderer&&) = delete;

    virtual void            Commit() = 0;
    virtual bool            IsInCommitState() const = 0;
    virtual AttribInfoList  AttributeInfo() const = 0;
    virtual void            PushAttribute(uint32_t attributeIndex,
                                          TransientData data,
                                          const GPUQueue& q) = 0;
    //
    virtual void                StartRender(const CamSurfaceId&) override;
    virtual void                StopRender() override;
    virtual std::string_view    Name() const override;

    virtual void                AttachOutputImg();
};