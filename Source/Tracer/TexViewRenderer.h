#pragma once

#include "RendererC.h"
#include "SpectrumC.h"
#include "Random.h"

class SpectrumContextJakob2019;

class TexViewRenderer final : public RendererBase
{
    public:
    static std::string_view TypeName();
    static AttribInfoList StaticAttributeInfo();

    using GlobalStateList   = TypePack<>;
    using RayStateList      = TypePack<>;
    using SpectrumContext = SpectrumContextIdentity;

    template<PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    using WorkFunctions = TypePack<>;

    template<LightGroupC LG, TransformGroupC TG>
    using LightWorkFunctions = TypePack<>;

    template<CameraGroupC CG, TransformGroupC TG>
    using CamWorkFunctions = TypePack<>;

    enum Mode
    {
        SHOW_TILING,
        SHOW_TEXTURES
    };
    struct Options
    {
        uint32_t    totalSPP   = 16'384;
        bool        isSpectral = false;
        Mode        renderMode = SHOW_TEXTURES;
    };

    private:
    Options     currentOptions  = {};
    Options     newOptions      = {};
    // State
    uint32_t    textureIndex    = 0;
    uint32_t    mipIndex        = 0;
    //
    bool        saveImage;
    //
    std::vector<const GenericTexture*>      textures;
    std::vector<const GenericTextureView*>  textureViews;
    // Spectral Mode related
    std::unique_ptr<SpectrumContextJakob2019>   spectrumContext;
    RNGeneratorPtr                              rnGenerator;
    DeviceMemory                                spectrumMem;
    Span<Spectrum>                              dThroughputs;
    Span<Spectrum>                              dWavelengthPDFs;
    Span<SpectrumWaves>                         dWavelengths;
    Span<RandomNumber>                          dRandomNumbers;

    void    RenderTextureAsData(const GPUQueue& processQueue);
    void    RenderTextureAsSpectral(const GPUQueue& processQueue);

    public:
    // Constructors & Destructor
                        TexViewRenderer(const RenderImagePtr&,
                                        TracerView,
                                        ThreadPool&,
                                        const GPUSystem&,
                                        const RenderWorkPack&);
                        TexViewRenderer(const TexViewRenderer&) = delete;
                        TexViewRenderer(TexViewRenderer&&) = delete;
    TexViewRenderer&    operator=(const TexViewRenderer&) = delete;
    TexViewRenderer&    operator=(TexViewRenderer&&) = delete;

    //
    AttribInfoList      AttributeInfo() const override;
    RendererOptionPack  CurrentAttributes() const override;
    void                PushAttribute(uint32_t attributeIndex,
                                      TransientData data,
                                      const GPUQueue& q) override;
    //
    RenderBufferInfo    StartRender(const RenderImageParams&,
                                    CamSurfaceId camSurfId,
                                    uint32_t customLogicIndex0 = 0,
                                    uint32_t customLogicIndex1 = 0) override;
    RendererOutput      DoRender() override;
    void                StopRender() override;
    size_t              GPUMemoryUsage() const override;
};

static_assert(RendererC<TexViewRenderer>, "\"TexViewRenderer\" does not "
              "satisfy renderer concept.");
