#pragma once

#include "RendererC.h"

class TexViewRenderer final : public RendererT<TexViewRenderer>
{
    public:
    static std::string_view TypeName();
    static AttribInfoList StaticAttributeInfo();

    using GlobalStateList   = PackedTypes<>;
    using RayStateList      = PackedTypes<>;
    using SpectrumConverterContext = SpectrumConverterContextIdentity;

    template<PrimitiveC P, MaterialC M, class S, class TContext,
             PrimitiveGroupC PG, MaterialGroupC MG, TransformGroupC TG>
    static constexpr auto WorkFunctions = std::tuple{};

    template<LightC L, LightGroupC LG, TransformGroupC TG>
    static constexpr auto LightWorkFunctions = std::tuple{};

    template<CameraC Camera, CameraGroupC CG, TransformGroupC TG>
    static constexpr auto CamWorkFunctions = std::tuple{};

    enum Mode
    {
        SHOW_TILING,
        SHOW_TEXTURES
    };
    struct Options
    {
        uint32_t totalSPP   = 32;
        Mode     mode       = SHOW_TEXTURES;
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
    std::vector<const GenericTexture*> textures;
    std::vector<const GenericTextureView*> textureViews;

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
