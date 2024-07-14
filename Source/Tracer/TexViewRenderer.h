#pragma once

#include "RendererC.h"
#include "Core/TypeGenFunction.h"
#include "Core/TypeNameGenerators.h"

//using MatWorkGenerator = GeneratorFuncType<GenericWorkBatch,
//                                           //
//                                           GenericGroupTransformT*,
//                                           GenericGroupMaterialT*,
//                                           GenericGroupPrimitiveT*,
//                                           GPUSystem&>;
//
//using CamWorkGenerator = GeneratorFuncType<GenericGroupTransformT*,
//                                           GenericGroupCameraT*,
//                                           GPUSystem&>;
//
//using LightWorkGenerator = GeneratorFuncType<GenericGroupTransformT*,
//                                             GenericGroupLightT*,
//                                             GPUSystem&>;

class TexViewRenderer final : public RendererT<TexViewRenderer>
{
    public:
    static std::string_view TypeName();

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
    uint32_t    curTileIndex    = 0;
    uint32_t    textureIndex    = 0;
    uint32_t    mipIndex        = 0;
    Vector2ui   mipSize         = Vector2ui::Zero();
    Vector2ui   tileCount       = Vector2ui::Zero();
    //
    std::vector<const CommonTexture*> textures;

    public:
    // Constructors & Destructor
                        TexViewRenderer(const RenderImagePtr&,
                                        TracerView, const GPUSystem&);
                        TexViewRenderer(const TexViewRenderer&) = delete;
                        TexViewRenderer(TexViewRenderer&&) = delete;
    TexViewRenderer&    operator=(const TexViewRenderer&) = delete;
    TexViewRenderer&    operator=(TexViewRenderer&&) = delete;

    //
    MRayError           Commit() override;
    AttribInfoList      AttributeInfo() const override;
    RendererOptionPack  CurrentAttributes() const override;
    void                PushAttribute(uint32_t attributeIndex,
                                      TransientData data,
                                      const GPUQueue& q) override;
    //


    //
    RenderBufferInfo    StartRender(const RenderImageParams&,
                                    const CameraKey&,
                                    uint32_t customLogicIndex0 = 0,
                                    uint32_t customLogicIndex1 = 0) override;
    RendererOutput      DoRender() override;
    void                StopRender() override;
    size_t              GPUMemoryUsage() const override;
};

inline
std::string_view TexViewRenderer::TypeName()
{
    using namespace std::string_view_literals;
    using namespace TypeNameGen::CompTime;
    static constexpr auto Name = "TexView"sv;
    return RendererTypeName<Name>;
}

inline
size_t TexViewRenderer::GPUMemoryUsage() const
{
    return 0;
}