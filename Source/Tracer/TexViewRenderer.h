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

    struct Options
    {
        uint32_t totalSPP = 32;
    };

    private:
    Options     currentOptions = {};
    Options     newOptions = {};
    // TEST
    uint32_t    pixelIndex = 0;

    public:
    // Constructors & Destructor
                        TexViewRenderer(const RenderImagePtr&, TracerView,
                                        const GPUSystem&);
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
                                    const CameraKey&) override;
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