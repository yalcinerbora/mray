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

class ImageRenderer final : public RendererT<ImageRenderer>
{
    public:
    static std::string_view TypeName();

    struct Options
    {
        uint32_t totalSPP = 32;
    };

    private:
    Options         currentOptions = {};
    Options         newOptions = {};

    public:
    // Constructors & Destructor
                    ImageRenderer(RenderImagePtr, TracerView,
                                  const GPUSystem&);
                    ImageRenderer(const ImageRenderer&) = delete;
                    ImageRenderer(ImageRenderer&&) = delete;
    ImageRenderer&  operator=(const ImageRenderer&) = delete;
    ImageRenderer&  operator=(ImageRenderer&&) = delete;

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
std::string_view ImageRenderer::TypeName()
{
    using namespace std::string_view_literals;
    using namespace TypeNameGen::CompTime;
    static constexpr auto Name = "Image"sv;
    return RendererTypeName<Name>;
}

inline
size_t ImageRenderer::GPUMemoryUsage() const
{
    return 0;
}