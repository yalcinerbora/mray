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
        SamplerType samplerType = SamplerType ::INDEPENDENT;
    };

    private:
    Options                 currentOptions = {};
    Options                 newOptions = {};

    public:
    // Constructors & Destructor
                            ImageRenderer(const GPUSystem& s);
                            ImageRenderer(const ImageRenderer&) = delete;
                            ImageRenderer(ImageRenderer&&) = delete;
    ImageRenderer&          operator=(const ImageRenderer&) = delete;
    ImageRenderer&          operator=(ImageRenderer&&) = delete;

    //
    MRayError       Commit() override;
    bool            IsInCommitState() const override;
    AttribInfoList  AttributeInfo() const override;
    void            PushAttribute(uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& q) override;
    //


    //
    RenderBufferInfo    StartRender(const RenderImageParams&,
                                    const CameraKey&) override;
    RendererOutput      DoRender() override;
};

inline
std::string_view ImageRenderer::TypeName()
{
    using namespace std::string_view_literals;
    using namespace TypeNameGen::CompTime;
    static constexpr auto Name = "Image"sv;
    return RendererTypeName<Name>;
}