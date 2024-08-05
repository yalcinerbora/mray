#pragma once

#include "RendererC.h"
#include "Core/TypeNameGenerators.h"

class SurfaceRenderer final : public RendererT<SurfaceRenderer>
{
    public:
    static std::string_view TypeName();

    enum Mode
    {
        FURNACE,
        WORLD_NORMAL,
        WORLD_POSITION,
        AO,
        //
        END
    };
    struct Options
    {
        uint32_t totalSPP   = 32;
        Mode     mode       = FURNACE;
    };

    private:
    Options     currentOptions  = {};
    Options     newOptions      = {};
    //
    uint32_t    curTileIndex    = 0;
    Vector2ui   tileCount       = Vector2ui::Zero();
    //
    RenderImageParams           rIParams  = {};
    Optional<CameraTransform>   transOverride = {};
    CamSurfaceId                curCamSurfaceId = CamSurfaceId(0);
    //

    public:
    // Constructors & Destructor
                        SurfaceRenderer(const RenderImagePtr&,
                                        TracerView, const GPUSystem&);
                        SurfaceRenderer(const SurfaceRenderer&) = delete;
                        SurfaceRenderer(SurfaceRenderer&&) = delete;
    SurfaceRenderer&    operator=(const SurfaceRenderer&) = delete;
    SurfaceRenderer&    operator=(SurfaceRenderer&&) = delete;

    //
    MRayError           Commit() override;
    AttribInfoList      AttributeInfo() const override;
    RendererOptionPack  CurrentAttributes() const override;
    void                PushAttribute(uint32_t attributeIndex,
                                      TransientData data,
                                      const GPUQueue& q) override;
    //
    RenderBufferInfo    StartRender(const RenderImageParams&,
                                    CamSurfaceId camSurfId,
                                    Optional<CameraTransform>,
                                    uint32_t customLogicIndex0 = 0,
                                    uint32_t customLogicIndex1 = 0) override;
    RendererOutput      DoRender() override;
    void                StopRender() override;
    size_t              GPUMemoryUsage() const override;
};

inline
std::string_view SurfaceRenderer::TypeName()
{
    using namespace std::string_view_literals;
    using namespace TypeNameGen::CompTime;
    static constexpr auto Name = "Surface"sv;
    return RendererTypeName<Name>;
}

inline
size_t SurfaceRenderer::GPUMemoryUsage() const
{
    return 0;
}