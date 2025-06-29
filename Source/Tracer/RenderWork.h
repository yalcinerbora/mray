#pragma once

#include "RendererC.h"
#include "TransformC.h"
#include "MaterialC.h"
#include "PrimitiveC.h"
#include "TracerTypes.h"
#include "Hit.h"
#include "Random.h"
#include "RayGenKernels.h"

#include "Core/TracerI.h"

// Render work kernel parameters
// There are too many parameters so these are
// packed in structs
struct RenderWorkInputs
{
    // Contiguous
    Span<const RayIndex>    dRayIndices;
    Span<const uint32_t>    dRandomNumbers;
    // Accesed by index
    Span<const RayCone>     dRayCones;
    Span<const RayGMem>     dRays;
    Span<const MetaHit>     dHits;
    Span<const HitKeyPack>  dKeys;
};

template<class GlobalState, class RayState,
         PrimitiveGroupC  PrimGroup,
         MaterialGroupC MatGroup, TransformGroupC TransGroup>
struct RenderWorkParams
{
    using PrimSoA       = typename PrimGroup::DataSoA;
    using MatSoA        = typename MatGroup::DataSoA;
    using TransSoA      = typename TransGroup::DataSoA;
    //
    using Inputs        = RenderWorkInputs;
    //
    //Outputs     out;
    RayState    rayState;
    Inputs      in;
    GlobalState globalState;
    PrimSoA     primSoA;
    MatSoA      matSoA;
    TransSoA    transSoA;
};

template<class GlobalState, class RayState,
         LightGroupC LightGroup, TransformGroupC TransGroup>
struct RenderLightWorkParams
{
    using PrimSoA   = typename LightGroup::PrimGroup::DataSoA;
    using LightSoA  = typename LightGroup::DataSoA;
    using TransSoA  = typename TransGroup::DataSoA;
    //
    using Inputs        = RenderWorkInputs;
    //
    RayState    rayState;
    Inputs      in;
    GlobalState globalState;
    PrimSoA     primSoA;
    LightSoA    lightSoA;
    TransSoA    transSoA;
};

template<class GlobalState, class RayState,
        CameraGroupC CamGroup, TransformGroupC TransGroup>
struct RenderCameraWorkParams
{
    using CamSoA    = typename CamGroup::DataSoA;
    using TransSoA  = typename TransGroup::DataSoA;
    //
    using Inputs = RenderWorkInputs;
    //
    RayState    rayState;
    Inputs      in;
    GlobalState globalState;
    CamSoA      camsSoA;
    TransSoA    transSoA;
};

// Some aliases for clarity
template<class Renderer, uint32_t I, PrimitiveGroupC PG,
         MaterialGroupC MG, TransformGroupC TG>
using RenderWorkParamsR = RenderWorkParams
<
    RenderGlobalState<Renderer, I>,
    RenderRayState<Renderer, I>,
    PG, MG, TG
>;
template<class Renderer, uint32_t I,
         LightGroupC LG, TransformGroupC TG>
using RenderLightWorkParamsR = RenderLightWorkParams
<
    RenderGlobalState<Renderer, I>,
    RenderRayState<Renderer, I>,
    LG, TG
>;

template<class Renderer, uint32_t I,
         CameraGroupC CG, TransformGroupC TG>
using RenderCameraWorkParamsR = RenderCameraWorkParams
<
    RenderGlobalState<Renderer, I>,
    RenderRayState<Renderer, I>,
    CG, TG
>;

template<RendererC R, PrimitiveGroupC PG,
         MaterialGroupC MG, TransformGroupC TG>
class RenderWork : public RenderWorkT<R>
{
    public:
    static std::string_view TypeName();

    private:
    const MG&           mg;
    const PG&           pg;
    const TG&           tg;
    const GPUSystem&    gpuSystem;

    template<uint32_t I>
    void DoWorkInternal(// I-O
                        const RenderRayState<R, I>& dRayStates,
                        // Input
                        // Contiguous
                        Span<const RayIndex> dRayIndicesIn,
                        Span<const RandomNumber> dRandomNumbers,
                        // Accessed by index
                        Span<const RayCone> dRayDiffsIn,
                        Span<const RayGMem> dRaysIn,
                        Span<const MetaHit> dHitsIn,
                        Span<const HitKeyPack> dKeysIn,
                        // Constants
                        const RenderGlobalState<R, I>& globalState,
                        const GPUQueue& queue) const;

    public:
    // Constructors & Destructor
            RenderWork(const GenericGroupMaterialT&,
                       const GenericGroupPrimitiveT&,
                       const GenericGroupTransformT&,
                       const GPUSystem&);
    //
    MRAY_RENDER_DO_WORK_DEF(0)
    MRAY_RENDER_DO_WORK_DEF(1)


    uint32_t         SampleRNCount(uint32_t workIndex) const override;
    std::string_view Name() const override;
};

template<RendererC R, LightGroupC LG, TransformGroupC TG>
class RenderLightWork : public RenderLightWorkT<R>
{
    public:
    static std::string_view TypeName();

    private:
    const LG&           lg;
    const TG&           tg;
    const GPUSystem&    gpuSystem;

    template<uint32_t I>
    void    DoBoundaryWorkInternal(// I-O
                                   const RenderRayState<R, I>& dRayStates,
                                   // Input
                                   // Contiguous
                                   Span<const RayIndex> dRayIndicesIn,
                                   Span<const uint32_t> dRandomNumbers,
                                   // Accessed by index
                                   Span<const RayCone> dRayDiffsIn,
                                   Span<const RayGMem> dRaysIn,
                                   Span<const MetaHit> dHitsIn,
                                   Span<const HitKeyPack> dKeysIn,
                                   // Constants
                                   const RenderGlobalState<R, I>& globalState,
                                   const GPUQueue& queue) const;

    public:
    // Constructors & Destructor
            RenderLightWork(const GenericGroupLightT&,
                            const GenericGroupTransformT&,
                            const GPUSystem&);
    //
    MRAY_RENDER_DO_LIGHT_WORK_DEF(0)
    MRAY_RENDER_DO_LIGHT_WORK_DEF(1)

    uint32_t            SampleRNCount(uint32_t workIndex) const override;
    std::string_view    Name() const override;
};

template<RendererC R, CameraGroupC CG, TransformGroupC TG>
class RenderCameraWork : public RenderCameraWorkT<R>
{
    public:
    static std::string_view TypeName();

    private:
    const CG&           cg;
    const TG&           tg;
    const GPUSystem&    gpuSystem;

    public:
    // Constructors & Destructor
            RenderCameraWork(const GenericGroupCameraT&,
                             const GenericGroupTransformT&,
                             const GPUSystem&);
    //
    void    GenerateSubCamera(// Output
                              Span<Byte> dCamBuffer,
                              // Constants
                              CameraKey camKey,
                              Optional<CameraTransform> camTransform,
                              Vector2ui stratumIndex,
                              Vector2ui stratumCount,
                              const GPUQueue& queue) const override;

    void    GenerateRays(// Output
                         const Span<RayCone>& dRayDiffsOut,
                         const Span<RayGMem>& dRaysOut,
                         const Span<ImageCoordinate>& dImageCoordsOut,
                         const Span<Float>& dSampleWeightsOut,
                         // Input
                         const Span<const uint32_t>& dRayIndices,
                         const Span<const uint32_t>& dRandomNums,
                         // Type erased buffer
                         Span<const Byte> dCamBuffer,
                         TransformKey transKey,
                         // Constants
                         uint64_t globalPixelIndex,
                         const Vector2ui regionCount,
                         const GPUQueue& queue) const override;
    void    GenRaysStochasticFilter(// Output
                                    const Span<RayCone>& dRayDiffsOut,
                                    const Span<RayGMem>& dRaysOut,
                                    const Span<ImageCoordinate>& dImageCoordsOut,
                                    const Span<Float>& dSampleWeightsOut,
                                    // Input
                                    const Span<const uint32_t>& dRayIndices,
                                    const Span<const uint32_t>& dRandomNums,
                                    // Type erased buffer
                                    Span<const Byte> dCamBuffer,
                                    TransformKey transKey,
                                    // Constants
                                    uint64_t globalPixelIndex,
                                    const Vector2ui regionCount,
                                    FilterType filterType,
                                    const GPUQueue& queue) const override;

    std::string_view    Name() const override;
    uint32_t            SampleRayRNCount() const override;
    uint32_t            StochasticFilterSampleRayRNCount() const override;
};

template<RendererC R, uint32_t I, PrimitiveGroupC PG,
         MaterialGroupC MG, TransformGroupC TG,
         auto WorkFunction,
         auto GenerateTransformContext = MRAY_PRIM_TGEN_FUNCTION(PG, TG)>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCRenderWork(MRAY_GRID_CONSTANT const RenderWorkParamsR<R, I, PG, MG, TG> params);

template<RendererC R, uint32_t I, LightGroupC LG, TransformGroupC TG,
         auto WorkFunction,
         auto GenerateTransformContext = MRAY_PRIM_TGEN_FUNCTION(typename LG::PrimGroup, TG)>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCRenderLightWork(MRAY_GRID_CONSTANT const RenderLightWorkParamsR<R, I, LG, TG> params);

template<RendererC R, uint32_t I, PrimitiveGroupC PG,
         CameraGroupC CG, TransformGroupC TG,
         auto WorkFunction>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCRenderCameraWork(MRAY_GRID_CONSTANT const RenderCameraWorkParamsR<R, I, CG, TG> params);

template<RendererC R, PrimitiveGroupC PG,
         MaterialGroupC MG, TransformGroupC TG>
std::string_view RenderWork<R, PG, MG, TG>::TypeName()
{
    using namespace TypeNameGen::CompTime;
    static const std::string name = RenderWorkTypeName(MG::TypeName(),
                                                       PG::TypeName(),
                                                       TG::TypeName());
    return name;
}

template<RendererC R, PrimitiveGroupC P,
         MaterialGroupC M, TransformGroupC T>
RenderWork<R, P, M, T>::RenderWork(const GenericGroupMaterialT& mgIn,
                                   const GenericGroupPrimitiveT& pgIn,
                                   const GenericGroupTransformT& tgIn,
                                   const GPUSystem& sys)
    : mg(static_cast<const M&>(mgIn))
    , pg(static_cast<const P&>(pgIn))
    , tg(static_cast<const T&>(tgIn))
    , gpuSystem(sys)
{}

template<RendererC R, PrimitiveGroupC PG,
         MaterialGroupC MG, TransformGroupC TG>
template<uint32_t I>
void RenderWork<R, PG, MG, TG>::DoWorkInternal(// I-O
                                               const RenderRayState<R, I>& dRayStates,
                                               // Input
                                               // Contiguous
                                               Span<const RayIndex> dRayIndicesIn,
                                               Span<const RandomNumber> dRandomNumbers,
                                               // Accessed by index
                                               Span<const RayCone> dRayDiffsIn,
                                               Span<const RayGMem> dRaysIn,
                                               Span<const MetaHit> dHitsIn,
                                               Span<const HitKeyPack> dKeysIn,
                                               // Constants
                                               const RenderGlobalState<R, I>& globalState,
                                               const GPUQueue& queue) const
{
    // Please check the kernel for details
    using TC = typename PrimTransformContextType<PG, TG>::Result;
    using M = typename MG:: template Material<typename R::SpectrumConverterContext>;
    using P = typename PG:: template Primitive<TC>;
    using S = typename M::Surface;
    static constexpr auto WF = R::template WorkFunctions<P, M, S, TC, PG, MG, TG>;

    if constexpr(I >= std::tuple_size_v<decltype(WF)>)
    {
        throw MRayError("[{}]: Runtime call to \"DoWork_{}\" which does not have a kernel "
                        "associated with it!", R::TypeName(), I);
    }
    else
    {
        using GlobalState = RenderGlobalState<R, I>;
        using RayState = RenderRayState<R, I>;

        const RenderWorkParams<GlobalState, RayState, PG, MG, TG> params =
        {
            .rayState = dRayStates,
            .in =
            {
                .dRayIndices    = dRayIndicesIn,
                .dRandomNumbers = dRandomNumbers,
                .dRayCones      = dRayDiffsIn,
                .dRays          = dRaysIn,
                .dHits          = dHitsIn,
                .dKeys          = dKeysIn
            },
            .globalState = globalState,
            .primSoA        = pg.SoA(),
            .matSoA         = mg.SoA(),
            .transSoA       = tg.SoA()
        };

        uint32_t rayCount = static_cast<uint32_t>(params.in.dRayIndices.size());
        using namespace std::string_literals;
        static const std::string KernelName = std::string(TypeName()) + "-Work"s;
        static constexpr auto WorkFunc = std::get<I>(WF);
        static constexpr auto Kernel = KCRenderWork<R, I, PG, MG, TG, WorkFunc>;
        queue.IssueWorkKernel<Kernel>
        (
            KernelName,
            DeviceWorkIssueParams{.workCount = rayCount},
            params
        );
    }
}

template<RendererC R, PrimitiveGroupC P,
         MaterialGroupC M, TransformGroupC T>
uint32_t RenderWork<R, P, M, T>::SampleRNCount(uint32_t) const
{
    return 0;
}

template<RendererC R, PrimitiveGroupC P,
         MaterialGroupC M, TransformGroupC T>
std::string_view RenderWork<R, P, M, T>::Name() const
{
    return TypeName();
}

template<RendererC R, LightGroupC L, TransformGroupC T>
std::string_view RenderLightWork<R, L, T>::TypeName()
{
    using namespace TypeNameGen::CompTime;
    static const
    std::string name = RenderLightWorkTypeName(L::TypeName(), T::TypeName());
    return name;
}

template<RendererC R, LightGroupC L, TransformGroupC T>
RenderLightWork<R, L, T>::RenderLightWork(const GenericGroupLightT& l,
                 const GenericGroupTransformT& t,
                 const GPUSystem& sys)
    : lg(static_cast<const L&>(l))
    , tg(static_cast<const T&>(t))
    , gpuSystem(sys)
{}

template<RendererC R, LightGroupC LG, TransformGroupC TG>
template<uint32_t I>
void RenderLightWork<R, LG, TG>::DoBoundaryWorkInternal(// I-O
                                                        const RenderRayState<R, I>& dRayStates,
                                                        // Input
                                                        // Contiguous
                                                        Span<const RayIndex> dRayIndicesIn,
                                                        Span<const uint32_t> dRandomNumbers,
                                                        // Accessed by index
                                                        Span<const RayCone> dRayDiffsIn,
                                                        Span<const RayGMem> dRaysIn,
                                                        Span<const MetaHit> dHitsIn,
                                                        Span<const HitKeyPack> dKeysIn,
                                                        // Constants
                                                        const RenderGlobalState<R, I>& globalState,
                                                        const GPUQueue& queue) const
{
    // Please check the kernel for details
    using PG    = typename LG::PrimGroup;
    using TC    = typename PrimTransformContextType<PG, TG>::Result;
    using L     = typename LG::template Light<TC, typename R::SpectrumConverterContext>;
    static constexpr auto WF = R:: template LightWorkFunctions<L, LG, TG>;

    if constexpr(I >= std::tuple_size_v<decltype(WF)>)
    {
        throw MRayError("[{}]: Runtime call to \"DoBoundaryWork_{}\" which does not have a kernel "
                        "associated with it!", R::TypeName(), I);
    }
    else
    {
        using GlobalState = RenderGlobalState<R, I>;
        using RayState = RenderRayState<R, I>;

        const auto& pg = lg.PrimitiveGroup();
        const RenderLightWorkParams<GlobalState, RayState, LG, TG> params =
        {
            .rayState = dRayStates,
            .in =
            {
                .dRayIndices    = dRayIndicesIn,
                .dRandomNumbers = dRandomNumbers,
                .dRayCones      = dRayDiffsIn,
                .dRays          = dRaysIn,
                .dHits          = dHitsIn,
                .dKeys          = dKeysIn
            },
            .globalState    = globalState,
            .primSoA        = pg.SoA(),
            .lightSoA       = lg.SoA(),
            .transSoA       = tg.SoA()
        };

        uint32_t rayCount = static_cast<uint32_t>(params.in.dRayIndices.size());
        using namespace std::string_literals;
        static const std::string KernelName = std::string(TypeName()) + "-BoundaryWork"s;
        static constexpr auto WorkFunc = std::get<I>(WF);
        static constexpr auto Kernel = KCRenderLightWork<R, I, LG, TG, WorkFunc>;
        queue.IssueWorkKernel<Kernel>
        (
            TypeName(),
            DeviceWorkIssueParams{.workCount = rayCount},
            params
        );
    }
}

template<RendererC R, LightGroupC L, TransformGroupC T>
uint32_t RenderLightWork<R, L, T>::SampleRNCount(uint32_t) const
{
    return 0;
}

template<RendererC R, LightGroupC L, TransformGroupC T>
std::string_view RenderLightWork<R, L, T>::Name() const
{
    return TypeName();
}

template<RendererC R, CameraGroupC C, TransformGroupC T>
std::string_view RenderCameraWork<R, C, T>::TypeName()
{
    using namespace TypeNameGen::CompTime;
    static const
    std::string name = RenderCameraWorkTypeName(C::TypeName(), T::TypeName());
    return name;
}

template<RendererC R, CameraGroupC C, TransformGroupC T>
RenderCameraWork<R, C, T>::RenderCameraWork(const GenericGroupCameraT& c,
                                            const GenericGroupTransformT& t,
                                            const GPUSystem& sys)
    : cg(static_cast<const C&>(c))
    , tg(static_cast<const T&>(t))
    , gpuSystem(sys)
{}

template<RendererC R, CameraGroupC C, TransformGroupC T>
void RenderCameraWork<R, C, T>::GenerateSubCamera(// Output
                                                  Span<Byte> dCamBuffer,
                                                  // Constants
                                                  CameraKey camKey,
                                                  Optional<CameraTransform> camTransform,
                                                  Vector2ui stratumIndex,
                                                  Vector2ui stratumCount,
                                                  const GPUQueue& queue) const
{
    using Camera = typename C::Camera;
    assert(dCamBuffer.size_bytes() >= sizeof(Camera));
    assert(uintptr_t(dCamBuffer.data()) % alignof(Camera) == 0);
    Camera* dCamera = reinterpret_cast<Camera*>(dCamBuffer.data());

    static constexpr auto Kernel = KCGenerateSubCamera<C, T>;
    using namespace std::string_literals;
    static const std::string KernelName = std::string(TypeName()) + "-GenSubCam"s;
    //
    queue.IssueBlockKernel<Kernel>
    (
        KernelName,
        DeviceBlockIssueParams{.gridSize = 1, .blockSize = 1},
        // Out
        dCamera,
        // Constants
        camKey,
        cg.SoA(),
        camTransform,
        // In
        stratumIndex,
        stratumCount
    );
}

template<RendererC R, CameraGroupC C, TransformGroupC T>
void RenderCameraWork<R, C, T>::GenerateRays(// Output
                                             const Span<RayCone>& dRayDiffsOut,
                                             const Span<RayGMem>& dRaysOut,
                                             const Span<ImageCoordinate>& dImageCoordsOut,
                                             const Span<Float>& dSampleWeightsOut,
                                             // Input
                                             const Span<const uint32_t>& dRayIndices,
                                             const Span<const uint32_t>& dRandomNums,
                                             // The actual pair to be used
                                             Span<const Byte> dCamBuffer,
                                             TransformKey transKey,
                                             // Constants
                                             uint64_t globalPixelIndex,
                                             const Vector2ui regionCount,
                                             const GPUQueue& queue) const
{
    using Camera = typename C::Camera;
    assert(dRayIndices.size() * Camera::SampleRayRNCount == dRandomNums.size());
    assert(sizeof(Camera) <= dCamBuffer.size_bytes());
    assert(uintptr_t(dCamBuffer.data()) % alignof(Camera) == 0);
    const Camera* dCamera = reinterpret_cast<const Camera*>(dCamBuffer.data());

    uint32_t rayCount = static_cast<uint32_t>(dRayIndices.size());
    static constexpr auto Kernel = KCGenerateCamRays<Camera, T>;
    using namespace std::string_literals;
    static const std::string KernelName = std::string(TypeName()) + "-GenRays"s;
    //
    queue.IssueWorkKernel<Kernel>
    (
        KernelName,
        DeviceWorkIssueParams{.workCount = rayCount},
        // Out
        dRayDiffsOut,
        dRaysOut,
        dImageCoordsOut,
        dSampleWeightsOut,
        // In
        dRayIndices,
        dRandomNums,
        // Constants
        dCamera,
        transKey,
        tg.SoA(),
        globalPixelIndex,
        regionCount
    );
}

template<RendererC R, CameraGroupC C, TransformGroupC T>
void RenderCameraWork<R, C, T>::GenRaysStochasticFilter(// Output
                                                        const Span<RayCone>& dRayDiffsOut,
                                                        const Span<RayGMem>& dRaysOut,
                                                        const Span<ImageCoordinate>& dImageCoordsOut,
                                                        const Span<Float>& dSampleWeightsOut,
                                                        // Input
                                                        const Span<const uint32_t>& dRayIndices,
                                                        const Span<const uint32_t>& dRandomNums,
                                                        // The actual pair to be used
                                                        Span<const Byte> dCamBuffer,
                                                        TransformKey transKey,
                                                        // Constants
                                                        uint64_t globalPixelIndex,
                                                        const Vector2ui regionCount,
                                                        FilterType filterType,
                                                        const GPUQueue& queue) const
{
    using Camera = typename C::Camera;
    assert(dRayIndices.size() * StochasticFilterSampleRayRNCount() == dRandomNums.size());
    assert(sizeof(Camera) <= dCamBuffer.size_bytes());
    assert(uintptr_t(dCamBuffer.data()) % alignof(Camera) == 0);
    const Camera* dCamera = reinterpret_cast<const Camera*>(dCamBuffer.data());
    Float filterRadius = filterType.radius;

    auto LaunchKernel = [&]<class Filter>(Filter&& filter)
    {
        uint32_t rayCount = static_cast<uint32_t>(dRayIndices.size());
        static constexpr auto Kernel = KCGenerateCamRaysStochastic<Camera, T, Filter>;
        static const std::string KernelName = MRAY_FORMAT("{}-{}-GenRays",
                                                          TypeName(),
                                                          FilterType::ToString(filterType.type));
        //
        queue.IssueWorkKernel<Kernel>
        (
            KernelName,
            DeviceWorkIssueParams{.workCount = rayCount},
            // Out
            dRayDiffsOut,
            dRaysOut,
            dImageCoordsOut,
            dSampleWeightsOut,
            // In
            dRayIndices,
            dRandomNums,
            // Constants
            dCamera,
            transKey,
            tg.SoA(),
            globalPixelIndex,
            regionCount,
            filter
        );
    };

    switch(filterType.type)
    {
        using enum FilterType::E;
        case BOX:       LaunchKernel(BoxFilter(filterRadius)); break;
        case TENT:      LaunchKernel(TentFilter(filterRadius)); break;
        case GAUSSIAN:  LaunchKernel(GaussianFilter(filterRadius)); break;
        case MITCHELL_NETRAVALI:
                        LaunchKernel(MitchellNetravaliFilter(filterRadius)); break;
        default: throw MRayError("Unkown filter type!");
    }
}

template<RendererC R, CameraGroupC C, TransformGroupC T>
std::string_view RenderCameraWork<R, C, T>::Name() const
{
    return TypeName();
}

template<RendererC R, CameraGroupC C, TransformGroupC T>
uint32_t RenderCameraWork<R, C, T>::SampleRayRNCount() const
{
    return C::SampleRayRNCount;
}

template<RendererC R, CameraGroupC C, TransformGroupC T>
uint32_t RenderCameraWork<R, C, T>::StochasticFilterSampleRayRNCount() const
{
    return 2u;
}