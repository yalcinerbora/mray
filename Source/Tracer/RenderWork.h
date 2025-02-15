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
static void KCRenderWork(MRAY_GRID_CONSTANT const RenderWorkParamsR<R, I, PG, MG, TG> params);

template<RendererC R, uint32_t I, LightGroupC LG, TransformGroupC TG,
         auto WorkFunction,
         auto GenerateTransformContext = MRAY_PRIM_TGEN_FUNCTION(typename LG::PrimGroup, TG)>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static void KCRenderLightWork(MRAY_GRID_CONSTANT const RenderLightWorkParamsR<R, I, LG, TG> params);

template<RendererC R, uint32_t I, PrimitiveGroupC PG,
         CameraGroupC CG, TransformGroupC TG,
         auto WorkFunction>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static void KCRenderCameraWork(MRAY_GRID_CONSTANT const RenderCameraWorkParamsR<R, I, CG, TG> params);

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
        queue.IssueSaturatingKernel<Kernel>
        (
            KernelName,
            KernelIssueParams{.workCount = rayCount},
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
        queue.IssueSaturatingKernel<Kernel>
        (
            TypeName(),
            KernelIssueParams{.workCount = rayCount},
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
    queue.IssueExactKernel<Kernel>
    (
        KernelName,
        KernelExactIssueParams{.gridSize = 1, .blockSize = 1},
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
    queue.IssueSaturatingKernel<Kernel>
    (
        KernelName,
        KernelIssueParams{.workCount = rayCount},
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
        queue.IssueSaturatingKernel<Kernel>
        (
            KernelName,
            KernelIssueParams{.workCount = rayCount},
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

template<RendererC R, uint32_t I, PrimitiveGroupC PG,
         MaterialGroupC MG, TransformGroupC TG,
         auto WorkFunction, auto GenerateTransformContext>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static void KCRenderWork(MRAY_GRID_CONSTANT const RenderWorkParamsR<R, I, PG, MG, TG> params)
{
    using SpectrumConv  = typename R::SpectrumConverterContext;
    // Define the types
    // First, this kernel uses a transform context.
    // Thil will be used to transform the primitive.
    // We may not be able act on the primitive without the primitive's data
    // (For example skinned meshes, we need weights and transform indices)
    // "TransformContext" abstraction handles these
    using TransContext = typename PrimTransformContextType<PG, TG>::Result;
    // Similarly Renderer may not be RGB, but Material is
    // So it sends the spectrum converter contex, which is either identity
    // or a function that is constructed via wavelengths etc.
    using Material = typename MG:: template Material<SpectrumConv>;
    // Primitive is straightforward but we get a type with the proper
    // transform context
    using Primitive = typename PG:: template Primitive<TransContext>;
    // And finally, the the definition of the surface that is dictated
    // by the material group
    using Surface = typename Material::Surface;
    // We need to compile-time check that this primitive supports such a surface
    static_assert(PrimitiveWithSurfaceC<Primitive, PG, Surface>,
                  "This primitive does not support the surface "
                  "required by a material");
    // The hit type of this specific primitive
    using Hit = typename Primitive::Hit;

    // Now finally we can start the runtime stuff
    uint32_t rayCount = static_cast<uint32_t>(params.in.dRayIndices.size());
    KernelCallParams kp;
    for(uint32_t globalId = kp.GlobalId();
        globalId < rayCount; globalId += kp.TotalSize())
    {
        RNGDispenser rng(params.in.dRandomNumbers, globalId, rayCount);
        // Here is the potential drawback of sorting based
        // partitioning, all of the parameters are loaded
        // via scattered reads.
        //
        // Queue based partitioning does scattered writes
        // so we transferred the non-contigious memory I/O to
        // here
        // Ray
        RayIndex rIndex = params.in.dRayIndices[globalId];
        auto [ray, tMM] = RayFromGMem(params.in.dRays, rIndex);
        // Keys
        HitKeyPack keys = params.in.dKeys[rIndex];
        // Hit (TODO: What about single parameters?)
        Hit hit = params.in.dHits[rIndex].template AsVector<Hit::Dims>();
        // Advance the differential to the hit location
        RayCone rayCone = params.in.dRayCones[rIndex].Advance(tMM[1]);

        // Get instantiation of converter
        // TODO: Add spectrum related stuff, this should not be
        // default constructed
        typename SpectrumConv::Converter specConverter;
        // Create transform context
        TransContext tContext = GenerateTransformContext(params.transSoA,
                                                         params.primSoA,
                                                         keys.transKey,
                                                         keys.primKey);
        // Construct Primitive
        auto primitive = Primitive(tContext,
                                   params.primSoA,
                                   keys.primKey);
        // Generate the surface
        Surface surface;
        RayConeSurface rayConeSurface;
        primitive.GenerateSurface(surface, rayConeSurface,
                                  hit, ray, rayCone);
        // Generate the material
        auto material = Material(specConverter, params.matSoA,
                                 std::bit_cast<MaterialKey>(keys.lightOrMatKey));
        // Call the function
        WorkFunction(primitive, material, surface, rayConeSurface,
                     tContext, rng, params, rIndex);
        // All Done!
    }
}

template<RendererC R, uint32_t I, LightGroupC LG, TransformGroupC TG,
         auto WorkFunction, auto GenerateTransformContext>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static void KCRenderLightWork(MRAY_GRID_CONSTANT const RenderLightWorkParamsR<R, I, LG, TG> params)
{
    using SpectrumConv = typename R::SpectrumConverterContext;
    //
    using PG = typename LG::PrimGroup;
    // Define the types
    // First, this kernel uses a transform context.
    // Thil will be used to transform the primitive.
    // We may not be able act on the primitive without the primitive's data
    // (For example skinned meshes, we need weights and transform indices)
    // "TransformContext" abstraction handles these
    using TransContext = typename PrimTransformContextType<PG, TG>::Result;
    // Primitive is straightforward but we get a type with the proper
    // transform context
    using Primitive = typename PG:: template Primitive<TransContext>;
    // Light
    using Light = typename LG:: template Light<TransContext, SpectrumConv>;
    // Now finally we can start the runtime stuff
    uint32_t rayCount = static_cast<uint32_t>(params.in.dRayIndices.size());
    KernelCallParams kp;
    for(uint32_t globalId = kp.GlobalId();
        globalId < rayCount; globalId += kp.TotalSize())
    {
        RNGDispenser rng(params.in.dRandomNumbers, globalId, rayCount);
        RayIndex rIndex = params.in.dRayIndices[globalId];
        // Keys
        HitKeyPack keys = params.in.dKeys[rIndex];
        LightKey lKey = LightKey::CombinedKey(keys.lightOrMatKey.FetchBatchPortion(),
                                              keys.lightOrMatKey.FetchIndexPortion());
        // Create transform context
        TransContext tContext = GenerateTransformContext(params.transSoA,
                                                         params.primSoA,
                                                         keys.transKey,
                                                         keys.primKey);

        // Get instantiation of converter
        // TODO: Add spectrum related stuff, this should not be
        // default constructed
        typename SpectrumConv::Converter specConverter;
        // Construct Primitive
        auto primitive = Primitive(tContext,
                                   params.primSoA,
                                   keys.primKey);
        // Construct light
        auto light = Light(specConverter, primitive, params.lightSoA, lKey);

        // Call the function
        WorkFunction(light, rng, params, rIndex);
        // All Done!
    }
}

template<RendererC R, uint32_t I, CameraGroupC CG, TransformGroupC TG,
         auto WorkFunction>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
static void KCRenderCameraWork(MRAY_GRID_CONSTANT const RenderCameraWorkParamsR<R, I, CG, TG> params)
{

}