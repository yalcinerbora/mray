#include "Tracer.h"

#include "Tracer/PrimitiveDefaultTriangle.h"
#include "Tracer/TransformsDefault.h"
#include "Tracer/LightsDefault.h"
#include "Tracer/MaterialsDefault.h"
#include "Tracer/CamerasDefault.h"
#include "Tracer/MediumsDefault.h"
#include "Tracer/AcceleratorWork.h"
#include "Tracer/AcceleratorLinear.h"
#include "Tracer/MetaLight.h"

template <class BaseAccel, class AccelGTypes, class AccelWorkTypes>
struct AccelTypePack
{
    using BaseType      = BaseAccel;
    using GroupTypes    = AccelGTypes;
    using WorkTypes     = AccelWorkTypes;
};

template <class PG, class MG, class TG>
struct RenderTriplet
{
    using PrimitiveGType    = PG;
    using MaterialGType     = MG;
    using TransformGType    = TG;
};

// ================= //
//     Primitives    //
// ================= //
using PrimGTypes = Tuple
<
    PrimGroupTriangle,
    PrimGroupSkinnedTriangle,
    PrimGroupEmpty
    // Sphere
>;

// ================= //
//     Transforms    //
// ================= //
using TransformGTypes = Tuple
<
    TransformGroupIdentity,
    TransformGroupSingle,
    TransformGroupMulti
>;

// ================= //
//      Lights       //
// ================= //
using LightGTypes = Tuple
<
    LightGroupNull,
    LightGroupPrim<PrimGroupTriangle>,
    LightGroupSkysphere<CoOctoCoordConverter>
>;

using MetaLightList = MetaLightArray
<
    // Transforms
    Variant
    <
        TransformContextIdentity,
        TransformContextSingle
    >,
    // Lights
    Variant
    <
        typename LightGroupSkysphere<CoOctoCoordConverter>:: template Light<TransformContextSingle>,
        typename LightGroupSkysphere<CoOctoCoordConverter>:: template Light<TransformContextIdentity>,
        typename LightGroupPrim<PrimGroupTriangle>:: template Light<TransformContextSingle>,
        typename LightGroupPrim<PrimGroupTriangle>:: template Light<TransformContextIdentity>
    >
>;

using MetaLight = typename MetaLightList::MetaLight;
using MetaLightView = typename MetaLightList::MetaLightView<MetaHit>;

// ================= //
//     Materials     //
// ================= //
using MatGTypes = Tuple
<
    MatGroupLambert,
    MatGroupReflect,
    MatGroupRefract,
    MatGroupUnreal
>;

// ================= //
//      Cameras      //
// ================= //
using CamGTypes = Tuple
<
    CameraGroupPinhole
>;

// ================= //
//      Mediums      //
// ================= //
using MedGTypes = Tuple
<
    MediumGroupVacuum,
    MediumGroupHomogeneous
>;

// ================= //
//    Accelerators   //
// ================= //
template <class Base, template<class> class Group>
using DefaultAccelTypePack = AccelTypePack
<
    Base,
    Tuple
    <
        Group<PrimGroupTriangle>,
        Group<PrimGroupSkinnedTriangle>
    >,
    Tuple
    <
        //
        AcceleratorWork<Group<PrimGroupTriangle>, TransformGroupIdentity>,
        AcceleratorWork<Group<PrimGroupTriangle>, TransformGroupSingle>,
        //
        AcceleratorWork<Group<PrimGroupSkinnedTriangle>, TransformGroupIdentity>,
        AcceleratorWork<Group<PrimGroupSkinnedTriangle>, TransformGroupSingle>,
        AcceleratorWork<Group<PrimGroupSkinnedTriangle>, TransformGroupMulti>
    >
>;

using DefaultLinearAccelTypePack = DefaultAccelTypePack<BaseAcceleratorLinear, AcceleratorGroupLinear>;
//using DefaultBVHAccelTypePack = DefaultAccelTypePack<BaseAcceleratorBVH, AcceleratorGroupBVH>;
//using DefaultDeviceAccelTypePack = DefaultAccelTypePack<BaseAcceleratorDevice, AcceleratorGroupDevice>;

//
//
//// Mat-Prim-Transform Triplets
//using RenderTypeTriplets = Tuple
//<
//    RenderTriplet<PrimGroupTriangle, MatGroupLambert, TransformGroupIdentity>,
//    RenderTriplet<PrimGroupTriangle, MatGroupLambert, TransformGroupSingle>,
//    RenderTriplet<PrimGroupSkinnedTriangle, MatGroupLambert, TransformGroupMulti>
//>;
//

TypeGeneratorPack Tracer::GLOBAL_TYPE_GEN = {};

void Tracer::AddPrimGenerators(Map<std::string_view, PrimGenerator>& map)
{
    using Args = Tuple<uint32_t, const GPUSystem&>;

    Args*           resolver0 = nullptr;
    PrimGTypes*     resolver1 = nullptr;
    GenerateMapping<PrimGenerator, GenericGroupPrimitiveT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddTransformGenerators(Map<std::string_view, TransGenerator>& map)
{
    using Args = Tuple<uint32_t, const GPUSystem&>;

    Args*               resolver0 = nullptr;
    TransformGTypes*    resolver1 = nullptr;
    GenerateMapping<TransGenerator, GenericGroupTransformT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddLightGenerators(Map<std::string_view, LightGenerator>& map)
{
    using Args = Tuple<uint32_t, const GPUSystem&, const TextureViewMap&,
                       GenericGroupPrimitiveT&>;

    Args*               resolver0 = nullptr;
    LightGTypes*        resolver1 = nullptr;
    GenerateMapping<LightGenerator, GenericGroupLightT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddCamGenerators(Map<std::string_view, CamGenerator>& map)
{
    using Args = Tuple<uint32_t, const GPUSystem&>;

    Args*       resolver0 = nullptr;
    CamGTypes*  resolver1 = nullptr;
    GenerateMapping<CamGenerator, GenericGroupCameraT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddMaterialGenerators(Map<std::string_view, MatGenerator>& map)
{
    using Args = Tuple<uint32_t, const GPUSystem&,
                       const TextureViewMap&>;

    Args*       resolver0 = nullptr;
    MatGTypes*  resolver1 = nullptr;
    GenerateMapping<MatGenerator, GenericGroupMaterialT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddMediumGenerators(Map<std::string_view, MedGenerator>& map)
{
    using Args = Tuple<uint32_t, const GPUSystem&,
                       const TextureViewMap&>;

    Args*       resolver0 = nullptr;
    MedGTypes*  resolver1 = nullptr;
    GenerateMapping<MedGenerator, GenericGroupMediumT>
    (
        map, resolver0, resolver1
    );
}

template<class AcceleratorPack>
static void AddAccelGeneratorsGeneric(Map<AcceleratorType, BaseAccelGenerator>& baseMap,
                                      Map<AcceleratorType, AccelGroupGenMap>& groupMap,
                                      Map<AcceleratorType, AccelWorkGenMap>& workMap,
                                      AcceleratorType t)
{
        // Base
    using LinAccel = typename AcceleratorPack::BaseType;

    baseMap.emplace(t, &GenerateType<BaseAcceleratorI, LinAccel,
                                     BS::thread_pool&, const GPUSystem&,
                                     const AccelGroupGenMap&,
                                     const AccelWorkGenMap&>);

    using GroupGenArgs = Tuple<uint32_t, BS::thread_pool&, const GPUSystem&,
                               const GenericGroupPrimitiveT&,
                               const AccelWorkGenMap&>;
    using AccelGTypes = typename AcceleratorPack::GroupTypes;
    GroupGenArgs*   groupResolver0 = nullptr;
    AccelGTypes*    groupResolver1 = nullptr;
    auto& genMap = groupMap.emplace(t, AccelGroupGenMap()).first->second;
    GenerateMapping<AccelGroupGenerator, AcceleratorGroupI>
    (
        genMap,
        groupResolver0,
        groupResolver1
    );

    // Works
    auto& workMapGlobal = workMap.emplace(t, AccelWorkGenMap()).first->second;
    using WorkGenArgs = Tuple<const AcceleratorGroupI&, const GenericGroupTransformT&>;
    using AccelWTypes = typename AcceleratorPack::WorkTypes;
    WorkGenArgs* workResolver0 = nullptr;
    AccelWTypes* workResolver1 = nullptr;
    GenerateMapping<AccelWorkGenerator, AcceleratorWorkI>
    (
        workMapGlobal,
        workResolver0,
        workResolver1
    );
}

void Tracer::AddAccelGenerators(Map<AcceleratorType, BaseAccelGenerator>& baseMap,
                                Map<AcceleratorType, AccelGroupGenMap>& groupMap,
                                Map<AcceleratorType, AccelWorkGenMap>& workMap)
{
    using enum AcceleratorType;
    AddAccelGeneratorsGeneric<DefaultLinearAccelTypePack>
    (
        baseMap,
        groupMap,
        workMap,
        SOFTWARE_NONE
    );
}