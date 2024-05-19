#include "Tracer.h"

#include "Tracer/PrimitiveDefaultTriangle.h"
#include "Tracer/TransformsDefault.h"
#include "Tracer/LightsDefault.h"
#include "Tracer/MaterialsDefault.h"
#include "Tracer/CamerasDefault.h"
#include "Tracer/MediumsDefault.h"

#include "Tracer/AcceleratorLinear.h"

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
    //MatGroupUnreal
>;

// ================= //
//      Cameras      //
// ================= //
using CamGTypes = Tuple
<
    //CamGroupPinhole
>;

// ================= //
//      Mediums      //
// ================= //
using MedGTypes = Tuple
<
    //MedGroupVacuum
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

void Tracer::AddPrimGenerators(std::map<std::string_view, PrimGenerator>& map)
{
    using Args = Tuple<uint32_t, GPUSystem&>;

    Args*           resolver0 = nullptr;
    PrimGTypes*     resolver1 = nullptr;
    GenerateMapping<PrimGenerator, GenericGroupPrimitiveT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddTransformGenerators(std::map<std::string_view, TransGenerator>& map)
{
    using Args = Tuple<uint32_t, GPUSystem&>;

    Args*               resolver0 = nullptr;
    TransformGTypes*    resolver1 = nullptr;
    GenerateMapping<TransGenerator, GenericGroupTransformT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddLightGenerators(std::map<std::string_view, LightGenerator>& map)
{
    using Args = Tuple<uint32_t, GPUSystem&, const TextureViewMap&,
                       GenericGroupPrimitiveT&>;

    Args*               resolver0 = nullptr;
    LightGTypes*        resolver1 = nullptr;
    GenerateMapping<LightGenerator, GenericGroupLightT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddCamGenerators(std::map<std::string_view, CamGenerator>& map)
{
    using Args = Tuple<uint32_t, GPUSystem&>;

    Args*       resolver0 = nullptr;
    CamGTypes*  resolver1 = nullptr;
    GenerateMapping<CamGenerator, GenericGroupCameraT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddMaterialGenerators(std::map<std::string_view, MatGenerator>& map)
{
    using Args = Tuple<uint32_t, GPUSystem&,
                       const TextureViewMap&>;

    Args*       resolver0 = nullptr;
    MatGTypes*  resolver1 = nullptr;
    GenerateMapping<MatGenerator, GenericGroupMaterialT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddMediumGenerators(std::map<std::string_view, MedGenerator>& map)
{
    using Args = Tuple<uint32_t, GPUSystem&,
                       const TextureViewMap&>;

    Args*       resolver0 = nullptr;
    MedGTypes*  resolver1 = nullptr;
    GenerateMapping<MedGenerator, GenericGroupMediumT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddAccelGenerators(std::map<AcceleratorType, BaseAccelGenerator>& baseMap,
                                std::map<AcceleratorType, AccelGroupGenMap>& groupMap,
                                std::map<AcceleratorType, AccelWorkGenMap>& workMap)
{
    // Base
    using LinAccel = typename DefaultLinearAccelTypePack::BaseType;
    using enum AcceleratorType;
    baseMap.emplace(SOFTWARE_NONE, &GenerateType<BaseAcceleratorI, LinAccel,
                                                 BS::thread_pool&, GPUSystem&,
                                                 const AccelGroupGenMap&,
                                                 const AccelWorkGenMap&>);

    using GroupGenArgs = Tuple<uint32_t, BS::thread_pool&, GPUSystem&,
                               const GenericGroupPrimitiveT&,
                               const AccelWorkGenMap&>;
    using AccelGTypes = typename DefaultLinearAccelTypePack::GroupTypes;
    GroupGenArgs*   groupResolver0 = nullptr;
    AccelGTypes*    groupResolver1 = nullptr;
    auto& genMap = groupMap.emplace(SOFTWARE_NONE, AccelGroupGenMap()).first->second;
    GenerateMapping<AccelGroupGenerator, AcceleratorGroupI>
    (
        genMap,
        groupResolver0,
        groupResolver1
    );

    // Now the hard part
    auto& workMapGlobal = workMap.emplace(SOFTWARE_NONE, AccelWorkGenMap()).first->second;
    using WorkGenArgs = Tuple<AcceleratorGroupI&, GenericGroupTransformT&>;
    using AccelWTypes = typename DefaultLinearAccelTypePack::WorkTypes;
    WorkGenArgs* workResolver0 = nullptr;
    AccelWTypes* workResolver1 = nullptr;
    //GenerateMapping<AccelWorkGenerator, AcceleratorWorkI>
    //(
    //    workMapGlobal,
    //    workResolver0,
    //    workResolver1
    //);

}