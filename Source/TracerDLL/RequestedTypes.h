#pragma once

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

// Mat-Prim-Transform Triplets
// We will enable this when renderers need the types
//
//using RenderTypeTriplets = Tuple
//<
//    RenderTriplet<PrimGroupTriangle, MatGroupLambert, TransformGroupIdentity>,
//    RenderTriplet<PrimGroupTriangle, MatGroupLambert, TransformGroupSingle>,
//    RenderTriplet<PrimGroupTriangle, MatGroupLambert, TransformGroupSingle>,
//    RenderTriplet<PrimGroupSkinnedTriangle, MatGroupLambert, TransformGroupMulti>
//>;
