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
    LightGroupSkysphere<CoOctaCoordConverter>,
    LightGroupSkysphere<SphericalCoordConverter>
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
        typename LightGroupSkysphere<CoOctaCoordConverter>:: template Light<TransformContextSingle>,
        typename LightGroupSkysphere<CoOctaCoordConverter>:: template Light<TransformContextIdentity>,
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

// ================= //
//     Renderers     //
// ================= //
template <class Renderer,
          template<class, class, class> class RenderWorkT,
          template<class, class> class RenderLightWorkT,
          template<class, class> class RenderCameraWorkT>
using RendererWorkTypes = RenderWorkTypePack
<
    Renderer,
    // RenderWork
    Tuple
    <
        // Triangle
        RenderWorkT<PrimGroupTriangle, MatGroupLambert, TransformGroupIdentity>,
        RenderWorkT<PrimGroupTriangle, MatGroupReflect, TransformGroupIdentity>,
        RenderWorkT<PrimGroupTriangle, MatGroupRefract, TransformGroupIdentity>,
        RenderWorkT<PrimGroupTriangle, MatGroupUnreal, TransformGroupIdentity>,

        RenderWorkT<PrimGroupTriangle, MatGroupLambert, TransformGroupSingle>,
        RenderWorkT<PrimGroupTriangle, MatGroupReflect, TransformGroupSingle>,
        RenderWorkT<PrimGroupTriangle, MatGroupRefract, TransformGroupSingle>,
        RenderWorkT<PrimGroupTriangle, MatGroupUnreal, TransformGroupSingle>,

        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupLambert, TransformGroupMulti>,
        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupReflect, TransformGroupMulti>,
        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupRefract, TransformGroupMulti>,
        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupUnreal, TransformGroupMulti>

        // Sphere
        // ...
    >,
    // Lights
    Tuple
    <
        RenderLightWorkT<LightGroupNull, TransformGroupIdentity>,
        //
        RenderLightWorkT<LightGroupPrim<PrimGroupTriangle>, TransformGroupIdentity>,
        RenderLightWorkT<LightGroupPrim<PrimGroupTriangle>, TransformGroupSingle>,
        //
        RenderLightWorkT<LightGroupSkysphere<CoOctaCoordConverter>, TransformGroupIdentity>,
        RenderLightWorkT<LightGroupSkysphere<CoOctaCoordConverter>, TransformGroupSingle>,
        //
        RenderLightWorkT<LightGroupSkysphere<SphericalCoordConverter>, TransformGroupIdentity>,
        RenderLightWorkT<LightGroupSkysphere<SphericalCoordConverter>, TransformGroupSingle>
    >,
    // And finally Camera
    Tuple
    <
        RenderCameraWorkT<CameraGroupPinhole, TransformGroupIdentity>
    >
>;