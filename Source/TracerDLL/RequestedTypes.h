#pragma once

#include "Tracer/PrimitiveDefaultTriangle.h"
#include "Tracer/PrimitivesDefault.h"
#include "Tracer/TransformsDefault.h"
#include "Tracer/LightsDefault.h"
#include "Tracer/MaterialsDefault.h"
#include "Tracer/CamerasDefault.h"
#include "Tracer/MediumsDefault.h"
#include "Tracer/AcceleratorLinear.h"
#include "Tracer/AcceleratorLBVH.h"
#include "Tracer/MetaLight.h"
#include "Tracer/RenderWork.h"

#if defined(MRAY_GPU_BACKEND_CUDA) && defined(MRAY_ENABLE_HW_ACCELERATION)
    #include "Tracer/OptiX/AcceleratorOptiX.h"
#endif

// ================= //
//     Primitives    //
// ================= //
using PrimGTypes = Tuple
<
    PrimGroupTriangle,
    PrimGroupSkinnedTriangle,
    PrimGroupEmpty,
    // Sphere
    PrimGroupSphere
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

using MetaLightList = MetaLightArrayT
<
    Tuple<LightGroupSkysphere<CoOctaCoordConverter>, TransformGroupIdentity>,
    Tuple<LightGroupSkysphere<CoOctaCoordConverter>, TransformGroupSingle>,

    Tuple<LightGroupSkysphere<SphericalCoordConverter>, TransformGroupIdentity>,
    Tuple<LightGroupSkysphere<SphericalCoordConverter>, TransformGroupSingle>,

    Tuple<LightGroupPrim<PrimGroupTriangle>, TransformGroupIdentity>,
    Tuple<LightGroupPrim<PrimGroupTriangle>, TransformGroupSingle>,

    Tuple<LightGroupPrim<PrimGroupSkinnedTriangle>, TransformGroupMulti>,

    Tuple<LightGroupPrim<PrimGroupSphere>, TransformGroupIdentity>,
    Tuple<LightGroupPrim<PrimGroupSphere>, TransformGroupSingle>
>;

//using MetaLight = typename MetaLightList::MetaLight;
//using MetaLightView = typename MetaLightList::MetaLightView<MetaHit>;

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
        Group<PrimGroupSkinnedTriangle>,
        Group<PrimGroupSphere>
    >,
    Tuple
    <
        //
        AcceleratorWork<Group<PrimGroupTriangle>, TransformGroupIdentity>,
        AcceleratorWork<Group<PrimGroupTriangle>, TransformGroupSingle>,
        //
        AcceleratorWork<Group<PrimGroupSkinnedTriangle>, TransformGroupIdentity>,
        AcceleratorWork<Group<PrimGroupSkinnedTriangle>, TransformGroupSingle>,
        AcceleratorWork<Group<PrimGroupSkinnedTriangle>, TransformGroupMulti>,
        //
        AcceleratorWork<Group<PrimGroupSphere>, TransformGroupIdentity>,
        AcceleratorWork<Group<PrimGroupSphere>, TransformGroupSingle>
    >
>;

using DefaultLinearAccelTypePack = DefaultAccelTypePack<BaseAcceleratorLinear, AcceleratorGroupLinear>;
using DefaultBVHAccelTypePack = DefaultAccelTypePack<BaseAcceleratorLBVH, AcceleratorGroupLBVH>;
#if defined(MRAY_GPU_BACKEND_CUDA) && defined(MRAY_ENABLE_HW_ACCELERATION)
    using DefaultDeviceAccelTypePack = DefaultAccelTypePack<BaseAcceleratorOptiX, AcceleratorGroupOptiX>;
#endif

// ================= //
//     Renderers     //
// ================= //
template <class Renderer>
using EmptyRendererWorkTypes = RenderWorkTypePack
<
    Renderer, Tuple<>, Tuple<>, Tuple<>
>;

template <class Renderer>
using RendererWorkTypes = RenderWorkTypePack
<
    Renderer,
    // RenderWork
    Tuple
    <
        // Triangle
        RenderWork<Renderer, PrimGroupTriangle, MatGroupLambert, TransformGroupIdentity>,
        RenderWork<Renderer, PrimGroupTriangle, MatGroupReflect, TransformGroupIdentity>,
        RenderWork<Renderer, PrimGroupTriangle, MatGroupRefract, TransformGroupIdentity>,
        RenderWork<Renderer, PrimGroupTriangle, MatGroupUnreal, TransformGroupIdentity>,

        RenderWork<Renderer, PrimGroupTriangle, MatGroupLambert, TransformGroupSingle>,
        RenderWork<Renderer, PrimGroupTriangle, MatGroupReflect, TransformGroupSingle>,
        RenderWork<Renderer, PrimGroupTriangle, MatGroupRefract, TransformGroupSingle>,
        RenderWork<Renderer, PrimGroupTriangle, MatGroupUnreal, TransformGroupSingle>,
        //
        RenderWork<Renderer, PrimGroupSkinnedTriangle, MatGroupLambert, TransformGroupIdentity>,
        RenderWork<Renderer, PrimGroupSkinnedTriangle, MatGroupReflect, TransformGroupIdentity>,
        RenderWork<Renderer, PrimGroupSkinnedTriangle, MatGroupRefract, TransformGroupIdentity>,
        RenderWork<Renderer, PrimGroupSkinnedTriangle, MatGroupUnreal, TransformGroupIdentity>,

        RenderWork<Renderer, PrimGroupSkinnedTriangle, MatGroupLambert, TransformGroupSingle>,
        RenderWork<Renderer, PrimGroupSkinnedTriangle, MatGroupReflect, TransformGroupSingle>,
        RenderWork<Renderer, PrimGroupSkinnedTriangle, MatGroupRefract, TransformGroupSingle>,
        RenderWork<Renderer, PrimGroupSkinnedTriangle, MatGroupUnreal, TransformGroupSingle>,

        RenderWork<Renderer, PrimGroupSkinnedTriangle, MatGroupLambert, TransformGroupMulti>,
        RenderWork<Renderer, PrimGroupSkinnedTriangle, MatGroupReflect, TransformGroupMulti>,
        RenderWork<Renderer, PrimGroupSkinnedTriangle, MatGroupRefract, TransformGroupMulti>,
        RenderWork<Renderer, PrimGroupSkinnedTriangle, MatGroupUnreal, TransformGroupMulti>,
        // Sphere
        RenderWork<Renderer, PrimGroupSphere, MatGroupLambert, TransformGroupIdentity>,
        RenderWork<Renderer, PrimGroupSphere, MatGroupReflect, TransformGroupIdentity>,
        RenderWork<Renderer, PrimGroupSphere, MatGroupRefract, TransformGroupIdentity>,
        RenderWork<Renderer, PrimGroupSphere, MatGroupUnreal, TransformGroupIdentity>,

        RenderWork<Renderer, PrimGroupSphere, MatGroupLambert, TransformGroupSingle>,
        RenderWork<Renderer, PrimGroupSphere, MatGroupReflect, TransformGroupSingle>,
        RenderWork<Renderer, PrimGroupSphere, MatGroupRefract, TransformGroupSingle>,
        RenderWork<Renderer, PrimGroupSphere, MatGroupUnreal, TransformGroupSingle>
    >,
    // Lights
    Tuple
    <
        RenderLightWork<Renderer, LightGroupNull, TransformGroupIdentity>,
        //
        RenderLightWork<Renderer, LightGroupPrim<PrimGroupTriangle>, TransformGroupIdentity>,
        RenderLightWork<Renderer, LightGroupPrim<PrimGroupTriangle>, TransformGroupSingle>,
        //
        RenderLightWork<Renderer, LightGroupSkysphere<CoOctaCoordConverter>, TransformGroupIdentity>,
        RenderLightWork<Renderer, LightGroupSkysphere<CoOctaCoordConverter>, TransformGroupSingle>,
        //
        RenderLightWork<Renderer, LightGroupSkysphere<SphericalCoordConverter>, TransformGroupIdentity>,
        RenderLightWork<Renderer, LightGroupSkysphere<SphericalCoordConverter>, TransformGroupSingle>
    >,
    // And finally Camera
    Tuple
    <
        RenderCameraWork<Renderer, CameraGroupPinhole, TransformGroupIdentity>
    >
>;