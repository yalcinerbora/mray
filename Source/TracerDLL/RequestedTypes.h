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
        //
        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupLambert, TransformGroupIdentity>,
        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupReflect, TransformGroupIdentity>,
        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupRefract, TransformGroupIdentity>,
        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupUnreal, TransformGroupIdentity>,

        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupLambert, TransformGroupSingle>,
        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupReflect, TransformGroupSingle>,
        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupRefract, TransformGroupSingle>,
        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupUnreal, TransformGroupSingle>,

        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupLambert, TransformGroupMulti>,
        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupReflect, TransformGroupMulti>,
        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupRefract, TransformGroupMulti>,
        RenderWorkT<PrimGroupSkinnedTriangle, MatGroupUnreal, TransformGroupMulti>,
        // Sphere
        RenderWorkT<PrimGroupSphere, MatGroupLambert, TransformGroupIdentity>,
        RenderWorkT<PrimGroupSphere, MatGroupReflect, TransformGroupIdentity>,
        RenderWorkT<PrimGroupSphere, MatGroupRefract, TransformGroupIdentity>,
        RenderWorkT<PrimGroupSphere, MatGroupUnreal, TransformGroupIdentity>,

        RenderWorkT<PrimGroupSphere, MatGroupLambert, TransformGroupSingle>,
        RenderWorkT<PrimGroupSphere, MatGroupReflect, TransformGroupSingle>,
        RenderWorkT<PrimGroupSphere, MatGroupRefract, TransformGroupSingle>,
        RenderWorkT<PrimGroupSphere, MatGroupUnreal, TransformGroupSingle>
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