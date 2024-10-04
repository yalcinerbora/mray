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

template <class Renderer,
          template<class, class, class, class> class RenderWorkT,
          template<class, class, class> class RenderLightWorkT,
          template<class, class, class> class RenderCameraWorkT>
using RendererWorkTypes = RenderWorkTypePack
<
    Renderer,
    // RenderWork
    Tuple
    <
        // Triangle
        RenderWorkT<Renderer, PrimGroupTriangle, MatGroupLambert, TransformGroupIdentity>,
        RenderWorkT<Renderer, PrimGroupTriangle, MatGroupReflect, TransformGroupIdentity>,
        RenderWorkT<Renderer, PrimGroupTriangle, MatGroupRefract, TransformGroupIdentity>,
        RenderWorkT<Renderer, PrimGroupTriangle, MatGroupUnreal, TransformGroupIdentity>,

        RenderWorkT<Renderer, PrimGroupTriangle, MatGroupLambert, TransformGroupSingle>,
        RenderWorkT<Renderer, PrimGroupTriangle, MatGroupReflect, TransformGroupSingle>,
        RenderWorkT<Renderer, PrimGroupTriangle, MatGroupRefract, TransformGroupSingle>,
        RenderWorkT<Renderer, PrimGroupTriangle, MatGroupUnreal, TransformGroupSingle>,
        //
        RenderWorkT<Renderer, PrimGroupSkinnedTriangle, MatGroupLambert, TransformGroupIdentity>,
        RenderWorkT<Renderer, PrimGroupSkinnedTriangle, MatGroupReflect, TransformGroupIdentity>,
        RenderWorkT<Renderer, PrimGroupSkinnedTriangle, MatGroupRefract, TransformGroupIdentity>,
        RenderWorkT<Renderer, PrimGroupSkinnedTriangle, MatGroupUnreal, TransformGroupIdentity>,

        RenderWorkT<Renderer, PrimGroupSkinnedTriangle, MatGroupLambert, TransformGroupSingle>,
        RenderWorkT<Renderer, PrimGroupSkinnedTriangle, MatGroupReflect, TransformGroupSingle>,
        RenderWorkT<Renderer, PrimGroupSkinnedTriangle, MatGroupRefract, TransformGroupSingle>,
        RenderWorkT<Renderer, PrimGroupSkinnedTriangle, MatGroupUnreal, TransformGroupSingle>,

        RenderWorkT<Renderer, PrimGroupSkinnedTriangle, MatGroupLambert, TransformGroupMulti>,
        RenderWorkT<Renderer, PrimGroupSkinnedTriangle, MatGroupReflect, TransformGroupMulti>,
        RenderWorkT<Renderer, PrimGroupSkinnedTriangle, MatGroupRefract, TransformGroupMulti>,
        RenderWorkT<Renderer, PrimGroupSkinnedTriangle, MatGroupUnreal, TransformGroupMulti>,
        // Sphere
        RenderWorkT<Renderer, PrimGroupSphere, MatGroupLambert, TransformGroupIdentity>,
        RenderWorkT<Renderer, PrimGroupSphere, MatGroupReflect, TransformGroupIdentity>,
        RenderWorkT<Renderer, PrimGroupSphere, MatGroupRefract, TransformGroupIdentity>,
        RenderWorkT<Renderer, PrimGroupSphere, MatGroupUnreal, TransformGroupIdentity>,

        RenderWorkT<Renderer, PrimGroupSphere, MatGroupLambert, TransformGroupSingle>,
        RenderWorkT<Renderer, PrimGroupSphere, MatGroupReflect, TransformGroupSingle>,
        RenderWorkT<Renderer, PrimGroupSphere, MatGroupRefract, TransformGroupSingle>,
        RenderWorkT<Renderer, PrimGroupSphere, MatGroupUnreal, TransformGroupSingle>
    >,
    // Lights
    Tuple
    <
        RenderLightWorkT<Renderer, LightGroupNull, TransformGroupIdentity>,
        //
        RenderLightWorkT<Renderer, LightGroupPrim<PrimGroupTriangle>, TransformGroupIdentity>,
        RenderLightWorkT<Renderer, LightGroupPrim<PrimGroupTriangle>, TransformGroupSingle>,
        //
        RenderLightWorkT<Renderer, LightGroupSkysphere<CoOctaCoordConverter>, TransformGroupIdentity>,
        RenderLightWorkT<Renderer, LightGroupSkysphere<CoOctaCoordConverter>, TransformGroupSingle>,
        //
        RenderLightWorkT<Renderer, LightGroupSkysphere<SphericalCoordConverter>, TransformGroupIdentity>,
        RenderLightWorkT<Renderer, LightGroupSkysphere<SphericalCoordConverter>, TransformGroupSingle>
    >,
    // And finally Camera
    Tuple
    <
        RenderCameraWorkT<Renderer, CameraGroupPinhole, TransformGroupIdentity>
    >
>;