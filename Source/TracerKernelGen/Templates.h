#pragma once

#include <string_view>

using std::string_view_literals::operator""sv;

static constexpr std::string_view TYPEGEN_HEADER_FILE_TEMPLATE_FMT =
R"(#pragma once
// =========================== //
// GENERATED FILE DO NOT EDIT! //
// =========================== //

// Includes
{}
// Mandatory Headers
#include "Tracer/MetaLight.h"
//#include "Tracer/RenderWork.h"
#include "Tracer/AcceleratorWork.h"
// Guarded Headers
#if defined({}) && defined(MRAY_ENABLE_HW_ACCELERATION)
    #include {}
#endif

// ================= //
//     Primitives    //
// ================= //
using PrimGTypes = TypePack
<
{}
>;

// ================= //
//     Materials     //
// ================= //
using MatGTypes = TypePack
<
{}
>;

// ================= //
//     Transforms    //
// ================= //
using TransformGTypes = TypePack
<
{}
>;

// ================= //
//      Cameras      //
// ================= //
using CamGTypes = TypePack
<
{}
>;

// ================= //
//      Mediums      //
// ================= //
using MedGTypes = TypePack
<
{}
>;

// ================= //
//      Lights       //
// ================= //
using LightGTypes = TypePack
<
{}
>;

using MetaLightList = MetaLightArrayT
<
{}
>;

// ================= //
//    Accelerators   //
// ================= //
template <class Base, template<class> class Group>
using DefaultAccelTypePack = AccelTypePack
<
    Base,
    TypePack
    <
    {}
    >,
    TypePack
    <
    {}
    >
>;

using DefaultLinearAccelTypePack = DefaultAccelTypePack<{}, {}>;
using DefaultBVHAccelTypePack = DefaultAccelTypePack<{}, {}>;
#if defined({}) && defined(MRAY_ENABLE_HW_ACCELERATION)
    using DefaultDeviceAccelTypePack = DefaultAccelTypePack<{}, {}>;
#endif
)"sv;


static constexpr std::string_view TYPEGEN_RENDER_HEADER_FILE_TEMPLATE_FMT =
R"(#pragma once
// =========================== //
// GENERATED FILE DO NOT EDIT! //
// =========================== //

// Includes
{}
// Mandatory Headers
#include "Tracer/RenderWork.h"
#include "_GEN_RequestedTypes.h"

// ================= //
//     Renderers     //
// ================= //
template <class Renderer>
using EmptyRendererWorkTypes = RenderWorkTypePack
<
    Renderer, TypePack<>, TypePack<>, TypePack<>, TypePack<>
>;

template <class Renderer,
          template<class, class, class, class> class RenderWorkT,
          template<class, class, class> class RenderLightWorkT,
          template<class, class, class> class RenderCameraWorkT,
          template<class, class, class> class RenderMediumWorkT>
using RendererWorkTypes = RenderWorkTypePack
<
    Renderer,
    // RenderWork
    TypePack
    <
    {}
    >,
    // Lights
    TypePack
    <
    {}
    >,
    // Camera
    TypePack
    <
    {}
    >,
    // And finally, Media
    TypePack
    <
    {}
    >
>;

using RendererTypeList = TypePack
<
{}
>;

// Currently empty
using RendererWorkTypesList = TypePack
<
{}
>;
)"sv;

static constexpr auto KERNEL_FILE_TEMPLATE =
R"(// =========================== //
// GENERATED FILE DO NOT EDIT! //
// =========================== //
#ifdef MRAY_WINDOWS
    // After nvcc passes through
    // some residual code caught by msvc
    // and "unreachable code" is generated
    // TODO: Investigate
    #pragma warning( disable : 4702)
#endif

// Definitions
#include "Tracer/RayGenKernels.h"
#include "Tracer/RenderWork.h"
#include "Tracer/AcceleratorWork.h"

// Implementations
#include "Tracer/RayGenKernels.kt.h"
#include "Tracer/RenderWork.kt.h"
#include "Tracer/AcceleratorWork.kt.h"
#include "Tracer/TextureView.hpp"

// Types
#include "InstantiationMacros.h"

#include "RequestedTypes.h"
#include "RequestedRenderers.h"

// Kernel Work Instantiations
{}
// Kernel Light Work Instantiations
{}
// Kernel Camera Work Instantiations
{}
// Kernel Media Work Instantiations
{}
// Accelerator Work Instantiations
{}
// Camera Raygen Instantiations
{}
)"sv;
