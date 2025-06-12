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
using PrimGTypes = PackedTypes
<
{}
>;

// ================= //
//     Materials     //
// ================= //
using MatGTypes = PackedTypes
<
{}
>;

// ================= //
//     Transforms    //
// ================= //
using TransformGTypes = PackedTypes
<
{}
>;

// ================= //
//      Cameras      //
// ================= //
using CamGTypes = PackedTypes
<
{}
>;

// ================= //
//      Mediums      //
// ================= //
using MedGTypes = PackedTypes
<
{}
>;

// ================= //
//      Lights       //
// ================= //
using LightGTypes = PackedTypes
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
    PackedTypes
    <
    {}
    >,
    PackedTypes
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
    Renderer, PackedTypes<>, PackedTypes<>, PackedTypes<>
>;

template <class Renderer,
          template<class, class, class, class> class RenderWorkT,
          template<class, class, class> class RenderLightWorkT,
          template<class, class, class> class RenderCameraWorkT>
using RendererWorkTypes = RenderWorkTypePack
<
    Renderer,
    // RenderWork
    PackedTypes
    <
    {}
    >,
    // Lights
    PackedTypes
    <
    {}
    >,
    // And finally Camera
    PackedTypes
    <
    {}
    >
>;

using RendererTypeList = PackedTypes
<
{}
>;

// Currently empty
using RendererWorkTypesList = PackedTypes
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
    // some residual code catched by msvc
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

// Accelerator Work Instantiations
{}
// Camera Raygen Instantiations
{}
)"sv;
