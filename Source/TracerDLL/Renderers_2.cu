#include "Tracer.h"
#include "RequestedTypes.h"
#include "PathTracerRenderer.h"

#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgGeneric.h"

// ================= //
//     Renderers     //
// ================= //
using RendererTypes_2 = PackedTypes
<
    PathTracerRenderer
>;

using RendererWorkTypes_2 = PackedTypes
<
    RendererWorkTypes<PathTracerRenderer, PathTracerRenderWork,
                      PathTracerRenderLightWork, PathTracerRenderCamWork>
>;

void Tracer::AddRendererGenerators_2(Map<std::string_view, RendererGenerator>& map,
                                     Map<std::string_view, RenderWorkPack>& workMap)
{
    using Args = PackedTypes<const RenderImagePtr&, TracerView,
                             ThreadPool&, const GPUSystem&,
                             const RenderWorkPack&>;

    Args* resolver0 = nullptr;
    RendererTypes_2* resolver1 = nullptr;
    GenerateMapping<RendererGenerator, RendererI>
    (
        map, resolver0, resolver1
    );

    // Work portion
    RendererWorkTypes_2* resolver2 = nullptr;
    AddRenderWorks(workMap, resolver2);
}