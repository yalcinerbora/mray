#include "Tracer.h"
#include "RequestedTypes.h"

#include "Tracer/SurfaceRenderer.h"

// ================= //
//     Renderers     //
// ================= //
using RendererTypes_1 = Tuple
<
    SurfaceRenderer
>;

using RendererWorkTypes_1 = Tuple
<
     RendererWorkTypes<SurfaceRenderer, RenderWork,
                       RenderLightWork, RenderCameraWork>
>;

void Tracer::AddRendererGenerators_1(Map<std::string_view, RendererGenerator>& map,
                                     Map<std::string_view, RenderWorkPack>& workMap)
{
    using Args = Tuple<const RenderImagePtr&, TracerView,
                       BS::thread_pool&, const GPUSystem&,
                       const RenderWorkPack&>;

    Args* resolver0 = nullptr;
    RendererTypes_1* resolver1 = nullptr;
    GenerateMapping<RendererGenerator, RendererI>
    (
        map, resolver0, resolver1
    );

    // Work portion
    RendererWorkTypes_1* resolver2 = nullptr;
    AddRenderWorks(workMap, resolver2);
}