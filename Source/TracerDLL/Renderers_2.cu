#include "Tracer.h"
#include "RequestedTypes.h"

#include "Tracer/PathTracerRenderer.h"

// ================= //
//     Renderers     //
// ================= //
using RendererTypes_2 = Tuple
<
    PathTracerRenderer<MetaLightList>
>;

using RendererWorkTypes_2 = Tuple
<
    RendererWorkTypes<PathTracerRenderer<MetaLightList>>
>;

// Since Path Tracer requires meta light list
// we do check the concept here
static_assert(RendererC<PathTracerRenderer<MetaLightList>>, "\"PathTracerRenderer\" does not "
              "satisfy renderer concept.");

void Tracer::AddRendererGenerators_2(Map<std::string_view, RendererGenerator>& map,
                                     Map<std::string_view, RenderWorkPack>& workMap)
{
    using Args = Tuple<const RenderImagePtr&, TracerView,
                       BS::thread_pool&, const GPUSystem&,
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