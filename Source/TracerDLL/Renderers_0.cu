#include "Tracer.h"
#include "RequestedTypes.h"

#include "Tracer/TexViewRenderer.h"

// ================= //
//     Renderers     //
// ================= //
using RendererTypes_0 = Tuple
<
    TexViewRenderer
>;

// Currently empty
using RendererWorkTypes_0 = Tuple
<
    EmptyRendererWorkTypes<TexViewRenderer>
>;

void Tracer::AddRendererGenerators_0(Map<std::string_view, RendererGenerator>& map,
                                     Map<std::string_view, RenderWorkPack>& workMap)
{
    using Args = Tuple<const RenderImagePtr&,
                       const RenderWorkPack&,
                       TracerView, const GPUSystem&>;

    Args* resolver0 = nullptr;
    RendererTypes_0* resolver1 = nullptr;
    GenerateMapping<RendererGenerator, RendererI>
    (
        map, resolver0, resolver1
    );

    // Work portion
    RendererWorkTypes_0* resolver2 = nullptr;
    AddRenderWorks(workMap, resolver2);
}