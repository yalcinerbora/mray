#include "Tracer.h"
#include "RequestedTypes.h"

#include "Tracer/TexViewRenderer.h"


// ================= //
//     Renderers     //
// ================= //
using RendererTypes_0 = std::tuple
<
    TexViewRenderer
>;

// Currently empty
using RendererWorkTypes_0 = std::tuple
<
    EmptyRendererWorkTypes<TexViewRenderer>
>;

void Tracer::AddRendererGenerators_0(Map<std::string_view, RendererGenerator>& map,
                                     Map<std::string_view, RenderWorkPack>& workMap)
{
    using Args = std::tuple<const RenderImagePtr&, TracerView,
                            ThreadPool&, const GPUSystem&,
                            const RenderWorkPack&>;

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