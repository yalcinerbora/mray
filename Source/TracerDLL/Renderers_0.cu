#include "Tracer.h"
#include "RequestedTypes.h"

#include "Tracer/ImageRenderer.h"

// ================= //
//     Renderers     //
// ================= //
using RendererTypes_0 = Tuple
<
    ImageRenderer
>;

void Tracer::AddRendererGenerators_0(Map<std::string_view, RendererGenerator>& map)
{
    using Args = Tuple<RenderImagePtr, TracerView, const GPUSystem&>;

    Args* resolver0 = nullptr;
    RendererTypes_0* resolver1 = nullptr;
    GenerateMapping<RendererGenerator, RendererI>
    (
        map, resolver0, resolver1
    );
}