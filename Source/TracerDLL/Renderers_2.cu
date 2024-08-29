#include "Tracer.h"

void Tracer::AddRendererGenerators_2(Map<std::string_view, RendererGenerator>&,
                                     Map<std::string_view, RenderWorkPack>&)
{
    using Args = Tuple<const RenderImagePtr&, TracerView,
                       BS::thread_pool&, const GPUSystem&,
                       const RenderWorkPack&>;
}