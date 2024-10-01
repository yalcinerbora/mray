#include "PathTracerRenderer.h"

std::string_view PathTracerRenderer::TypeName()
{
    using namespace std::string_view_literals;
    using namespace TypeNameGen::CompTime;
    static constexpr auto Name = "PathTracer"sv;
    return RendererTypeName<Name>;
}

size_t PathTracerRenderer::GPUMemoryUsage() const
{
    return (rayPartitioner.UsedGPUMemory() +
            rnGenerator->UsedGPUMemory() +
            redererGlobalMem.Size());
}