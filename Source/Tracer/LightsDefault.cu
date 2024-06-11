#include "LightsDefault.h"
#include "PrimitiveDefaultTriangle.h"

std::string_view LightGroupNull::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Null"sv;
    return LightTypeName<Name>;
    //return "(L)Null";
}

LightGroupNull::LightGroupNull(uint32_t groupId,
                               const GPUSystem& system,
                               const TextureViewMap& texMap,
                               const GenericGroupPrimitiveT& pg)
    : GenericGroupLight(groupId, system, texMap)
    , primGroup(static_cast<const PrimGroup&>(pg))
{}

void LightGroupNull::CommitReservations()
{
    isCommitted = true;
}

LightAttributeInfoList LightGroupNull::AttributeInfo() const
{
    return LightAttributeInfoList{};
}

void LightGroupNull::PushAttribute(LightKey,
                                   uint32_t,
                                   TransientData,
                                   const GPUQueue&)
{}

void LightGroupNull::PushAttribute(LightKey,
                                   uint32_t,
                                   const Vector2ui&,
                                   TransientData,
                                   const GPUQueue&)
{}

void LightGroupNull::PushAttribute(LightKey, LightKey,
                                   uint32_t,
                                   TransientData,
                                   const GPUQueue&)
{}

void LightGroupNull::PushTexAttribute(LightKey, LightKey,
                                      uint32_t,
                                      TransientData,
                                      std::vector<Optional<TextureId>>,
                                      const GPUQueue&)
{}

void LightGroupNull::PushTexAttribute(LightKey, LightKey,
                                      uint32_t,
                                      std::vector<Optional<TextureId>>,
                                      const GPUQueue&)

{}

void LightGroupNull::PushTexAttribute(LightKey, LightKey,
                                      uint32_t,
                                      std::vector<TextureId>,
                                      const GPUQueue&)
{}

typename LightGroupNull::DataSoA
LightGroupNull::SoA() const
{
    return EmptyType{};
}

const typename LightGroupNull::PrimGroup&
LightGroupNull::PrimitiveGroup() const
{
    return primGroup;
}

const GenericGroupPrimitiveT& LightGroupNull::GenericPrimGroup() const
{
    return primGroup;
}

bool LightGroupNull::IsPrimitiveBacked() const
{
    return false;
}

template class LightGroupSkysphere<CoOctoCoordConverter>;
template class LightGroupSkysphere<SphericalCoordConverter>;

template class LightGroupPrim<PrimGroupTriangle>;