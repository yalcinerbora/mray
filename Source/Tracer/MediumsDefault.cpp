#include "MediumsDefault.h"
#include "Core/TypeNameGenerators.h"

std::string_view MediumGroupVacuum::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Vacuum"sv;
    return MediumTypeName<Name>;
}

MediumGroupVacuum::MediumGroupVacuum(uint32_t groupId,
                                     const GPUSystem& sys,
                                     const TextureViewMap& map)
    : GenericGroupMedium<MediumGroupVacuum>(groupId,
                                            sys, map)
{}

void MediumGroupVacuum::CommitReservations()
{
    isCommitted = true;
}

MediumAttributeInfoList MediumGroupVacuum::AttributeInfo() const
{
    return AttribInfoList{};
}

void MediumGroupVacuum::PushAttribute(MediumKey,
                                      uint32_t,
                                      TransientData,
                                      const GPUQueue&)
{}

void MediumGroupVacuum::PushAttribute(MediumKey,
                                      uint32_t,
                                      const Vector2ui&,
                                      TransientData,
                                      const GPUQueue&)
{}

void MediumGroupVacuum::PushAttribute(MediumKey, MediumKey,
                                      uint32_t,
                                      TransientData,
                                      const GPUQueue&)
{}

void MediumGroupVacuum::PushTexAttribute(MediumKey, MediumKey,
                                         uint32_t,
                                         TransientData,
                                         std::vector<Optional<TextureId>>,
                                         const GPUQueue&)
{}

void MediumGroupVacuum::PushTexAttribute(MediumKey, MediumKey,
                                         uint32_t,
                                         std::vector<Optional<TextureId>>,
                                         const GPUQueue&)
{}

void MediumGroupVacuum::PushTexAttribute(MediumKey, MediumKey,
                                         uint32_t,
                                         std::vector<TextureId>,
                                         const GPUQueue&)
{}

typename MediumGroupVacuum::DataSoA
MediumGroupVacuum::SoA() const
{
    return DataSoA{};
}