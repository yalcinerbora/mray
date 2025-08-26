#include "MediumsDefault.h"
#include "GenericGroup.hpp"

#include "Core/TypeNameGenerators.h"

#ifdef MRAY_GPU_BACKEND_CPU
    #include "Device/GPUSystem.hpp"
#endif

std::string_view MediumGroupVacuum::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Vacuum"sv;
    return MediumTypeName<Name>;
}

MediumGroupVacuum::MediumGroupVacuum(uint32_t groupId,
                                     const GPUSystem& sys,
                                     const TextureViewMap& map,
                                     const TextureMap&)
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


std::string_view MediumGroupHomogeneous::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Homogeneous"sv;
    return MediumTypeName<Name>;
}

MediumGroupHomogeneous::MediumGroupHomogeneous(uint32_t groupId,
                                               const GPUSystem& sys,
                                               const TextureViewMap& map,
                                               const TextureMap&)
    : GenericGroupMedium<MediumGroupHomogeneous>(groupId, sys, map)
{}

void MediumGroupHomogeneous::CommitReservations()
{
    GenericCommit(Tie(dSigmaA, dSigmaS, dEmission, dPhaseVal),
                  {0, 0, 0, 0});

    soa = DataSoA(ToConstSpan(dSigmaA), ToConstSpan(dSigmaS),
                  ToConstSpan(dEmission), ToConstSpan(dPhaseVal));
}

MediumAttributeInfoList MediumGroupHomogeneous::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeOptionality;
    using enum AttributeTexturable;
    using enum AttributeIsArray;
    using enum AttributeIsColor;
    static const MatAttributeInfoList LogicList =
    {
        MatAttributeInfo("sigmaA", MRayDataTypeRT(MR_VECTOR_3), IS_SCALAR,
                         MR_MANDATORY, MR_CONSTANT_ONLY, IS_COLOR),
        MatAttributeInfo("sigmaS", MRayDataTypeRT(MR_VECTOR_3), IS_SCALAR,
                         MR_MANDATORY, MR_CONSTANT_ONLY, IS_COLOR),
        MatAttributeInfo("emission", MRayDataTypeRT(MR_VECTOR_3), IS_SCALAR,
                         MR_MANDATORY, MR_CONSTANT_ONLY, IS_COLOR),
        MatAttributeInfo("hgPhase", MRayDataTypeRT(MR_FLOAT), IS_SCALAR,
                         MR_MANDATORY, MR_CONSTANT_ONLY, IS_PURE_DATA)
    };
    return LogicList;
}

void MediumGroupHomogeneous::PushAttribute(MediumKey id,
                                           uint32_t attributeIndex,
                                           TransientData data,
                                           const GPUQueue& queue)
{
    auto PushData = [&]<class T>(const Span<T>& d)
    {
        GenericPushData(d, id.FetchIndexPortion(),
                        attributeIndex,
                        std::move(data), queue);
    };

    using enum DataSoA::I;
    switch(attributeIndex)
    {
        case SIGMA_A:   PushData(dSigmaA);      break;
        case SIGMA_S:   PushData(dSigmaS);      break;
        case EMISSION:  PushData(dEmission);    break;
        case HG_PHASE:  PushData(dPhaseVal);    break;
        default:
            throw MRayError("{:s}:{:d}: Unknown Attribute Index {:d}",
                            TypeName(), this->groupId, attributeIndex);
    }
}

void MediumGroupHomogeneous::PushAttribute(MediumKey id,
                                           uint32_t attributeIndex,
                                           const Vector2ui& subRange,
                                           TransientData data,
                                           const GPUQueue& queue)
{
    auto PushData = [&]<class T>(const Span<T>& d)
    {
        GenericPushData(d, id.FetchIndexPortion(),
                        attributeIndex, subRange,
                        std::move(data), queue);
    };

    using enum DataSoA::I;
    switch(attributeIndex)
    {
        case SIGMA_A:   PushData(dSigmaA);      break;
        case SIGMA_S:   PushData(dSigmaS);      break;
        case EMISSION:  PushData(dEmission);    break;
        case HG_PHASE:  PushData(dPhaseVal);    break;
        default:
            throw MRayError("{:s}:{:d}: Unknown Attribute Index {:d}",
                            TypeName(), this->groupId, attributeIndex);
    }
}

void MediumGroupHomogeneous::PushAttribute(MediumKey idStart, MediumKey idEnd,
                                           uint32_t attributeIndex,
                                           TransientData data,
                                           const GPUQueue& queue)
{
    auto PushData = [&]<class T>(const Span<T>& d)
    {
        Vector<2, IdInt> idRange(idStart.FetchIndexPortion(),
                                 idEnd.FetchIndexPortion());
        GenericPushData(d, idRange, attributeIndex,
                        std::move(data), queue);
    };

    using enum DataSoA::I;
    switch(attributeIndex)
    {
        case SIGMA_A:   PushData(dSigmaA);      break;
        case SIGMA_S:   PushData(dSigmaS);      break;
        case EMISSION:  PushData(dEmission);    break;
        case HG_PHASE:  PushData(dPhaseVal);    break;
        default:
            throw MRayError("{:s}:{:d}: Unknown Attribute Index {:d}",
                            TypeName(), this->groupId, attributeIndex);
    }
}

void MediumGroupHomogeneous::PushTexAttribute(MediumKey, MediumKey,
                                              uint32_t,
                                              TransientData,
                                              std::vector<Optional<TextureId>>,
                                              const GPUQueue&)
{
    throw MRayError("{:s} do not have any texture related attributes!",
                    TypeName());
}

void MediumGroupHomogeneous::PushTexAttribute(MediumKey, MediumKey,
                                              uint32_t,
                                              std::vector<Optional<TextureId>>,
                                              const GPUQueue&)
{
    throw MRayError("{:s} do not have any texture related attributes!",
                    TypeName());
}

void MediumGroupHomogeneous::PushTexAttribute(MediumKey, MediumKey,
                                              uint32_t,
                                              std::vector<TextureId>,
                                              const GPUQueue&)
{
    throw MRayError("{:s} do not have any texture related attributes!",
                    TypeName());
}

typename MediumGroupHomogeneous::DataSoA
MediumGroupHomogeneous::SoA() const
{
    return soa;
}