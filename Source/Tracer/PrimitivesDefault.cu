#include "PrimitivesDefault.h"
#include "GenericGroup.hpp"

#include "Device/GPUMemory.h"
#include "Device/GPUAlgReduce.h"

#ifdef MRAY_GPU_BACKEND_CPU
    #include "Device/GPUSystem.hpp"
#endif

PrimGroupSphere::PrimGroupSphere(uint32_t primGroupId,
                                 const GPUSystem& sys)
    : GenericGroupPrimitive(primGroupId, sys,
                            DefaultSphereDetail::DeviceMemAllocationGranularity,
                            DefaultSphereDetail::DeviceMemReservationSize)
{}

void PrimGroupSphere::CommitReservations()
{
    GenericCommit(Tie(dCenters, dRadius), {0, 0});

    soa.centers = ToConstSpan(dCenters);
    soa.radius = ToConstSpan(dRadius);
}

PrimAttributeInfoList PrimGroupSphere::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum PrimitiveAttributeLogic;
    using enum AttributeOptionality;
    using enum AttributeIsArray;
    // Here we mark them as "IS_SCALAR", because primitive group is group of primitives
    // and not primitive batches
    static const PrimAttributeInfoList LogicList =
    {
        PrimAttributeInfo(POSITION, MRayDataTypeRT(MR_VECTOR_3),IS_SCALAR, MR_MANDATORY),
        PrimAttributeInfo(RADIUS,   MRayDataTypeRT(MR_FLOAT),   IS_SCALAR, MR_MANDATORY)
    };
    return LogicList;
}

void PrimGroupSphere::PushAttribute(PrimBatchKey batchKey,
                                    uint32_t attributeIndex,
                                    TransientData data,
                                    const GPUQueue& queue)
{
    auto PushData = [&]<class T>(const Span<T>& d)
    {
        GenericPushData(d, batchKey.FetchIndexPortion(),
                        attributeIndex,
                        std::move(data), queue);
    };

    switch(attributeIndex)
    {
        case CENTER_ATTRIB_INDEX: PushData(dCenters);   break;
        case RADIUS_ATTRIB_INDEX: PushData(dRadius);    break;
        default:
            throw MRayError("{:s}:{:d}: Unknown Attribute Index {:d}",
                            TypeName(), this->groupId, attributeIndex);
    }
}

void PrimGroupSphere::PushAttribute(PrimBatchKey batchKey,
                                    uint32_t attributeIndex,
                                    const Vector2ui& subRange,
                                    TransientData data,
                                    const GPUQueue& queue)
{
    auto PushData = [&]<class T>(const Span<T>& d)
    {
        GenericPushData(d, batchKey.FetchIndexPortion(),
                        attributeIndex, subRange,
                        std::move(data), queue);
    };

    switch(attributeIndex)
    {
        case CENTER_ATTRIB_INDEX: PushData(dCenters);   break;
        case RADIUS_ATTRIB_INDEX: PushData(dRadius);    break;
        default:
            throw MRayError("{:s}:{:d}: Unknown Attribute Index {:d}",
                            TypeName(), this->groupId, attributeIndex);
    }
}

void PrimGroupSphere::PushAttribute(PrimBatchKey idStart, PrimBatchKey idEnd,
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

    switch(attributeIndex)
    {
        case CENTER_ATTRIB_INDEX:   PushData(dCenters); break;
        case RADIUS_ATTRIB_INDEX:   PushData(dRadius);  break;
        default:
            throw MRayError("{:s}:{:d}: Unknown Attribute Index {:d}",
                            TypeName(), this->groupId, attributeIndex);
    }
}

void PrimGroupSphere::Finalize(const GPUQueue&)
{}

Vector2ui PrimGroupSphere::BatchRange(PrimBatchKey key) const
{
    auto range = FindRange(static_cast<CommonKey>(key))[CENTER_ATTRIB_INDEX];
    return Vector2ui(range);
}

size_t PrimGroupSphere::TotalPrimCount() const
{
    return this->TotalPrimCountImpl(0);
}

typename PrimGroupSphere::DataSoA PrimGroupSphere::SoA() const
{
    return soa;
}