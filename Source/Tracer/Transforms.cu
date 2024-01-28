#include "Transforms.h"
#include "Device/GPUSystem.hpp"

std::string_view TransformGroupSingle::TypeName()
{
    using namespace std::literals;
    static std::string_view name = "(T)Identity"sv;
    return name;
}

typename TransformGroupIdentity::DataSoA TransformGroupIdentity::SoA() const
{
    return EmptyType{};
}

TransformGroupSingle::TransformGroupSingle(uint32_t groupId,
                     const GPUSystem& s)
    : BaseType(groupId, s)
{}

void TransformGroupSingle::CommitReservations()
{
    auto [t, it] = GenericCommit<Matrix4x4, Matrix4x4>({true, true});
    transforms = t;
    invTransforms = it;

    soa.transforms = ToConstSpan(transforms);
    soa.invTransforms = ToConstSpan(invTransforms);
}

void TransformGroupSingle::PushAttribute(TransformKey id , uint32_t attributeIndex,
                                         MRayInput data)
{
    switch(attributeIndex)
    {
        case 0: GenericPushData(id, transforms, std::move(data), true); break;
        case 1: GenericPushData(id, invTransforms, std::move(data), true); break;
        default: throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
                                             TypeName(), attributeIndex));
    }
}

void TransformGroupSingle::PushAttribute(TransformKey id,
                                         const Vector2ui& subRange,
                                         uint32_t attributeIndex,
                                         MRayInput data)
{
    switch(attributeIndex)
    {
        case 0: GenericPushData(id, subRange, transforms, std::move(data), true); break;
        case 1: GenericPushData(id, subRange, invTransforms, std::move(data), true); break;
        default: throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
                                             TypeName(), attributeIndex));
    }
}

void TransformGroupSingle::PushAttribute(const Vector<2, TransformKey::Type>& idRange,
                                         uint32_t attributeIndex, MRayInput data)
{
    switch(attributeIndex)
    {
        case 0: GenericPushData(idRange, transforms, std::move(data), true, true); break;
        case 1: GenericPushData(idRange, invTransforms, std::move(data), true, true); break;
        default: throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
                                             TypeName(), attributeIndex));
    }
}

TransformGroupSingle::AttribInfoList TransformGroupSingle::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeOptionality;
    static const std::array<TransAttributeInfo, 2> LogicList =
    {
        TransAttributeInfo("Transform", MR_MANDATORY, MRayDataType<MR_MATRIX_4x4>()),
        TransAttributeInfo("InvTransform", MR_OPTIONAL, MRayDataType<MR_MATRIX_4x4>())
    };
    return std::vector(LogicList.cbegin(), LogicList.cend());
}

typename TransformGroupSingle::DataSoA TransformGroupSingle::SoA() const
{
    return soa;
}

std::string_view TransformGroupMulti::TypeName()
{
    using namespace std::literals;
    static std::string_view name = "(T)Multi"sv;
    return name;
}

TransformGroupMulti::TransformGroupMulti(uint32_t groupId,
                                         const GPUSystem& s)
    : BaseType(groupId, s)
{}

void TransformGroupMulti::CommitReservations()
{
    Span<Span<const Matrix4x4>> batchedTransforms;
    Span<Span<const Matrix4x4>> batchedInvTransforms;

    //auto [t, it, bt22, bit22] =
    auto [t, it, batchT, batchInvT] =
        GenericCommit<Matrix4x4, Matrix4x4,
                      Span<const Matrix4x4>, Span<const Matrix4x4>>({true, true, true, true});
    transforms = t;
    invTransforms = it;
    batchedTransforms = batchT;
    batchedTransforms = batchInvT;

    // TODO: Improve this?
    // Locally creating buffers ...
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
    DeviceLocalMemory mem(gpuSystem.BestDevice());
    Span<Vector<2, IdInteger>> dFlattenedRanges;
    MemAlloc::AllocateMultiData(std::tie(dFlattenedRanges),
                                mem, {itemRanges.size()});

    std::vector<Vector<2, IdInteger>> hFlattenedRanges;
    hFlattenedRanges.reserve(itemRanges.size());

    for(const auto& range : itemRanges)
    {
        hFlattenedRanges.push_back(range.second.itemRange);
    }
    using HostSpan = Span<const Vector<2, IdInteger>>;
    queue.MemcpyAsync(dFlattenedRanges,
                      HostSpan(hFlattenedRanges.cbegin(),
                               hFlattenedRanges.cend()));

    uint32_t workCount = static_cast<uint32_t>(hFlattenedRanges.size());
    using namespace std::literals;
    queue.IssueSaturatingLambda
    (
        "MultiTransform Construct Spans"sv,
        KernelIssueParams{.workCount = workCount},
        [=, this] MRAY_HYBRID(KernelCallParams kp)
        {
            for(uint32_t i = 0; i < workCount; i += kp.TotalSize())
            {
                Vector<2, IdInteger> range = dFlattenedRanges[i];
                IdInteger size = range[1] - range[0];

                batchedTransforms[i] = transforms.subspan(range[0], size);
                batchedInvTransforms[i] = invTransforms.subspan(range[0], size);
            }
        }
    );


    soa.transforms = ToConstSpan(batchedTransforms);
    soa.invTransforms = ToConstSpan(batchedInvTransforms);

    // Wait here before locally deleting stuff.
    queue.Barrier().Wait();
}

void TransformGroupMulti::PushAttribute(TransformKey id , uint32_t attributeIndex,
                                         MRayInput data)
{
    switch(attributeIndex)
    {
        case 0: GenericPushData(id, transforms, std::move(data), true); break;
        case 1: GenericPushData(id, invTransforms, std::move(data), true); break;
        default: throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
                                             TypeName(), attributeIndex));
    }
}

void TransformGroupMulti::PushAttribute(TransformKey id,
                                        const Vector2ui& subRange,
                                        uint32_t attributeIndex,
                                        MRayInput data)
{
    switch(attributeIndex)
    {
        case 0: GenericPushData(id, subRange, transforms, std::move(data), true); break;
        case 1: GenericPushData(id, subRange, invTransforms, std::move(data), true); break;
        default: throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
                                             TypeName(), attributeIndex));
    }
}

void TransformGroupMulti::PushAttribute(const Vector<2, TransformKey::Type>& idRange,
                                        uint32_t attributeIndex, MRayInput data)
{
    switch(attributeIndex)
    {
        case 0: GenericPushData(idRange, transforms, std::move(data), true, true); break;
        case 1: GenericPushData(idRange, invTransforms, std::move(data), true, true); break;
        default: throw MRayError(MRAY_FORMAT("{:s}: Unkown AttributeIndex {:d}",
                                             TypeName(), attributeIndex));
    }
}

TransformGroupMulti::AttribInfoList TransformGroupMulti::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeOptionality;
    static const std::array<TransAttributeInfo, 2> LogicList =
    {
        TransAttributeInfo("Transform", MR_MANDATORY, MRayDataType<MR_MATRIX_4x4>()),
        TransAttributeInfo("InvTransform", MR_OPTIONAL, MRayDataType<MR_MATRIX_4x4>())
    };
    return std::vector(LogicList.cbegin(), LogicList.cend());
}

typename TransformGroupMulti::DataSoA TransformGroupMulti::SoA() const
{
    return soa;
}