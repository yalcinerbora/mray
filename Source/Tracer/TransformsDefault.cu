#include "TransformsDefault.h"

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgGeneric.h"
#include "Device/GPUMemory.h"

struct KCInvertTransforms
{
    MRAY_HYBRID Matrix4x4 operator()(const Matrix4x4&) const;
};

MRAY_HYBRID Matrix4x4 KCInvertTransforms::operator()(const Matrix4x4& matrix) const
{
    return matrix.Inverse();
}

typename TransformGroupIdentity::DataSoA TransformGroupIdentity::SoA() const
{
    return EmptyType{};
}

TransformGroupSingle::TransformGroupSingle(uint32_t groupId,
                                           const GPUSystem& s)
    : GenericGroupTransform<TransformGroupSingle>(groupId, s)
{}

void TransformGroupSingle::CommitReservations()
{
    auto [t, it] = GenericCommit<Matrix4x4, Matrix4x4>({0, 0});
    transforms = t;
    invTransforms = it;

    soa.transforms = ToConstSpan(transforms);
    soa.invTransforms = ToConstSpan(invTransforms);
}

TransAttributeInfoList TransformGroupSingle::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeOptionality;
    using enum AttributeIsArray;
    static const TransAttributeInfoList LogicList =
    {
        TransAttributeInfo("Transform", MRayDataType<MR_MATRIX_4x4>(), IS_SCALAR, MR_MANDATORY)
    };
    return LogicList;
}

void TransformGroupSingle::PushAttribute(TransformKey id , uint32_t attributeIndex,
                                         TransientData data, const GPUQueue& queue)
{
    if(attributeIndex == 0)
    {
        GenericPushData(transforms, id.FetchIndexPortion(),
                        attributeIndex,
                        std::move(data), queue);
    }
    else throw MRayError("{:s}: Unkown AttributeIndex {:d}",
                         TypeName(), attributeIndex);
}

void TransformGroupSingle::PushAttribute(TransformKey id,
                                         uint32_t attributeIndex,
                                         const Vector2ui& subRange,
                                         TransientData data,
                                         const GPUQueue& queue)
{
    if(attributeIndex == 0)
    {
        GenericPushData(transforms, id.FetchIndexPortion(),
                        attributeIndex, subRange,
                        std::move(data), queue);
    }
    else throw MRayError("{:s}: Unkown AttributeIndex {:d}",
                         TypeName(), attributeIndex);
}

void TransformGroupSingle::PushAttribute(TransformKey idStart, TransformKey idEnd,
                                         uint32_t attributeIndex, TransientData data,
                                         const GPUQueue& queue)
{
    if(attributeIndex == 0)
    {
        auto idRange = Vector<2, IdInt>(idStart.FetchIndexPortion(),
                                        idEnd.FetchIndexPortion());
        GenericPushData(transforms, idRange, attributeIndex,
                        std::move(data), queue);
    }
    else throw MRayError("{:s}: Unkown AttributeIndex {:d}",
                         TypeName(), attributeIndex);
}

void TransformGroupSingle::Finalize(const GPUQueue& queue)
{
    DeviceAlgorithms::Transform(invTransforms, ToConstSpan(transforms), queue,
                                KCInvertTransforms());
}

typename TransformGroupSingle::DataSoA TransformGroupSingle::SoA() const
{
    return soa;
}

TransformGroupMulti::TransformGroupMulti(uint32_t groupId,
                                         const GPUSystem& s)
    : GenericGroupTransform<TransformGroupMulti>(groupId, s)
{}

void TransformGroupMulti::CommitReservations()
{
    Span<Span<const Matrix4x4>> batchedTransforms;
    Span<Span<const Matrix4x4>> batchedInvTransforms;

    //auto [t, it, bt22, bit22] =
    auto [t, it, batchT, batchInvT] =
        GenericCommit<Matrix4x4, Matrix4x4,
                      Span<const Matrix4x4>,
                      Span<const Matrix4x4>>({0, 0, 0, 0});
    transforms = t;
    invTransforms = it;
    batchedTransforms = batchT;
    batchedTransforms = batchInvT;

    // TODO: Improve this?
    // Locally creating buffers ...
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    DeviceLocalMemory tempMemory(gpuSystem.BestDevice());
    Span<Vector<2, size_t>> dFlattenedRanges;
    MemAlloc::AllocateMultiData(std::tie(dFlattenedRanges),
                                tempMemory, {itemRanges.size()});

    std::vector<Vector<2, size_t>> hFlattenedRanges;
    hFlattenedRanges.reserve(itemRanges.size());
    for(const auto& range : itemRanges)
        hFlattenedRanges.push_back(range.second[0]);

    using HostSpan = Span<const Vector<2, size_t>>;
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
                Vector<2, size_t> range = dFlattenedRanges[i];
                size_t size = range[1] - range[0];

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

TransAttributeInfoList TransformGroupMulti::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeOptionality;
    using enum AttributeIsArray;
    static const TransAttributeInfoList LogicList =
    {
        TransAttributeInfo("Transform", MRayDataType<MR_MATRIX_4x4>(), IS_ARRAY, MR_MANDATORY)
    };
    return LogicList;
}

void TransformGroupMulti::PushAttribute(TransformKey id, uint32_t attributeIndex,
                                        TransientData data, const GPUQueue& queue)
{
    if(attributeIndex == 0)
    {
        GenericPushData(transforms, id.FetchIndexPortion(), attributeIndex,
                        std::move(data), queue);
    }
    else throw MRayError("{:s}: Unkown AttributeIndex {:d}",
                         TypeName(), attributeIndex);
}

void TransformGroupMulti::PushAttribute(TransformKey id,
                                        uint32_t attributeIndex,
                                        const Vector2ui& subRange,
                                        TransientData data,
                                        const GPUQueue& queue)
{
    if(attributeIndex == 0)
    {
        GenericPushData(transforms, id.FetchIndexPortion(),
                        attributeIndex, subRange,
                        std::move(data), queue);
    }
    else throw MRayError("{:s}: Unkown AttributeIndex {:d}",
                         TypeName(), attributeIndex);
}

void TransformGroupMulti::PushAttribute(TransformKey idStart, TransformKey idEnd,
                                        uint32_t attributeIndex, TransientData data,
                                        const GPUQueue& queue)
{
    if(attributeIndex == 0)
    {
        auto idRange = Vector<2, IdInt>(idStart.FetchIndexPortion(),
                                        idEnd.FetchIndexPortion());
        GenericPushData(transforms, idRange, attributeIndex,
                        std::move(data), queue);
    }
    else throw MRayError("{:s}: Unkown AttributeIndex {:d}",
                         TypeName(), attributeIndex);
}

void TransformGroupMulti::Finalize(const GPUQueue& queue)
{
    // Altough Multi tranform has non-regular arrayed transformations
    // We can directly convert each transform individually
    // since each transform is independent from each other.
    // So a single invert call should suffice
    DeviceAlgorithms::Transform(invTransforms, ToConstSpan(transforms), queue,
                                KCInvertTransforms());
}

typename TransformGroupMulti::DataSoA TransformGroupMulti::SoA() const
{
    return soa;
}