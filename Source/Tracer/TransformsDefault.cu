#include "TransformsDefault.h"
#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgorithms.h"

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

void TransformGroupSingle::PushAttribute(TransformKey id , uint32_t attributeIndex,
                                         TransientData data, const GPUQueue& queue)
{
    if(attributeIndex == 0)
    {
        GenericPushData(transforms, id.FetchIndexPortion(),
                        attributeIndex,
                        std::move(data), queue);

        auto range = FindRange(id.FetchIndexPortion())[0];
        size_t count = range[1] - range[0];
        Span<Matrix4x4> subTRange = transforms.subspan(range[0], count);
        Span<Matrix4x4> subInvTRange = invTransforms.subspan(range[0], count);

        DeviceAlgorithms::Transform(subInvTRange, ToConstSpan(subTRange), queue,
                                    KCInvertTransforms());

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

        auto range = FindRange(id.FetchIndexPortion())[0];
        auto innerRange = Vector2ui(range[0] + subRange[0], subRange[1]);
        size_t count = innerRange[1] - innerRange[0];

        Span<Matrix4x4> subTRange = transforms.subspan(innerRange[0], count);
        Span<Matrix4x4> subInvTRange = invTransforms.subspan(innerRange[0], count);

        DeviceAlgorithms::Transform(subInvTRange, ToConstSpan(subTRange), queue,
                                    KCInvertTransforms());
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

        auto rangeStart = FindRange(idRange[0])[0];
        auto rangeEnd = FindRange(idRange[1])[1];
        size_t count = rangeEnd[1] - rangeStart[0];

        Span<Matrix4x4> subTRange = transforms.subspan(rangeStart[0], count);
        Span<Matrix4x4> subInvTRange = invTransforms.subspan(rangeStart[0], count);

        DeviceAlgorithms::Transform(subInvTRange, ToConstSpan(subTRange), queue,
                                    KCInvertTransforms());
    }
    else throw MRayError("{:s}: Unkown AttributeIndex {:d}",
                         TypeName(), attributeIndex);
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
    const GPUQueue& queue = gpuSystem.BestDevice().GetQueue(0);
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

void TransformGroupMulti::PushAttribute(TransformKey id, uint32_t attributeIndex,
                                        TransientData data, const GPUQueue& queue)
{
    if(attributeIndex == 0)
    {
        GenericPushData(transforms, id.FetchIndexPortion(), attributeIndex,
                        std::move(data), queue);

        auto range = FindRange(id.FetchIndexPortion())[0];
        size_t count = range[1] - range[0];
        Span<Matrix4x4> subTRange = transforms.subspan(range[0], count);
        Span<Matrix4x4> subInvTRange = invTransforms.subspan(range[0], count);

        DeviceAlgorithms::Transform(subInvTRange, ToConstSpan(subTRange), queue,
                                    KCInvertTransforms());
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

        auto range = FindRange(id.FetchIndexPortion())[attributeIndex];
        auto innerRange = Vector2ui(range[0] + subRange[0], subRange[1]);
        size_t count = innerRange[1] - innerRange[0];

        Span<Matrix4x4> subTRange = transforms.subspan(innerRange[0], count);
        Span<Matrix4x4> subInvTRange = invTransforms.subspan(innerRange[0], count);

        DeviceAlgorithms::Transform(subInvTRange, ToConstSpan(subTRange), queue,
                                    KCInvertTransforms());
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

        auto rangeStart = (FindRange(idRange[0]))[attributeIndex];
        auto rangeEnd   = (FindRange(idRange[1]))[attributeIndex];
        size_t count = rangeEnd[1] - rangeStart[0];

        Span<Matrix4x4> subTRange = transforms.subspan(rangeStart[0], count);
        Span<Matrix4x4> subInvTRange = invTransforms.subspan(rangeStart[0], count);

        DeviceAlgorithms::Transform(subInvTRange, ToConstSpan(subTRange), queue,
                                    KCInvertTransforms());
    }
    else throw MRayError("{:s}: Unkown AttributeIndex {:d}",
                         TypeName(), attributeIndex);
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

typename TransformGroupMulti::DataSoA TransformGroupMulti::SoA() const
{
    return soa;
}