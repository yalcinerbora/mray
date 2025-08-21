#include "GenericGroup.h"

#ifdef MRAY_GPU_BACKEND_CPU
    #include "Device/GPUSystem.hpp" // IWYU pragma: keep
#endif

template<class ID, class AI>
const AttributeRanges& GenericGroupT<ID, AI>::FindRange(IdInt id) const
{
    auto range = itemRanges.at(ID(id).FetchIndexPortion());
    if(!range)
    {
        throw MRayError("{:s}:{:d}: Unkown key {}",
                        this->Name(), this->groupId, id);
    }
    return range.value().get();
}


template<class ID, class AI>
template <class T>
void GenericGroupT<ID, AI>::GenericPushData(const Span<T>& dAttributeRegion,
                                            //
                                            IdInt id, uint32_t attribIndex,
                                            TransientData data,
                                            const GPUQueue& deviceQueue) const
{
    if(!isCommitted)
    {
        throw MRayError("{:s}:{:d}: is not committed yet. "
                        "You cannot push data!",
                        this->Name(), groupId);
    }

    assert(data.IsFull());
    auto range = FindRange(id)[attribIndex];
    size_t itemCount = range[1] - range[0];
    assert(data.Size<T>() == itemCount);

    Span<T> dSubBatch = dAttributeRegion.subspan(range[0], itemCount);
    deviceQueue.MemcpyAsync(dSubBatch, ToSpan<const T>(data));
    deviceQueue.IssueBufferForDestruction(std::move(data));
}

template<class ID, class AI>
template <class T>
void GenericGroupT<ID, AI>::GenericPushData(const Span<T>& dAttributeRegion,
                                            //
                                            Vector<2, IdInt> idRange,
                                            uint32_t attribIndex,
                                            TransientData data,
                                            const GPUQueue& deviceQueue) const
{
    if(!isCommitted)
    {
        throw MRayError("{:s}:{:d}: is not committed yet. "
                        "You cannot push data!",
                        this->Name(), groupId);
    }
    assert(data.IsFull());
    auto rangeStart = FindRange(idRange[0])[attribIndex];
    auto rangeEnd   = FindRange(idRange[1])[attribIndex];
    size_t itemCount = rangeEnd[1] - rangeStart[0];
    assert(data.Size<T>() == itemCount);

    Span<T> dSubBatch = dAttributeRegion.subspan(rangeStart[0], itemCount);
    deviceQueue.MemcpyAsync(dSubBatch, ToSpan<const T>(data));
    deviceQueue.IssueBufferForDestruction(std::move(data));
}

template<class ID, class AI>
template <class T>
void GenericGroupT<ID, AI>::GenericPushData(const Span<T>& dAttributeRegion,
                                            //
                                            IdInt id, uint32_t attribIndex,
                                            const Vector2ui& subRange,
                                            TransientData data,
                                            const GPUQueue& deviceQueue) const
{
    if(!isCommitted)
    {
        throw MRayError("{:s}:{:d}: is not committed yet. "
                        "You cannot push data!",
                        this->Name(), groupId);
    }
    assert(data.IsFull());
    auto range = FindRange(id)[attribIndex];
    size_t itemCount = subRange[1] - subRange[0];
    assert(data.Size<T>() <= itemCount);

    auto dLocalSpan = dAttributeRegion.subspan(range[0], range[1] - range[0]);
    auto dLocalSubspan = dLocalSpan.subspan(subRange[0], itemCount);
    deviceQueue.MemcpyAsync(dLocalSubspan, ToSpan<const T>(data));
    deviceQueue.IssueBufferForDestruction(std::move(data));
}

template<class ID, class AI>
GenericGroupT<ID, AI>::GenericGroupT(uint32_t groupId, const GPUSystem& s,
                                     size_t allocationGranularity,
                                     size_t initialReservationSize)
    : gpuSystem(s)
    , isCommitted(false)
    , groupId(groupId)
    , deviceMem(gpuSystem.AllGPUs(), allocationGranularity, initialReservationSize)
{}

template<class ID, class AI>
typename GenericGroupT<ID, AI>::IdList
GenericGroupT<ID, AI>::Reserve(const std::vector<AttributeCountList>& countArrayList)
{
    std::lock_guard<std::mutex> lock(mutex);
    assert(!countArrayList.empty());
    if(isCommitted)
    {
        throw MRayError("{:s}:{:d}: is in committed state, "
                        "you cannot change reservations!",
                        this->Name(), groupId);
    }
    // Lets not use zero
    IdInt lastId = (itemCounts.empty()) ? 0 : std::prev(itemCounts.end())->first + 1;
    IdList result(countArrayList.size());
    for(size_t i = 0; i < countArrayList.size(); i++)
    {
        IdInt id = lastId++;

        [[maybe_unused]]
        auto r = itemCounts.emplace(id, countArrayList[i]);
        assert(r.second);

        // Convert result to actual groupId packed id
        result[i] = ID::CombinedKey(groupId, id);
    }
    return result;
}

template<class ID, class AI>
bool GenericGroupT<ID, AI>::IsInCommitState() const
{
    return isCommitted;
}

template<class ID, class AI>
size_t GenericGroupT<ID, AI>::GPUMemoryUsage() const
{
    return deviceMem.Size();
}

template<class ID, class AI>
typename GenericGroupT<ID, AI>::IdInt
GenericGroupT<ID, AI>::GroupId() const
{
    return groupId;
}

template<class ID, class AI>
size_t GenericGroupT<ID, AI>::TotalItemCount() const
{
    return itemCounts.size();
}

template<class I, class A>
template<uint32_t D, class T>
std::vector<TracerTexView<D, T>>
GenericTexturedGroupT<I, A>::ConvertToView(std::vector<TextureId> texIds,
                                           uint32_t attributeIndex) const
{
    using ViewType = TracerTexView<D, T>;
    std::vector<ViewType> result;
    result.reserve(texIds.size());
    for(const auto& texId : texIds)
    {
        auto optView = globalTextureViews.at(texId);
        if(!optView)
        {
            throw MRayError("{:s}:{:d}: Given texture({:d}) is not found",
                            this->Name(), this->groupId,
                            static_cast<CommonKey>(texId));
        }
        const GenericTextureView& view = optView.value();
        if(!std::holds_alternative<ViewType>(view))
        {
            throw MRayError("{:s}:{:d}: Given texture({:d}) does not have "
                            "a correct type for, Attribute {:d}",
                            this->Name(), this->groupId,
                            static_cast<CommonKey>(texId), attributeIndex);
            return std::vector<TracerTexView<D, T>>{};
        }
        result.push_back(std::get<ViewType>(view));
    }
    return result;
}

template<class I, class A>
template<uint32_t D, class T>
std::vector<Optional<TracerTexView<D, T>>>
GenericTexturedGroupT<I, A>::ConvertToView(std::vector<Optional<TextureId>> texIds,
                                           uint32_t attributeIndex) const
{
    using ViewType = TracerTexView<D, T>;

    std::vector<Optional<ViewType>> result;
    result.reserve(texIds.size());
    for(const auto& texId : texIds)
    {
        if(!texId)
        {
            result.push_back(std::nullopt);
            continue;
        }
        auto optView = globalTextureViews.at(texId.value());
        if(!optView)
        {
            throw MRayError("{:s}:{:d}: Given texture({:d}) is not found",
                            this->Name(), this->groupId,
                            static_cast<CommonKey>(texId.value()));
        }
        const GenericTextureView& view = optView.value();
        if(!std::holds_alternative<ViewType>(view))
        {
            throw MRayError("{:s}:{:d}: Given texture({:d}) does not have "
                            "a correct type for, Attribute {:d}",
                            this->Name(), this->groupId,
                            static_cast<CommonKey>(texId.value()),
                            attributeIndex);
        }
        result.push_back(std::get<ViewType>(view));
    }
    return result;
}


template<class I, class A>
GenericTexturedGroupT<I, A>::GenericTexturedGroupT(uint32_t groupId, const GPUSystem& s,
                                                   const TextureViewMap& map,
                                                   size_t allocationGranularity,
                                                   size_t initialReservationSize)
    : Parent(groupId, s,
             allocationGranularity,
             initialReservationSize)
    , globalTextureViews(map)
{}

template<class I, class A>
template<uint32_t D, class T>
void GenericTexturedGroupT<I, A>::GenericPushTexAttribute(Span<ParamVaryingData<D, T>> dAttributeSpan,
                                                          //
                                                          I idStart, I idEnd,
                                                          uint32_t attributeIndex,
                                                          TransientData hData,
                                                          std::vector<Optional<TextureId>> optionalTexIds,
                                                          const GPUQueue& queue)
{
    if(!this->isCommitted)
    {
        throw MRayError("{:s}:{:d}: is not committed yet. "
                        "You cannot push data!",
                        this->Name(), this->groupId);
    }
    assert(hData.IsFull());
    auto hOptTexViews = ConvertToView<D, T>(std::move(optionalTexIds),
                                            attributeIndex);

    // Now we need to be careful
    auto rangeStart = this->FindRange(idStart.FetchIndexPortion())[attributeIndex];
    auto rangeEnd = this->FindRange(idEnd.FetchIndexPortion())[attributeIndex];
    size_t count = rangeEnd[1] - rangeStart[0];
    Span<ParamVaryingData<D, T>> dSubspan = dAttributeSpan.subspan(rangeStart[0], count);

    assert(dSubspan.size() == hOptTexViews.size());
    assert(hOptTexViews.size() == hData.Size<T>());

    // Construct in host, then memcpy
    std::vector<ParamVaryingData<D, T>> hParamVaryingData;
    hParamVaryingData.reserve(dSubspan.size());
    Span<const T> hDataSpan = hData.AccessAs<T>();
    for(uint32_t i = 0; i < hOptTexViews.size(); i++)
    {
        auto pvd = hOptTexViews[i].has_value()
            ? ParamVaryingData<D, T>(hOptTexViews[i].value())
            : ParamVaryingData<D, T>(hDataSpan[i]);
        hParamVaryingData.push_back(pvd);
    }
    auto hParamVaryingDataSpan = Span(hParamVaryingData.cbegin(),
                                      hParamVaryingData.cend());
    queue.MemcpyAsync(dSubspan, hParamVaryingDataSpan);
    // TODO: Try to find a way to remove this wait
    queue.Barrier().Wait();
}

template<class I, class A>
template<uint32_t D, class T>
void GenericTexturedGroupT<I, A>::GenericPushTexAttribute(Span<Optional<TracerTexView<D, T>>> dAttributeSpan,
                                                          //
                                                          I idStart, I idEnd,
                                                          uint32_t attributeIndex,
                                                          std::vector<Optional<TextureId>> optionalTexIds,
                                                          const GPUQueue& queue)
{
    if(!this->isCommitted)
    {
        throw MRayError("{:s}:{:d}: is not committed yet. "
                        "You cannot push data!",
                        this->Name(), this->groupId);
    }
    auto hOptTexViews = ConvertToView<D, T>(std::move(optionalTexIds),
                                            attributeIndex);

    // YOLO memcpy here! Hopefully it works
    auto rangeStart = this->FindRange(idStart.FetchIndexPortion())[attributeIndex];
    auto rangeEnd = this->FindRange(idEnd.FetchIndexPortion())[attributeIndex];
    size_t count = rangeEnd[1] - rangeStart[0];
    Span<Optional<TracerTexView<D, T>>> dSubspan = dAttributeSpan.subspan(rangeStart[0],
                                                                          count);
    Span<Optional<TracerTexView<D, T>>> hSpan(hOptTexViews.begin(), hOptTexViews.end());
    assert(hSpan.size() == dSubspan.size());
    queue.MemcpyAsync(dSubspan, ToConstSpan(hSpan));
    // TODO: Try to find a way to remove this wait
    queue.Barrier().Wait();
}

template<class I, class A>
template<uint32_t D, class T>
void GenericTexturedGroupT<I, A>::GenericPushTexAttribute(Span<TracerTexView<D, T>> dAttributeSpan,
                                                          //
                                                          I idStart, I idEnd,
                                                          uint32_t attributeIndex,
                                                          std::vector<TextureId> textureIds,
                                                          const GPUQueue& queue)
{
    if(!this->isCommitted)
    {
        throw MRayError("{:s}:{:d}: is not committed yet. "
                        "You cannot push data!",
                        this->Name(), this->groupId);
    }
    auto hTexViews = ConvertToView<D, T>(std::move(textureIds),
                                         attributeIndex);

    // YOLO memcpy here! Hopefully it works
    auto rangeStart = this->FindRange(idStart.FetchIndexPortion())[attributeIndex];
    auto rangeEnd = this->FindRange(idEnd.FetchIndexPortion())[attributeIndex];
    size_t count = rangeEnd[1] - rangeStart[0];
    Span<TracerTexView<D, T>> dSubspan = dAttributeSpan.subspan(rangeStart[0],
                                                              count);
    Span<TracerTexView<D, T>> hSpan(hTexViews.begin(), hTexViews.end());
    assert(hSpan.size() == dSubspan.size());
    queue.MemcpyAsync(dSubspan, ToConstSpan(hSpan));
    // TODO: Try to find a way to remove this wait
    queue.Barrier().Wait();
}

// ================== //
//   INSTANTIATIONS   //
// ================== //
//     PRIMITIVE      //
// ================== //
MRAY_GENERIC_PUSH_ATTRIB_INST(, PrimBatchKey, PrimAttributeInfo, Float);
MRAY_GENERIC_PUSH_ATTRIB_INST(, PrimBatchKey, PrimAttributeInfo, Vector2);
MRAY_GENERIC_PUSH_ATTRIB_INST(, PrimBatchKey, PrimAttributeInfo, Vector3);
MRAY_GENERIC_PUSH_ATTRIB_INST(, PrimBatchKey, PrimAttributeInfo, Vector3ui);
MRAY_GENERIC_PUSH_ATTRIB_INST(, PrimBatchKey, PrimAttributeInfo, Vector4);
MRAY_GENERIC_PUSH_ATTRIB_INST(, PrimBatchKey, PrimAttributeInfo, Quaternion);
// Skeleton-related
MRAY_GENERIC_PUSH_ATTRIB_INST(, PrimBatchKey, PrimAttributeInfo, Vector4uc);
MRAY_GENERIC_PUSH_ATTRIB_INST(, PrimBatchKey, PrimAttributeInfo, UNorm4x8);

// ================== //
//     TRANSFORM      //
// ================== //
MRAY_GENERIC_PUSH_ATTRIB_INST(, TransformKey, TransAttributeInfo, Matrix4x4);

// ================== //
//       CAMERA       //
// ================== //
MRAY_GENERIC_PUSH_ATTRIB_INST(, CameraKey, CamAttributeInfo, Float);
MRAY_GENERIC_PUSH_ATTRIB_INST(, CameraKey, CamAttributeInfo, Vector2);
MRAY_GENERIC_PUSH_ATTRIB_INST(, CameraKey, CamAttributeInfo, Vector3);
MRAY_GENERIC_PUSH_ATTRIB_INST(, CameraKey, CamAttributeInfo, Vector4);

// ================== //
//       MEDIUM       //
// ================== //
MRAY_GENERIC_PUSH_ATTRIB_INST(, MediumKey, MediumAttributeInfo, Float);
MRAY_GENERIC_PUSH_ATTRIB_INST(, MediumKey, MediumAttributeInfo, Vector2);
MRAY_GENERIC_PUSH_ATTRIB_INST(, MediumKey, MediumAttributeInfo, Vector3);
MRAY_GENERIC_PUSH_ATTRIB_INST(, MediumKey, MediumAttributeInfo, Vector4);
//
MRAY_GENERIC_TEX_PUSH_ATTRIB_INST(, MediumKey, MediumAttributeInfo, 3);

// ================== //
//       LIGHT        //
// ================== //
//MRAY_GENERIC_PUSH_ATTRIB_INST(, LightKey, LightAttributeInfo, Float);
//MRAY_GENERIC_PUSH_ATTRIB_INST(, LightKey, LightAttributeInfo, Vector2);
//MRAY_GENERIC_PUSH_ATTRIB_INST(, LightKey, LightAttributeInfo, Vector3);
//MRAY_GENERIC_PUSH_ATTRIB_INST(, LightKey, LightAttributeInfo, Vector4);
//
//MRAY_GENERIC_TEX_PUSH_ATTRIB_INST(, LightKey, LightAttributeInfo, 2);

// ================== //
//      MATERIAL      //
// ================== //
MRAY_GENERIC_PUSH_ATTRIB_INST(, MaterialKey, MatAttributeInfo, Float);
MRAY_GENERIC_PUSH_ATTRIB_INST(, MaterialKey, MatAttributeInfo, Vector2);
MRAY_GENERIC_PUSH_ATTRIB_INST(, MaterialKey, MatAttributeInfo, Vector3);
MRAY_GENERIC_PUSH_ATTRIB_INST(, MaterialKey, MatAttributeInfo, Vector4);
//
MRAY_GENERIC_TEX_PUSH_ATTRIB_INST(, MaterialKey, MatAttributeInfo, 2);

// ================== //
//      CLASSES       //
// ================== //
// Non-Textured
template class GenericGroupT<PrimBatchKey, PrimAttributeInfo>;
template class GenericGroupT<CameraKey, CamAttributeInfo>;
//template class GenericGroupT<TransformKey, TransAttributeInfo>;
// Textured
template class GenericTexturedGroupT<MediumKey, MediumAttributeInfo>;
template class GenericTexturedGroupT<MaterialKey, MatAttributeInfo>;
//template class GenericTexturedGroupT<LightKey, LightAttributeInfo>;