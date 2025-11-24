#include "AcceleratorC.h"
#include "Device/GPUSystem.hpp" // IWYU pragma: keep
#include <memory>

class KeyGeneratorFunctor
{
    CommonKey accelBatchId;
    CommonKey accelIdStart;
    Span<AcceleratorKey> dLocalKeyWriteRegion;

    public:
    KeyGeneratorFunctor(CommonKey accelBatchIdIn,
                        CommonKey accelIdStartIn,
                        Span<AcceleratorKey> dWriteRegions)
        : accelBatchId(accelBatchIdIn)
        , accelIdStart(accelIdStartIn)
        , dLocalKeyWriteRegion(dWriteRegions)
    {}

    MR_PF_DECL_V
    void operator()(KernelCallParams kp) const noexcept
    {
        uint32_t keyCount = static_cast<uint32_t>(dLocalKeyWriteRegion.size());
        for(uint32_t i = kp.GlobalId(); i < keyCount; i += kp.TotalSize())
            dLocalKeyWriteRegion[i] = AcceleratorKey::CombinedKey(accelBatchId, i + accelIdStart);
    }
};

AccelPartitionResult AcceleratorGroup::PartitionParamsForWork(const AccelGroupConstructParams& p)
{
    using IndexPack = typename AccelPartitionResult::IndexPack;
    AccelPartitionResult result = {};
    result.workPartitionOffsets.reserve(p.tGroupSurfs.size() + p.tGroupLightSurfs.size() + 1);
    result.packedIndices.reserve(p.tGroupSurfs.size() + p.tGroupLightSurfs.size());

    // This is somewhat iota but we will add lights
    for(uint32_t tIndex = 0; tIndex < p.tGroupSurfs.size(); tIndex++)
    {
        const auto& groupedSurf = p.tGroupSurfs[tIndex];
        // Add this directly
        result.packedIndices.emplace_back(IndexPack
                                          {
                                              .tId = groupedSurf.first,
                                              .tSurfIndex = tIndex,
                                              .tLightSurfIndex = std::numeric_limits<uint32_t>::max()
                                          });
    }

    // Add the lights as well
    for(uint32_t ltIndex = 0; ltIndex < p.tGroupLightSurfs.size(); ltIndex++)
    {
        const auto& groupedLightSurf = p.tGroupLightSurfs[ltIndex];
        TransGroupId tId = groupedLightSurf.first;
        auto loc = std::find_if(p.tGroupSurfs.cbegin(),
                                p.tGroupSurfs.cend(),
                                [tId](const auto& groupedSurf)
        {
            return groupedSurf.first == tId;
        });

        if(loc != p.tGroupSurfs.cend())
        {
            size_t index = size_t(std::distance(loc, p.tGroupSurfs.begin()));
            result.packedIndices[index].tLightSurfIndex = ltIndex;
        }
        else
        {
            result.packedIndices.emplace_back(IndexPack
                                              {
                                                  .tId = groupedLightSurf.first,
                                                  .tSurfIndex = std::numeric_limits<uint32_t>::max(),
                                                  .tLightSurfIndex = ltIndex
                                              });
        }
    }

    // Find the partitioned offsets
    result.workPartitionOffsets.resize(result.packedIndices.size() + 1, 0);
    std::transform_inclusive_scan(result.packedIndices.cbegin(),
                                  result.packedIndices.cend(),
                                  result.workPartitionOffsets.begin() + 1,
                                  std::plus{},
                                  [&p](const IndexPack& indexPack) -> uint32_t
    {
        uint32_t result = 0;
        if(indexPack.tSurfIndex != std::numeric_limits<uint32_t>::max())
        {
            size_t size = p.tGroupSurfs[indexPack.tSurfIndex].second.size();
            result += static_cast<uint32_t>(size);
        }
        if(indexPack.tLightSurfIndex != std::numeric_limits<uint32_t>::max())
        {
            size_t size = p.tGroupLightSurfs[indexPack.tLightSurfIndex].second.size();
            result += static_cast<uint32_t>(size);
        };
        return result;
    });
    assert(result.packedIndices.size() + 1 == result.workPartitionOffsets.size());
    result.totalInstanceCount = result.workPartitionOffsets.back();

    return result;

};

AccelLeafResult AcceleratorGroup::DetermineConcreteAccelCount(std::vector<SurfacePrimList> instancePrimBatches,
                                                              const std::vector<PrimRangeArray>& instancePrimRanges,
                                                              PrimTransformType primTransformType)
{
    assert(instancePrimBatches.size() == instancePrimRanges.size());

    std::vector<uint32_t> acceleratorIndices(instancePrimBatches.size());
    std::vector<uint32_t> uniqueIndices(instancePrimBatches.size());
    std::iota(acceleratorIndices.begin(), acceleratorIndices.end(), 0);
    using enum PrimTransformType;
    if(primTransformType == LOCALLY_CONSTANT_TRANSFORM)
    {
        // This means we can fully utilize the primitive
        // Sort and find the unique primitive groups
        // only generate accelerators for these, refer with other instances
        std::vector<uint32_t>& nonUniqueIndices = acceleratorIndices;
        // Commonalize the inner lists (sort it)
        for(SurfacePrimList& lst : instancePrimBatches)
        {
            // Hopefully stl calls insertion sort here or something...
            std::sort(lst.begin(), lst.end(), [](PrimBatchId lhs, PrimBatchId rhs)
            {
                return (std::bit_cast<CommonKey>(lhs) < std::bit_cast<CommonKey>(rhs));
            });
        }
        // Do an index/id sort here, c++ does not have it
        // so sort iota wrt. values and use it to find instances
        // Use stable sort here to keep the relative transform order of primitives
        std::stable_sort(nonUniqueIndices.begin(), nonUniqueIndices.end(),
                         [&instancePrimBatches](uint32_t l, uint32_t r)
        {
            const SurfacePrimList& lhs = instancePrimBatches[l];
            const SurfacePrimList& rhs = instancePrimBatches[r];
            bool result = std::lexicographical_compare(lhs.cbegin(), lhs.cend(),
                                                       rhs.cbegin(), rhs.cend(),
                                                       [](PrimBatchId lhs, PrimBatchId rhs)
            {
                return (std::bit_cast<CommonKey>(lhs) < std::bit_cast<CommonKey>(rhs));
            });
            return result;
        });
        auto endLoc = std::unique_copy(nonUniqueIndices.begin(), nonUniqueIndices.end(), uniqueIndices.begin(),
                                       [&instancePrimBatches](uint32_t l, uint32_t r)
        {
            const SurfacePrimList& lhs = instancePrimBatches[l];
            const SurfacePrimList& rhs = instancePrimBatches[r];
            auto result = std::lexicographical_compare_three_way(lhs.cbegin(), lhs.cend(),
                                                                 rhs.cbegin(), rhs.cend(),
                                                                 [](PrimBatchId lhs, PrimBatchId rhs)
            {
                return (std::bit_cast<CommonKey>(lhs) <=> std::bit_cast<CommonKey>(rhs));
            });
            return std::is_eq(result);
        });
        uniqueIndices.erase(endLoc, uniqueIndices.end());

        // Do an inverted search your id on unique indices
        std::iota(nonUniqueIndices.begin(), nonUniqueIndices.end(), 0);
        for(uint32_t& index : nonUniqueIndices)
        {
            auto loc = std::lower_bound(uniqueIndices.begin(), uniqueIndices.end(), index,
                                        [&instancePrimBatches](uint32_t value, uint32_t checked)
            {
                const SurfacePrimList& lhs = instancePrimBatches[value];
                const SurfacePrimList& rhs = instancePrimBatches[checked];
                auto result = std::lexicographical_compare_three_way(lhs.cbegin(), lhs.cend(),
                                                                     rhs.cbegin(), rhs.cend(),
                                                                     [](PrimBatchId lhs, PrimBatchId rhs)
                {
                    return (std::bit_cast<CommonKey>(lhs) <=> std::bit_cast<CommonKey>(rhs));
                });
                return std::is_lt(result);

            });
            index = static_cast<uint32_t>(std::distance(uniqueIndices.begin(), loc));
        }

    }
    else
    {
        // PG supports "PER_PRIMITIVE_TRANSFORM", we cannot refer to the same
        // accelerator, we need to construct an accelerator for each instance
        // Do not bother sorting etc here..
        // Copy here for continuation
        uniqueIndices = acceleratorIndices;
    }

    // Determine the leaf ranges
    // TODO: We are
    std::vector<uint32_t> uniquePrimCounts;
    uniquePrimCounts.reserve(uniqueIndices.size() + 1);
    uniquePrimCounts.push_back(0);
    for(uint32_t index : uniqueIndices)
    {
        uint32_t totalLocalPrimCount = 0;
        for(Vector2ui range : instancePrimRanges[index])
        {
            if(range == Vector2ui(std::numeric_limits<uint32_t>::max())) break;
            totalLocalPrimCount += (range[1] - range[0]);
        }
        uniquePrimCounts.push_back(totalLocalPrimCount);
    }
    std::inclusive_scan(uniquePrimCounts.cbegin() + 1, uniquePrimCounts.cend(),
                        uniquePrimCounts.begin() + 1);

    // Rename the variable for better maintenance
    std::vector<uint32_t>& uniquePrimOffsets = uniquePrimCounts;
    // Acquire instance leaf ranges
    std::vector<Vector2ui> instanceLeafRangeList;
    instanceLeafRangeList.reserve(instancePrimBatches.size());
    for(uint32_t index : acceleratorIndices)
    {
        instanceLeafRangeList.push_back(Vector2ui(uniquePrimOffsets[index],
                                                  uniquePrimOffsets[index + 1]));
    }

    // Find leaf of unique indices
    std::vector<Vector2ui> concreteLeafRangeList;
    std::vector<PrimRangeArray> concretePrimRangeList;
    concreteLeafRangeList.reserve(uniqueIndices.size());
    concretePrimRangeList.reserve(uniqueIndices.size());
    for(size_t index = 0; index < uniqueIndices.size(); index++)
    {
        concreteLeafRangeList.push_back(Vector2ui(uniquePrimOffsets[index],
                                                  uniquePrimOffsets[index + 1]));
        uint32_t lookupIndex = uniqueIndices[index];
        concretePrimRangeList.push_back(instancePrimRanges[lookupIndex]);
    }

    // All done!
    return AccelLeafResult
    {
        .concreteIndicesOfInstances = std::move(acceleratorIndices),
        .instanceLeafRanges = std::move(instanceLeafRangeList),
        .concreteLeafRanges = std::move(concreteLeafRangeList),
        .concretePrimRanges = std::move(concretePrimRangeList)
    };
}

LinearizedSurfaceData AcceleratorGroup::LinearizeSurfaceData(const AccelGroupConstructParams& p,
                                                             const AccelPartitionResult& partitions,
                                                             const GenericGroupPrimitiveT& pg,
                                                             std::string_view typeName)
{
    assert(std::is_sorted(p.globalVolumeList->cbegin(),
                          p.globalVolumeList->cend(),
                          [](const auto& a, const auto& b)
    {
        return a.first < b.first;
    }));

    LinearizedSurfaceData result = {};
    result.volumeIndices.reserve(partitions.totalInstanceCount);
    result.primRanges.reserve(partitions.totalInstanceCount);
    result.lightOrMatKeys.reserve(partitions.totalInstanceCount);
    result.alphaMaps.reserve(partitions.totalInstanceCount);
    result.cullFaceFlags.reserve(partitions.totalInstanceCount);
    result.transformKeys.reserve(partitions.totalInstanceCount);
    result.instancePrimBatches.reserve(partitions.totalInstanceCount);

    const auto InitRest = [&](uint32_t restStart)
    {
        using namespace TracerConstants;
        for(uint32_t i = restStart; i < static_cast<uint32_t>(MaxPrimBatchPerSurface); i++)
        {
            result.volumeIndices.back()[i] = VolumeIndex::InvalidKey();
            result.alphaMaps.back()[i] = std::nullopt;
            result.cullFaceFlags.back()[i] = false;
            result.lightOrMatKeys.back()[i] = LightOrMatKey::InvalidKey();
            result.primRanges.back()[i] = Vector2ui(std::numeric_limits<uint32_t>::max());
        }
    };

    const auto LoadSurf = [&](const SurfaceParams& surf)
    {
        result.volumeIndices.emplace_back();
        result.instancePrimBatches.push_back(surf.primBatches);
        result.alphaMaps.emplace_back();
        result.cullFaceFlags.emplace_back();
        result.lightOrMatKeys.emplace_back();
        result.primRanges.emplace_back();
        result.transformKeys.emplace_back(std::bit_cast<TransformKey>(surf.transformId));

        assert(surf.alphaMaps.size() == surf.cullFaceFlags.size());
        assert(surf.cullFaceFlags.size() == surf.materials.size());
        assert(surf.materials.size() == surf.primBatches.size());
        for(uint32_t i = 0; i < static_cast<uint32_t>(surf.alphaMaps.size()); i++)
        {
            if(surf.alphaMaps[i].has_value())
            {
                auto optView = p.textureViews->at(surf.alphaMaps[i].value());
                if(!optView)
                {
                    throw MRayError("{:s}: Alpha map texture({:d}) is not found",
                                    typeName, static_cast<CommonKey>(surf.alphaMaps[i].value()));
                }
                const GenericTextureView& view = optView.value();
                if(!std::holds_alternative<AlphaMap>(view))
                {
                    throw MRayError("{:s}: Alpha map texture({:d}) is not a single channel texture!",
                                    typeName, static_cast<CommonKey>(surf.alphaMaps[i].value()));
                }
                result.alphaMaps.back()[i] = std::get<AlphaMap>(view);
            }
            else result.alphaMaps.back()[i] = std::nullopt;

            result.cullFaceFlags.back()[i] = surf.cullFaceFlags[i];
            PrimBatchKey pBatchKey = std::bit_cast<PrimBatchKey>(surf.primBatches[i]);
            result.primRanges.back()[i] = pg.BatchRange(pBatchKey);
            MaterialKey mKey = std::bit_cast<MaterialKey>(surf.materials[i]);
            result.lightOrMatKeys.back()[i] = LightOrMatKey::CombinedKey(IS_MAT_KEY_FLAG,
                                                                         mKey.FetchBatchPortion(),
                                                                         mKey.FetchIndexPortion());
            //
            result.volumeIndices.back()[i] = VolumeIndex::InvalidKey();
            if(surf.volumes[i] != TracerConstants::InvalidVolume)
            {
                Pair<VolumeId, VolumeKeyPack> checkVol = {surf.volumes[i], {}};
                auto vLoc = std::lower_bound(p.globalVolumeList->cbegin(), p.globalVolumeList->cend(),
                                             checkVol,
                [](const auto& a, const auto& b) -> bool
                {
                    return a.first < b.first;
                });
                if(vLoc == p.globalVolumeList->cend() ||
                   vLoc->first != checkVol.first)
                {
                    throw MRayError("{:s}: Volume {} is not found!",
                                    typeName, CommonId(checkVol.first));
                }
                auto vIndex = CommonKey(std::distance(p.globalVolumeList->cbegin(), vLoc));
                using namespace TracerConstants;
                auto isInterfaceMat = (mKey.FetchBatchPortion() == CommonKey(PassthroughMatGroupId))
                                        ? IS_PASSTHROUGH_MAT_FLAG
                                        : IS_NOT_PASSTHROUGH_MAT_FLAG;
                result.volumeIndices.back()[i] = VolumeIndex::CombinedKey(isInterfaceMat, 0, vIndex);
            }
        }
        InitRest(static_cast<uint32_t>(surf.alphaMaps.size()));
    };

    const auto LoadLightSurf = [&](const LightSurfaceParams& lSurf)
    {
        result.volumeIndices.emplace_back();
        result.alphaMaps.emplace_back();
        result.cullFaceFlags.emplace_back();
        result.lightOrMatKeys.emplace_back();
        result.primRanges.emplace_back();
        result.instancePrimBatches.emplace_back();
        result.transformKeys.emplace_back();
        InitRest(0);

        LightKey lKey = std::bit_cast<LightKey>(lSurf.lightId);
        PrimBatchKey primBatchKey = p.lightGroup->LightPrimBatch(lKey);
        result.primRanges.back().front() = pg.BatchRange(primBatchKey);
        result.lightOrMatKeys.back().front() = LightOrMatKey::CombinedKey(IS_LIGHT_KEY_FLAG,
                                                                          lKey.FetchBatchPortion(),
                                                                          lKey.FetchIndexPortion());

        PrimBatchId primBatchId = std::bit_cast<PrimBatchId>(primBatchKey);
        result.transformKeys.back() = std::bit_cast<TransformKey>(lSurf.transformId);
        result.instancePrimBatches.back().push_back(primBatchId);
        // Lights are ray sinks, so it does not matter to set volume
        result.volumeIndices.back().front() = VolumeIndex::InvalidKey();
    };

    for(const auto& pIndices : partitions.packedIndices)
    {
        if(pIndices.tSurfIndex != std::numeric_limits<uint32_t>::max())
        {
            for(const auto& [_, surf] : p.tGroupSurfs[pIndices.tSurfIndex].second)
            {
                LoadSurf(surf);
            }
        }
        //
        if(pIndices.tLightSurfIndex != std::numeric_limits<uint32_t>::max())
        {
            for(const auto& [_, lSurf] : p.tGroupLightSurfs[pIndices.tLightSurfIndex].second)
            {
                LoadLightSurf(lSurf);
            }
        }
    }

    assert(result.volumeIndices.size() == partitions.totalInstanceCount);
    assert(result.alphaMaps.size() == partitions.totalInstanceCount);
    assert(result.cullFaceFlags.size() == partitions.totalInstanceCount);
    assert(result.lightOrMatKeys.size() == partitions.totalInstanceCount);
    assert(result.primRanges.size() == partitions.totalInstanceCount);
    assert(result.transformKeys.size() == partitions.totalInstanceCount);
    assert(result.instancePrimBatches.size() == partitions.totalInstanceCount);
    return result;
}

PreprocessResult
AcceleratorGroup::PreprocessConstructionParams(const AccelGroupConstructParams& p)
{
    assert(&pg == p.primGroup);
    // Instance Types (determined by transform type)
    AccelPartitionResult partitions = PartitionParamsForWork(p);
    // Total instance count (equivalently total surface count)
    auto linSurfData = LinearizeSurfaceData(p, partitions, pg, this->Name());
    // Find out the concrete accel count and offsets
    auto leafResult = DetermineConcreteAccelCount(std::move(linSurfData.instancePrimBatches),
                                                  linSurfData.primRanges,
                                                  pg.TransformLogicRT());

    concreteIndicesOfInstances = std::move(leafResult.concreteIndicesOfInstances);
    instanceLeafRanges = std::move(leafResult.instanceLeafRanges);
    concreteLeafRanges = std::move(leafResult.concreteLeafRanges);

    // Instantiate Works
    workInstanceOffsets = std::move(partitions.workPartitionOffsets);
    uint32_t i = 0;
    for(const auto& indices : partitions.packedIndices)
    {
        auto tGroupOpt = p.transformGroups->at(indices.tId);
        if(!tGroupOpt)
        {
            throw MRayError("{:s}:{:d}: Unable to find transform {:d}",
                            this->Name(), accelGroupId,
                            static_cast<CommonKey>(indices.tId));
        }
        const GenericGroupTransformT& tGroup = *tGroupOpt.value().get().get();

        using namespace TypeNameGen::CompTime;
        std::string workTypeName = AccelWorkTypeName(this->Name(), tGroup.Name());
        auto workGenOpt = accelWorkGenerators.at(workTypeName);
        if(!workGenOpt)
        {
            throw MRayError("{:s}:{:d}: Unable to find generator for work \"{:s}\"",
                            this->Name(), accelGroupId, workTypeName);
        }
        const auto& workGen = workGenOpt.value().get();
        workInstances.try_emplace(i, workGen(*this, tGroup));
        i++;
    }
    return PreprocessResult
    {
        .surfData = std::move(linSurfData),
        .concretePrimRanges = std::move(leafResult.concretePrimRanges)
    };
}

void AcceleratorGroup::WriteInstanceKeysAndAABBsInternal(Span<AABB3> aabbWriteRegion,
                                                         Span<AcceleratorKey> keyWriteRegion,
                                                         // Input
                                                         Span<const PrimitiveKey> dAllLeafs,
                                                         Span<const TransformKey> dTransformKeys,
                                                         // Constants
                                                         const GPUQueue& queue) const
{
    // Sanity Checks
    assert(aabbWriteRegion.size() == concreteIndicesOfInstances.size());
    assert(keyWriteRegion.size() == concreteIndicesOfInstances.size());

    size_t totalInstanceCount = concreteIndicesOfInstances.size();
    size_t concreteAccelCount = concreteLeafRanges.size();

    // We will use a temp memory here
    // TODO: Add stream ordered memory allocator stuff to the
    // Device abstraction side maybe?
    DeviceLocalMemory tempMem(*queue.Device());

    using enum PrimTransformType;
    if(pg.TransformLogicRT() == LOCALLY_CONSTANT_TRANSFORM)
    {
        using namespace DeviceAlgorithms;
        size_t tmSize = SegmentedTransformReduceTMSize<AABB3, PrimitiveKey>(concreteLeafRanges.size(),
                                                                            queue);
        Span<uint32_t> dConcreteIndicesOfInstances;
        Span<AABB3> dConcreteAABBs;
        Span<uint32_t> dConcreteLeafOffsets;
        Span<Byte> dTransformSegReduceTM;
        MemAlloc::AllocateMultiData(Tie(dConcreteIndicesOfInstances,
                                        dConcreteAABBs,
                                        dConcreteLeafOffsets,
                                        dTransformSegReduceTM),
                                    tempMem,
                                    {totalInstanceCount, concreteAccelCount,
                                     concreteAccelCount + 1,
                                     tmSize});
        Span<const uint32_t> hConcreteIndicesOfInstances(concreteIndicesOfInstances.data(),
                                                         concreteIndicesOfInstances.size());
        Span<const Vector2ui> hConcreteLeafRanges(concreteLeafRanges.data(),
                                                  concreteLeafRanges.size());
        // Normal copy to GPU
        queue.MemcpyAsync(dConcreteIndicesOfInstances, hConcreteIndicesOfInstances);

        // We need to copy the Vector2ui [(0, n_0), [n_0, n_1), ..., [n_{m-1}, n_m)]
        // As [n_0, n_1, ..., n_{m-1}, n_m]
        // This is technically UB maybe?
        // But it is hard to recognize by the compiler maybe? Dunno
        // Do a sanity check at least...
        static_assert(sizeof(Vector2ui) == 2 * sizeof(typename Vector2ui::InnerType));
        Span<const uint32_t> hConcreteLeafRangesInt(hConcreteLeafRanges.data()->AsArray().data(),
                                                    hConcreteLeafRanges.size() * Vector2ui::Dims);

        // Memset the first element to zero
        queue.MemsetAsync(dConcreteLeafOffsets.subspan(0, 1), 0x00);
        queue.MemcpyAsyncStrided(dConcreteLeafOffsets.subspan(1), 0,
                                 hConcreteLeafRangesInt.subspan(1), sizeof(Vector2ui));

        pg.SegmentedTransformReduceAABBs
        (
            dConcreteAABBs,
            dTransformSegReduceTM,
            dAllLeafs,
            ToConstSpan(dConcreteLeafOffsets),
            queue
        );
        // Now, copy (and transform) concreteAABBs (which are on local space)
        // to actual accelerator instance aabb's (after transform these will be
        // in world space)
        for(const auto& kv : workInstances)
        {
            CommonKey index = kv.first;
            const AccelWorkPtr& workPtr = kv.second;
            size_t size = (workInstanceOffsets[index + 1] -
                           workInstanceOffsets[index]);
            Span<const uint32_t> dLocalIndices = dConcreteIndicesOfInstances.subspan(workInstanceOffsets[index], size);
            Span<const TransformKey> dLocalTKeys = dTransformKeys.subspan(workInstanceOffsets[index], size);
            Span<AABB3> dLocalAABBWriteRegion = aabbWriteRegion.subspan(workInstanceOffsets[index], size);

            workPtr->TransformLocallyConstantAABBs(dLocalAABBWriteRegion,
                                                   dConcreteAABBs,
                                                   dLocalIndices,
                                                   dLocalTKeys,
                                                   queue);
        }

        // TODO: This is actually common part, but compiler gives unreachable code error
        // due to below part is not yet implemented. So it is moved here until that portion is
        // implemented.
        //
        // Now, copy (and transform) concreteAABBs (which are on local space)
        // to actual accelerator instance aabb's (after transform these will be
        // in world space)
        for(const auto& kv : workInstances)
        {
            CommonKey index = kv.first;
            size_t wIOffset = workInstanceOffsets[index];
            size_t size = (workInstanceOffsets[index + 1] - wIOffset);
            // Copy the keys as well
            Span<AcceleratorKey> dLocalKeyWriteRegion = keyWriteRegion.subspan(wIOffset, size);
            CommonKey accelBatchId = globalWorkIdToLocalOffset + index;
            using namespace std::string_literals;
            static const auto KernelName = "KCCopyLocalAccelKeys-"s + std::string(this->Name());

            queue.IssueWorkLambda
            (
                KernelName,
                DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(size)},
                KeyGeneratorFunctor(accelBatchId, static_cast<uint32_t>(wIOffset),
                                    dLocalKeyWriteRegion)
            );
        }
        //  Don't forget to wait for temp memory!
        queue.Barrier().Wait();
    }
    else
    {
        throw MRayError("{}: PER_PRIM_TRANSFORM Accel Construct not yet implemented", this->Name());
    }
}


void BaseAccelerator::PartitionSurfaces(std::vector<AccelGroupConstructParams>& partitions,
                                        const BaseAccelConstructParams& cParams)
{
    using SurfParam = typename BaseAccelConstructParams::SurfPair;
    assert(std::is_sorted(cParams.mSurfList.begin(),
                          cParams.mSurfList.end(), SurfaceLessThan));

    // TODO: One linear access to vector should be enough
    // to generate this after sort, but this is simpler to write
    // change this if this is a perf bottleneck.
    auto start = cParams.mSurfList.begin();
    while(start != cParams.mSurfList.end())
    {
        auto pBatchId = start->second.primBatches.front();
        CommonKey pGroupId = PrimGroupIdFetcher()(pBatchId);
        auto end = std::upper_bound(start, cParams.mSurfList.end(), pGroupId,
        [](CommonKey value, const SurfParam& surf)
        {
            CommonKey batchPortion = PrimGroupIdFetcher()(surf.second.primBatches.front());
            return value < batchPortion;
        });

        partitions.emplace_back(AccelGroupConstructParams{});
        auto pGroupOpt = cParams.primGroups.at(PrimGroupId(pGroupId));
        if(!pGroupOpt)
        {
            throw MRayError("{:s}: Unable to find primitive group()",
                            Name(), pGroupId);
        };
        partitions.back().primGroup = pGroupOpt.value().get().get();
        partitions.back().textureViews = &cParams.texViewMap;
        partitions.back().globalVolumeList = &cParams.globalVolumeList;
        partitions.back().transformGroups = &cParams.transformGroups;
        auto innerStart = start;
        while(innerStart != end)
        {
            TransformId tId = innerStart->second.transformId;
            CommonKey tGroupId = TransGroupIdFetcher()(tId);
            auto innerEnd = std::upper_bound(innerStart, end, tGroupId,
            [](CommonKey value, const SurfParam& surf)
            {
                auto tId = surf.second.transformId;
                return value < TransGroupIdFetcher()(tId);
            });

            auto surfSpan = Span<const SurfParam>(std::to_address(innerStart),
                                                 size_t(innerEnd - innerStart));
            partitions.back().tGroupSurfs.emplace_back(TransGroupId(tGroupId), surfSpan);
            innerStart = innerEnd;
        }
        start = end;
    }
}

void BaseAccelerator::AddLightSurfacesToPartitions(std::vector<AccelGroupConstructParams>& partitions,
                                                   const BaseAccelConstructParams& cParams)
{
    using LightSurfP = typename BaseAccelConstructParams::LightSurfPair;
    assert(std::is_sorted(cParams.lSurfList.begin(), cParams.lSurfList.end(),
                          LightSurfaceLessThan));

    // Now partition
    auto start = cParams.lSurfList.begin();
    while(start != cParams.lSurfList.end())
    {
        CommonKey lGroupId = LightGroupIdFetcher()(start->second.lightId);
        auto end = std::upper_bound(start, cParams.lSurfList.end(), lGroupId,
        [](CommonKey value, const LightSurfP& surf)
        {
            CommonKey batchPortion = LightGroupIdFetcher()(surf.second.lightId);
            return value < batchPortion;
        });

        //
        auto groupId = LightGroupId(lGroupId);
        auto lGroupOpt = cParams.lightGroups.at(groupId);
        if(!lGroupOpt)
        {
            throw MRayError("{:s}: Unable to find light group()",
                            Name(), lGroupId);
        }
        const GenericGroupLightT* lGroup = lGroupOpt.value().get().get();

        // Skip if not primitive backed
        if(!lGroup->IsPrimitiveBacked())
        {
            start = end;
            continue;
        }

        const GenericGroupPrimitiveT* pGroup = &lGroup->GenericPrimGroup();
        auto slot = std::find_if(partitions.begin(), partitions.end(),
        [pGroup](const auto& partition)
        {
            return (partition.primGroup == pGroup);
        });
        if(slot == partitions.end())
        {
            partitions.emplace_back(AccelGroupConstructParams
            {
                .transformGroups = &cParams.transformGroups,
                .textureViews = &cParams.texViewMap,
                .globalVolumeList = &cParams.globalVolumeList,
                .primGroup = pGroup,
                .lightGroup = lGroup,
                .tGroupSurfs = {},
                .tGroupLightSurfs = {}
            });
            slot = partitions.end() - 1;
        }
        else if(slot->lightGroup == nullptr) slot->lightGroup = lGroup;

        // Sub-partition wrt. transform
        auto innerStart = start;
        while(innerStart != end)
        {
            TransformId tId = innerStart->second.transformId;
            CommonKey tGroupId = TransGroupIdFetcher()(tId);
            auto innerEnd = std::upper_bound(innerStart, end, tGroupId,
            [](CommonKey value, const LightSurfP& surf) -> bool
            {
                auto tId = surf.second.transformId;
                return (value < TransGroupIdFetcher()(tId));
            });
            size_t elemCount = static_cast<size_t>(std::distance(innerStart, innerEnd));
            size_t startDistance = static_cast<size_t>(std::distance(cParams.lSurfList.begin(), innerStart));
            slot->tGroupLightSurfs.emplace_back(TransGroupId(tGroupId),
                                                cParams.lSurfList.subspan(startDistance, elemCount));
            innerStart = innerEnd;
        }
        start = end;
    }
}

void BaseAccelerator::Construct(BaseAccelConstructParams p)
{
    static const auto annotation = gpuSystem.CreateAnnotation("Accelerator Construct");
    const auto _ = annotation.AnnotateScope();

    std::vector<AccelGroupConstructParams> partitions;
    PartitionSurfaces(partitions, p);
    // Add primitive-backed lights surfaces as well
    AddLightSurfacesToPartitions(partitions, p);

    // Generate the accelerators
    GPUQueueIteratorRoundRobin qIt(gpuSystem);
    for(auto&& partition : partitions)
    {
        using namespace TypeNameGen::Runtime;
        std::string accelTypeName = std::string(Name()) + std::string(partition.primGroup->Name());
        uint32_t aGroupId = idCounter++;
        auto accelGenerator = accelGenerators.at(accelTypeName);
        if(!accelGenerator)
        {
            throw MRayError("{:s}: Unable to find generator for accelerator group \"{:s}\"",
                            Name(), accelTypeName);
        }
        auto GenerateAccelGroup = accelGenerator.value().get();
        auto accelPtr = GenerateAccelGroup(std::move(aGroupId),
                                           threadPool, gpuSystem,
                                           *partition.primGroup,
                                           workGenGlobalMap);
        auto loc = generatedAccels.emplace(aGroupId, std::move(accelPtr));
        AcceleratorGroupI* acc = loc.first->second.get();
        acc->PreConstruct(this);
        acc->Construct(std::move(partition), qIt.Queue());
        qIt.Next();
    }
    // Find the leaf count
    std::vector<size_t> instanceOffsets(generatedAccels.size() + 1, 0);
    std::transform_inclusive_scan(generatedAccels.cbegin(),
                                  generatedAccels.cend(),
                                  instanceOffsets.begin() + 1, std::plus{},
    [](const auto& pair) -> size_t
    {
        return pair.second->InstanceCount();
    });
    //
    std::vector<uint32_t> keyOffsets(generatedAccels.size() + 1, 0);
    std::transform_inclusive_scan(generatedAccels.cbegin(),
                                  generatedAccels.cend(),
                                  keyOffsets.begin() + 1, std::plus{},
    [](const auto& pair) -> uint32_t
    {
        return pair.second->InstanceTypeCount();
    });
    // Set the offsets
    uint32_t i = 0;
    for(auto& group : generatedAccels)
    {
        AcceleratorGroupI* aGroup = group.second.get();
        aGroup->SetKeyOffset(keyOffsets[i]);
        for(uint32_t key = keyOffsets[i]; key < keyOffsets[i + 1]; key++)
            accelInstances.emplace(key, aGroup);

        i++;
    }
    // Find the maximum bits used on key
    uint32_t keyBatchPortionMax = keyOffsets.back();
    uint32_t keyIdPortionMax = std::transform_reduce(generatedAccels.cbegin(),
                                                     generatedAccels.cend(),
                                                     uint32_t(0),
    [](uint32_t rhs, uint32_t lhs)
    {
        return Math::Max(rhs, lhs);
    },
    [](const auto& pair)
    {
        return pair.second->UsedIdBitsInKey();
    });
    using namespace Bit;
    maxBitsUsedOnKey = Vector2ui(RequiredBitsToRepresent(keyBatchPortionMax),
                                 RequiredBitsToRepresent(keyIdPortionMax));
    // Validate
    if(maxBitsUsedOnKey[0] > AcceleratorKey::BatchBits ||
       maxBitsUsedOnKey[1] > AcceleratorKey::IdBits)
    {
        throw MRayError("[{}]: Too many bits on accelerator [{}|{}], AcceleratorKey can hold "
                        "[{}|{}] amount of bits",
                        Name(), maxBitsUsedOnKey[0], maxBitsUsedOnKey[1],
                        AcceleratorKey::BatchBits, AcceleratorKey::IdBits);
    }

    // Internal construction routine,
    // we can not fetch the leaf data here because some accelerators are
    // constructed on CPU (due to laziness)
    if(partitions.size() != 0)
        sceneAABB = InternalConstruct(instanceOffsets);
    else
        sceneAABB = AABB3::Zero();
}


