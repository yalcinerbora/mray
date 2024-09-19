#pragma once

template<class T>
using BatchListOptix = StaticVector<T, TracerConstants::MaxPrimBatchPerSurface>;

struct BuildInputPackAABB
{
    BatchListOptix<OptixBuildInput> buildInputs;
    BatchListOptix<uint32_t>        geometryFlags;
    BatchListOptix<CUdeviceptr>     aabbPointers;
};

struct BuildInputPackTriangle
{
    BatchListOptix<OptixBuildInput> buildInputs;
    BatchListOptix<uint32_t>        geometryFlags;
    BatchListOptix<CUdeviceptr>     vertexPointers;
};

inline
BuildInputPackAABB GenBuildInputsAABB(const Span<const AABB3>& aabbs,
                                      const PrimRangeArray& primRanges)
{

    BuildInputPackAABB result;
    for(const auto& range : primRanges)
    {
        if(range[0] == std::numeric_limits<uint32_t>::max() &&
           range[1] == std::numeric_limits<uint32_t>::max())
            break;

        auto localAABBSpan = aabbs.subspan(range[0], range[1] - range[0]);
        result.aabbPointers.push_back(std::bit_cast<CUdeviceptr>(localAABBSpan.data()));
        unsigned int flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        result.geometryFlags.push_back(flags);
        result.buildInputs.emplace_back
        (
            OptixBuildInput
            {
                .type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
                .customPrimitiveArray = OptixBuildInputCustomPrimitiveArray
                {
                    .aabbBuffers = &result.aabbPointers.back(),
                    .numPrimitives = static_cast<uint32_t>(localAABBSpan.size()),
                    .strideInBytes = 0,
                    .flags = &result.geometryFlags.back(),
                    // SBT
                    .numSbtRecords = 1,
                    .sbtIndexOffsetBuffer = 0,
                    .sbtIndexOffsetSizeInBytes = 0,
                    .sbtIndexOffsetStrideInBytes = 0,
                    // We handle this in software
                    .primitiveIndexOffset = 0
                }
            }
        );
    }
    return result;
}

inline
BuildInputPackTriangle GenBuildInputsTriangle(const Span<const Vector3>& vertices,
                                              const Span<const Vector3ui>& indices,
                                              const PrimRangeArray& primRanges)
{
    BuildInputPackTriangle result;
    for(const auto& range : primRanges)
    {
        if(range[0] == std::numeric_limits<uint32_t>::max() &&
           range[1] == std::numeric_limits<uint32_t>::max())
            break;

        auto rangeIndices = indices.subspan(range[0], range[1] - range[0]);

        result.vertexPointers.push_back(std::bit_cast<CUdeviceptr>(vertices.data()));
        unsigned int flags = (OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT |
                              OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING);
        result.geometryFlags.push_back(flags);

        result.buildInputs.emplace_back
        (
            OptixBuildInput
            {
                .type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
                .triangleArray =
                {
                    .vertexBuffers = &result.vertexPointers.back(),
                    .numVertices = static_cast<uint32_t>(vertices.size()),
                    .vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
                    .vertexStrideInBytes = sizeof(Vector3),
                    .indexBuffer = std::bit_cast<CUdeviceptr>(rangeIndices.data()),
                    .numIndexTriplets = static_cast<uint32_t>(rangeIndices.size()),
                    .indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
                    .indexStrideInBytes = sizeof(Vector3ui),
                    .preTransform = CUdeviceptr(0),
                    .flags = &result.geometryFlags.back(),
                    .numSbtRecords = 1,
                    .sbtIndexOffsetBuffer = 0,
                    .sbtIndexOffsetSizeInBytes = 0,
                    .sbtIndexOffsetStrideInBytes = 0,
                    .primitiveIndexOffset = 0,
                    .transformFormat = OPTIX_TRANSFORM_FORMAT_NONE,
                }
            }
        );
    }
    return result;
}

inline void FixReferencesInInputs(BuildInputPackTriangle& pack)
{
    assert(pack.buildInputs.size() == pack.geometryFlags.size());
    assert(pack.geometryFlags.size() == pack.vertexPointers.size());
    for(size_t i = 0; i < pack.buildInputs.size(); i++)
    {
        pack.buildInputs[i].triangleArray.vertexBuffers =
            &pack.vertexPointers[i];
        pack.buildInputs[i].triangleArray.flags =
            &pack.geometryFlags[i];
    }
}

inline void FixReferencesInInputs(BuildInputPackAABB& pack)
{
    assert(pack.buildInputs.size() == pack.geometryFlags.size());
    assert(pack.geometryFlags.size() == pack.aabbPointers.size());
    for(size_t i = 0; i < pack.buildInputs.size(); i++)
    {
        pack.buildInputs[i].customPrimitiveArray.aabbBuffers=
            &pack.aabbPointers[i];
        pack.buildInputs[i].triangleArray.flags =
            &pack.geometryFlags[i];
    }
}

template<PrimitiveGroupC PG>
std::string_view AcceleratorGroupOptiX<PG>::TypeName()
{
    using namespace TypeNameGen::CompTime;
    static const auto Name = AccelGroupTypeName(BaseAcceleratorOptiX::TypeName(),
                                                PG::TypeName());
    return Name;
}

template<PrimitiveGroupC PG>
AcceleratorGroupOptiX<PG>::AcceleratorGroupOptiX(uint32_t accelGroupId,
                                                 BS::thread_pool& tp,
                                                 const GPUSystem& sys,
                                                 const GenericGroupPrimitiveT& pg,
                                                 const AccelWorkGenMap& wMap)
    : Base(accelGroupId, tp, sys, pg, wMap)
    , memory({this->gpuSystem.AllGPUs()}, 32_MiB, 64_MiB)
{}

template<PrimitiveGroupC PG>
void AcceleratorGroupOptiX<PG>::PreConstruct(const BaseAcceleratorI* a)
{
    const auto* baseAccel = static_cast<const BaseAcceleratorOptiX*>(a);
    contextOptiX = baseAccel->GetOptixDeviceHandle();
}

template<PrimitiveGroupC PG>
void AcceleratorGroupOptiX<PG>::MultiBuildTriangleCLT(const PreprocessResult& ppResult,
                                                      const GPUQueue& queue)
{
    using mray::cuda::ToHandleCUDA;
    size_t totalLeafCount = this->concreteLeafRanges.back()[1];

    // Do the build inputs
    std::vector<BuildInputPackTriangle> allBuildInputs;
    allBuildInputs.reserve(ppResult.concretePrimRanges.size());

    // Thankfully these are persistent, directly acquire from
    // primitive group
    Span<const Vector3> verts = this->pg.GetVertexPositionSpan();
    Span<const Vector3ui> indices = this->pg.GetIndexSpan();
    for(const auto& primRanges : ppResult.concretePrimRanges)
        allBuildInputs.emplace_back(GenBuildInputsTriangle(verts, indices, primRanges));
    // Fix the references
    std::for_each(allBuildInputs.begin(), allBuildInputs.end(),
    [](BuildInputPackTriangle& pack)
    {
        FixReferencesInInputs(pack);
    });

    // Here we will do "double dip" to find the compacted size of each accelerator,
    // We run build command once to get the compact sizes, and rerun with the addition
    // of compactation operation. With this we can allocate a single large memory for
    // all accelerators in the group.
    //
    // Find the input size for temp allocation
    OptixAccelBufferSizes tempBufferSize = std::transform_reduce
    (
        allBuildInputs.begin(), allBuildInputs.end(), OptixAccelBufferSizes{},
        // Reduction Operation
        [](const OptixAccelBufferSizes& a, const OptixAccelBufferSizes& b) -> OptixAccelBufferSizes
        {
            return OptixAccelBufferSizes
            {
                .outputSizeInBytes = std::max(a.outputSizeInBytes, b.outputSizeInBytes),
                .tempSizeInBytes = std::max(a.tempSizeInBytes, b.tempSizeInBytes),
                .tempUpdateSizeInBytes = std::max(a.tempUpdateSizeInBytes, b.tempUpdateSizeInBytes),
            };
        },
        // Transform Operation
        [this](const BuildInputPackTriangle& pack) -> OptixAccelBufferSizes
        {
            OptixAccelBufferSizes sizeList = {};
            OPTIX_CHECK(optixAccelComputeMemoryUsage(contextOptiX, &OptiXAccelDetail::BUILD_OPTIONS_OPTIX,
                                                     pack.buildInputs.data(),
                                                     static_cast<uint32_t>(pack.buildInputs.size()),
                                                     &sizeList));
            return sizeList;
        }
    );

    // Allocate the temp buffer
    size_t totalSize = (Math::NextMultiple(tempBufferSize.outputSizeInBytes, MemAlloc::DefaultSystemAlignment()) +
                        Math::NextMultiple(tempBufferSize.tempSizeInBytes, MemAlloc::DefaultSystemAlignment()));
    DeviceMemory tempMem({queue.Device()}, totalSize, totalSize << 1);
    Span<Byte> dAcceleratorMem;
    Span<Byte> dBuildTempMem;
    Span<uint64_t> dCompactSizes;
    // Might as well allocate these for prim key generation
    Span<Vector2ui> dConcreteLeafRanges;
    Span<PrimRangeArray> dConcretePrimRanges;
    MemAlloc::AllocateMultiData(std::tie(dAcceleratorMem, dBuildTempMem,
                                         dCompactSizes, dConcreteLeafRanges,
                                         dConcretePrimRanges),
                                tempMem,
                                {tempBufferSize.outputSizeInBytes,
                                 tempBufferSize.tempSizeInBytes,
                                 allBuildInputs.size(),
                                 this->concreteLeafRanges.size(),
                                 ppResult.concretePrimRanges.size()});

    // Build all accelerators, and overwrite to a single memory location
    // Find out the compacted memory sizes
    static_assert(MemAlloc::DefaultSystemAlignment() >= OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT,
                  "MRay and OptiX default alignments are not compatible");

    OptixTraversableHandle phonyHandle;
    for(size_t i = 0; i < allBuildInputs.size(); i++)
    {
        const auto& pack = allBuildInputs[i];

        OptixAccelEmitDesc emitDesc =
        {
            .result = std::bit_cast<CUdeviceptr>(dCompactSizes.data() + i),
            .type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE
        };
        OPTIX_CHECK(optixAccelBuild(contextOptiX, ToHandleCUDA(queue),
                                    &OptiXAccelDetail::BUILD_OPTIONS_OPTIX,
                                    pack.buildInputs.data(),
                                    static_cast<uint32_t>(pack.buildInputs.size()),
                                    std::bit_cast<CUdeviceptr>(dBuildTempMem.data()),
                                    dBuildTempMem.size(),
                                    std::bit_cast<CUdeviceptr>(dAcceleratorMem.data()),
                                    dAcceleratorMem.size(),
                                    &phonyHandle, &emitDesc, 1u));
    }
    std::vector<uint64_t> hCompactedSizes(dCompactSizes.size());
    queue.MemcpyAsync(Span<uint64_t>(hCompactedSizes), ToConstSpan(dCompactSizes));
    queue.Barrier().Wait();

    // Find the memory ranges
    std::vector<uint64_t> hCompactedOffsets(hCompactedSizes.size() + 1, 0u);
    std::transform(hCompactedSizes.cbegin(), hCompactedSizes.cend(),
                   hCompactedOffsets.begin() + 1,
    [](size_t in) -> size_t
    {
        return Math::NextMultiple(in, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
    });
    std::inclusive_scan(hCompactedOffsets.cbegin() + 1, hCompactedOffsets.cend(),
                        hCompactedOffsets.begin() + 1);

    // We can't allocate all these via "Allocate Multi Data"
    // So allocate in bulk
    hTransformSoAOffsets = std::vector<size_t>(this->workInstances.size() + 1, 0);
    std::transform_inclusive_scan
    (
        this->workInstances.cbegin(), this->workInstances.cend(),
        hTransformSoAOffsets.begin() + 1, std::plus{},
        [](const auto& work)
        {
            return Math::NextMultiple(work.second->TransformSoAByteSize(),
                                      MemAlloc::DefaultSystemAlignment());;
        }
    );
    MemAlloc::AllocateMultiData(std::tie(dAllAccelerators, dAllLeafs,
                                         dPrimGroupSoA, dTransformGroupSoAList),
                                memory,
                                {hCompactedOffsets.back(), totalLeafCount,
                                 1, hTransformSoAOffsets.back()});

    accelHandels.resize(allBuildInputs.size());
    // Now do this all over again
    for(size_t i = 0; i < allBuildInputs.size(); i++)
    {
        const auto& pack = allBuildInputs[i];
        OptixTraversableHandle handle;
        // Due to queue logic we can overwrite on to the same memory for each accel
        OPTIX_CHECK(optixAccelBuild(contextOptiX, ToHandleCUDA(queue),
                                    &OptiXAccelDetail::BUILD_OPTIONS_OPTIX,
                                    pack.buildInputs.data(),
                                    static_cast<uint32_t>(pack.buildInputs.size()),
                                    std::bit_cast<CUdeviceptr>(dBuildTempMem.data()),
                                    dBuildTempMem.size(),
                                    std::bit_cast<CUdeviceptr>(dAcceleratorMem.data()),
                                    dAcceleratorMem.size(),
                                    &handle, nullptr, 0u));


        auto curAccelMem = dAllAccelerators.subspan(hCompactedOffsets[i],
                                                    hCompactedOffsets[i + 1] - hCompactedOffsets[i]);
        OPTIX_CHECK(optixAccelCompact(contextOptiX, ToHandleCUDA(queue), handle,
                                      std::bit_cast<CUdeviceptr>(curAccelMem.data()),
                                      curAccelMem.size(),
                                      &accelHandels[i]));
    }

    // Generate the leaf keys etc.
    assert(ppResult.concretePrimRanges.size() == this->concreteLeafRanges.size());
    // Copy Ids to the leaf buffer
    auto hConcreteLeafRanges = Span<const Vector2ui>(this->concreteLeafRanges);
    auto hConcretePrimRanges = Span<const PrimRangeArray>(ppResult.concretePrimRanges);
    queue.MemcpyAsync(dConcreteLeafRanges, hConcreteLeafRanges);
    queue.MemcpyAsync(dConcretePrimRanges, hConcretePrimRanges);
    // Dedicate a block for each
    // concrete accelerator for copy
    uint32_t blockCount = queue.RecommendedBlockCountDevice(KCGeneratePrimitiveKeys,
                                                            StaticThreadPerBlock1D(),
                                                            0);
    using namespace std::string_literals;
    static const auto KernelName = "KCGeneratePrimitiveKeys-"s + std::string(TypeName());
    queue.IssueExactKernel<KCGeneratePrimitiveKeys>
    (
        KernelName,
        KernelExactIssueParams
        {
            .gridSize = blockCount,
            .blockSize = StaticThreadPerBlock1D()
        },
        // Output
        dAllLeafs,
        // Input
        dConcretePrimRanges,
        dConcreteLeafRanges,
        // Constant
        this->pg.GroupId()
    );

    // Finally calculate the hit record offsets and hit record prim ranges
    constexpr Vector2ui INVALID_BATCH = Vector2ui(std::numeric_limits<uint32_t>::max());
    size_t conservativeRecordCount = (ppResult.concretePrimRanges.size() *
                                      TracerConstants::MaxPrimBatchPerSurface);
    hConcreteHitRecordOffsets.reserve(ppResult.concretePrimRanges.size());
    hConcreteHitRecordPrimRanges.reserve(conservativeRecordCount);
    size_t recordOffset = 0;
    size_t primOffset = 0;
    for(const auto& primRanges : ppResult.concretePrimRanges)
    {
        size_t innerCount = 0;
        for(const auto& range : primRanges)
        {
            if(range == INVALID_BATCH) break;
            innerCount++;

            uint32_t primCount = range[1] - range[0];
            hConcreteHitRecordPrimRanges.emplace_back(primOffset,
                                                      primOffset + primCount);
            primOffset += primCount;
        }

        hConcreteHitRecordOffsets.emplace_back(recordOffset);
        recordOffset += innerCount;
    }
    hConcreteHitRecordOffsets.push_back(recordOffset);
}

template<PrimitiveGroupC PG>
void AcceleratorGroupOptiX<PG>::MultiBuildTrianglePPT(const PreprocessResult&,
                                                      const GPUQueue&)
{
    throw MRayError("NotImplemented!");
}

template<PrimitiveGroupC PG>
void AcceleratorGroupOptiX<PG>::MultiBuildGenericCLT(const PreprocessResult&,
                                                     const GPUQueue&)
{
    throw MRayError("NotImplemented!");
}

template<PrimitiveGroupC PG>
void AcceleratorGroupOptiX<PG>::MultiBuildGenericPPT(const PreprocessResult&,
                                                     const GPUQueue&)
{
    throw MRayError("NotImplemented!");
}

template<PrimitiveGroupC PG>
void AcceleratorGroupOptiX<PG>::Construct(AccelGroupConstructParams p,
                                          const GPUQueue& queue)
{
    static constexpr bool PER_PRIM_TRANSFORM = TransformLogic == PrimTransformType::PER_PRIMITIVE_TRANSFORM;
    const PreprocessResult& ppResult = this->PreprocessConstructionParams(p);
    // Actual construction
    if constexpr(IsTriangle && !PER_PRIM_TRANSFORM)
    {
        MultiBuildTriangleCLT(ppResult, queue);
    }
    else if constexpr(IsTriangle && PER_PRIM_TRANSFORM)
    {
        MultiBuildTrianglePPT(ppResult, queue);
    }
    else if constexpr(!IsTriangle && !PER_PRIM_TRANSFORM)
    {
        MultiBuildGenericCLT(ppResult, queue);
    }
    else if constexpr(!IsTriangle && !PER_PRIM_TRANSFORM)
    {
        MultiBuildGenericPPT(ppResult, queue);
    }
    else static_assert(!IsTriangle && !PER_PRIM_TRANSFORM,
                       "Unknown params on OptiX build");

    // Calculate hit records for each build input
    hHitRecords.reserve(this->InstanceCount() * TracerConstants::MaxPrimBatchPerSurface);
    for(size_t i = 0; i < this->InstanceCount(); i++)
    {
        uint32_t concreteIndex = this->concreteIndicesOfInstances[i];
        Vector2ui hrRange = Vector2ui(hConcreteHitRecordOffsets[concreteIndex],
                                      hConcreteHitRecordOffsets[concreteIndex + 1]);
        size_t recordCount = (hrRange[1] - hrRange[0]);

        // Find the actual range
        auto workOffset = std::upper_bound(this->workInstanceOffsets.cbegin(),
                                           this->workInstanceOffsets.cend(), i);
        assert(workOffset != this->workInstanceOffsets.cend());
        size_t transformStart = std::distance(this->workInstanceOffsets.cbegin(), workOffset - 1);
        size_t transformByteStart = transformStart * MemAlloc::DefaultSystemAlignment();
        const Byte* dTransSoAPtr = dTransformGroupSoAList.subspan(transformByteStart,
                                                                  MemAlloc::DefaultSystemAlignment()).data();
        MRAY_LOG("{}", i);
        for(size_t j = 0; j < recordCount; j++)
        {
            Vector2ui primRange = hConcreteHitRecordPrimRanges[hrRange[0] + j];
            GenericHitRecordData<> recordData =
            {
                .dPrimKeys = dAllLeafs.subspan(primRange[0], primRange[1] - primRange[0]),
                .transformKey = ppResult.surfData.transformKeys[i],
                .alphaMap = ppResult.surfData.alphaMaps[i][j],
                .lightOrMatKey = ppResult.surfData.lightOrMatKeys[i][j],
                // We are not responsible for this base accelerator will set it
                .acceleratorKey = AcceleratorKey::InvalidKey(),
                .primSoA = dPrimGroupSoA.data(),
                .transSoA = dTransSoAPtr
            };
            GenericHitRecord<> record = {};
            record.data = recordData;
            hHitRecords.push_back(record);
        }
    }
}

template<PrimitiveGroupC PG>
void AcceleratorGroupOptiX<PG>::WriteInstanceKeysAndAABBs(Span<AABB3> dAABBWriteRegion,
                                                          Span<AcceleratorKey> dKeyWriteRegion,
                                                          const GPUQueue& queue) const
{
    //this->WriteInstanceKeysAndAABBsInternal(dAABBWriteRegion,
    //                                        dKeyWriteRegion,
    //                                        dAllLeafs,
    //                                        dTransformKeys,
    //                                        queue);
}

template<PrimitiveGroupC PG>
void AcceleratorGroupOptiX<PG>::AcquireIASConstructionParams(Span<OptixTraversableHandle> dTraversableHandles,
                                                             Span<Matrix4x4> dInstanceMatrices,
                                                             Span<uint32_t> dSBTCounts,
                                                             Span<uint32_t> dFlags) const
{

}

template<PrimitiveGroupC PG>
std::vector<Vector2ui> AcceleratorGroupOptiX<PG>::GetRecordOffsets()
{
    // We have concrete offsets convert it to instance offsets
    std::vector<Vector2ui> result(this->InstanceCount());
    uint32_t offset = 0;
    for(uint32_t i = 0; i < this->InstanceCount(); i++)
    {
        uint32_t cIndex = this->concreteIndicesOfInstances[i];
        size_t recordCount = (hConcreteHitRecordOffsets[cIndex + 1] -
                              hConcreteHitRecordOffsets[cIndex]);
        result.emplace_back(offset, recordCount);
        offset += recordCount;
    }
    return result;
}

template<PrimitiveGroupC PG>
void AcceleratorGroupOptiX<PG>::WriteRecords(Span<GenericHitRecord<>> dRecordWriteRegion,
                                             const GPUQueue& queue) const
{
    queue.MemcpyAsync(dRecordWriteRegion, Span<GenericHitRecord<>>(hHitRecords));
}

template<PrimitiveGroupC PG>
void AcceleratorGroupOptiX<PG>::CastLocalRays(// Output
                                              Span<HitKeyPack> dHitIds,
                                              Span<MetaHit> dHitParams,
                                              // I-O
                                              Span<BackupRNGState> dRNGStates,
                                              Span<RayGMem> dRays,
                                              // Input
                                              Span<const RayIndex> dRayIndices,
                                              Span<const CommonKey> dAccelKeys,
                                              // Constants
                                              uint32_t workId,
                                              const GPUQueue& queue)
{

}

template<PrimitiveGroupC PG>
typename AcceleratorGroupOptiX<PG>::DataSoA
AcceleratorGroupOptiX<PG>::SoA() const
{
    return EmptyType{};
}

template<PrimitiveGroupC PG>
size_t AcceleratorGroupOptiX<PG>::GPUMemoryUsage() const
{
    return 0;
}