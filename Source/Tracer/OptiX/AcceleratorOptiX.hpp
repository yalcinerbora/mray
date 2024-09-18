#pragma once

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
{}

template<class T>
using BatchListOptix = StaticVector<T, TracerConstants::MaxPrimBatchPerSurface>;
using GeomFlagListOptix = StaticVector<OptixBuildInput, TracerConstants::MaxPrimBatchPerSurface>;

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
    BatchListOptix<CUdeviceptr>     indexPointers;
};

inline
BuildInputPackTriangle GenBuildInputsTriangle(const PreprocessResult& p, uint32_t index,
                                              Span<const AABB3> dAllAABBs,
                                              OptixDeviceContext context)
{
    return BuildInputPackTriangle{};
}

inline
BuildInputPackAABB GenBuildInputsAABB(const PreprocessResult& p, uint32_t index,
                                      Span<const AABB3> dAllAABBs,
                                      OptixDeviceContext context)
{
    //const OptixAccelBuildOptions buildOpts =
    //{
    //    .buildFlags = (OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
    //                   OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_DISABLE_OPACITY_MICROMAPS |
    //                   OptixBuildFlags::OPTIX_BUILD_FLAG_PREFER_FAST_TRACE),
    //    .operation = OPTIX_BUILD_OPERATION_BUILD,
    //    .motionOptions = OptixMotionOptions{ 1 }
    //};

    BuildInputPackAABB result;
    const auto& surfData = p.surfData;
    size_t batchCount = surfData.instancePrimBatches.size();
    for(size_t i = 0; i < batchCount; i++)
    {
        unsigned int geomFlags = OPTIX_GEOMETRY_FLAG_NONE;
        // Should we enable any hit?
        if(!surfData.alphaMaps[index][i])
            geomFlags |= OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        // Should we disable cull face?
        if(surfData.cullFaceFlags[index][i])
            geomFlags |= OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
        result.geometryFlags.push_back(geomFlags);

        // Find the AABB range
        Vector2ui primRange = surfData.primRanges[index][i];
        CUdeviceptr ptr;


        //p.concretePrimRanges.
        OptixBuildInput buildInput =
        {
            .type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
            .customPrimitiveArray = OptixBuildInputCustomPrimitiveArray
            {
                .aabbBuffers    = &ptr,
                .numPrimitives  = primRange[1] - primRange[0],
                .strideInBytes  = 0,
                .flags          = &result.geometryFlags.back(),
                // SBT
                .numSbtRecords                  = 1,
                .sbtIndexOffsetBuffer           = 0,
                .sbtIndexOffsetSizeInBytes      = 0,
                .sbtIndexOffsetStrideInBytes    = 0,
                // We handle this in software
                .primitiveIndexOffset = 0
            }
        };

    }

    return result;


    //std::array<OptixBuildInput, 2> buildInput =
    //{
    //    OptixBuildInput
    //    {
    //        .type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
    //        .triangleArray = OptixBuildInputTriangleArray
    //        {
    //            .vertexBuffers = ...,
    //            .numVertices = 1,
    //            .vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
    //            .vertexStrideInBytes = 0,
    //            .indexBuffer = ...,
    //            .numIndexTriplets = 1,
    //            .indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
    //            .indexStrideInBytes = 0,
    //            .preTransform = 0,
    //            .flags = OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING,
    //            // SBT
    //            .numSbtRecords = 1,
    //            .sbtIndexOffsetBuffer = 0,
    //            .sbtIndexOffsetSizeInBytes = 0,
    //            .sbtIndexOffsetStrideInBytes = 0,
    //            // We handle this in software
    //            .primitiveIndexOffset = 0,
    //            //
    //            .transformFormat = OPTIX_TRANSFORM_FORMAT_NONE,
    //            .opacityMicromap = {},
    //            .displacementMicromap = {}
    //        }
    //    },

    //};

}

template<PrimitiveGroupC PG>
void AcceleratorGroupOptiX<PG>::Construct(AccelGroupConstructParams p,
                                          const GPUQueue& queue)
{
    using mray::cuda::ToHandleCUDA;
    PreprocessResult ppResult = this->PreprocessConstructionParams(p);

    //size_t doubleDipBufferSize;
    //// If we are
    //if constexpr(PG::TransformLogic != PrimTransformType::LOCALLY_CONSTANT_TRANSFORM)
    //{
    //    for(Vector2ui range : concreteLeafRanges)
    //    {

    //    }

    //    OPTIX_CHECK(optixAccelComputeMemoryUsage());
    //}

    //// Concrete indices of instances
    ////






























    ////
    //OptixDeviceContext context;

    //const OptixAccelBuildOptions buildOpts =
    //{
    //    .buildFlags = (OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
    //                   OptixBuildFlags::OPTIX_BUILD_FLAG_ALLOW_DISABLE_OPACITY_MICROMAPS |
    //                   OptixBuildFlags::OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
    //    .operation = OPTIX_BUILD_OPERATION_BUILD,
    //    .motionOptions = OptixMotionOptions{ 1 }
    //};
    //Span<Byte> dTempBuffer;
    //Span<Byte> dOutputBuffer;
    //Span<uint64_t> dCompactSize;
    //OptixTraversableHandle* traverseHandle;
    //OptixAccelEmitDesc emitDesc =
    //{
    //    .result = std::bit_cast<Cudeviceptr>(dCompactSize.data()),
    //    .type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE
    //};

    //std::array<OptixBuildInput, 2> buildInput =
    //{
    //    OptixBuildInput
    //    {
    //        .type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
    //        .triangleArray = OptixBuildInputTriangleArray
    //        {
    //            .vertexBuffers = ...,
    //            .numVertices = 1,
    //            .vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
    //            .vertexStrideInBytes = 0,
    //            .indexBuffer = ...,
    //            .numIndexTriplets = 1,
    //            .indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
    //            .indexStrideInBytes = 0,
    //            .preTransform = 0,
    //            .flags = OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING,
    //            // SBT
    //            .numSbtRecords = 1,
    //            .sbtIndexOffsetBuffer = 0,
    //            .sbtIndexOffsetSizeInBytes = 0,
    //            .sbtIndexOffsetStrideInBytes = 0,
    //            // We handle this in software
    //            .primitiveIndexOffset = 0,
    //            //
    //            .transformFormat = OPTIX_TRANSFORM_FORMAT_NONE,
    //            .opacityMicromap = {},
    //            .displacementMicromap = {}
    //        }
    //    }
    //    OptixBuildInput
    //    {
    //        .type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
    //        .customPrimitiveArray = OptixBuildInputCustomPrimitiveArray
    //        {
    //            .aabbBuffers = ,
    //            .numPrimitives =,
    //            .strideInBytes = 0,
    //            .flags = OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING,
    //            // SBT
    //            .numSbtRecords = 1,
    //            .sbtIndexOffsetBuffer = 0,
    //            .sbtIndexOffsetSizeInBytes = 0,
    //            .sbtIndexOffsetStrideInBytes = 0,
    //            // We handle this in software
    //            .primitiveIndexOffset = 0
    //        }
    //    }
    //};

    //optixAccelBuild(context, ToHandleCUDA(queue), &buildOpts,
    //                buildInput.data(), buildInput.size(),
    //                dTempBuffer.data(), dTempBuffer.size(),
    //                dOutputBuffer.data(), dOutputBuffer.size(),
    //                &taverseHandle, numMitte);
}

template<PrimitiveGroupC PG>
void AcceleratorGroupOptiX<PG>::WriteInstanceKeysAndAABBs(Span<AABB3> dAABBWriteRegion,
                                                          Span<AcceleratorKey> dKeyWriteRegion,
                                                          const GPUQueue&) const
{

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