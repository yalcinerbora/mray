#pragma once

#include "AcceleratorEmbree.h"

template<PrimitiveGroupC PG, TransformGroupC TG = TransformGroupIdentity,
         auto GenerateTransformContext = MRAY_PRIM_TGEN_FUNCTION(PG, TG)>
void BoundsFuncEmbree(const RTCBoundsFunctionArguments* args)
{
    // Generic system to somewhat understandable types
    // Transform context is the primitive/transform group handshake
    // that generates a transform that can be applied.
    // For example bone-animated meshes will provide weight/weight indices
    // and transform will provide the matrices/dual quaternions w/e.
    using TransContext = typename PrimTransformContextType<PG, TG>::Result;
    // The primitive (if we continue the above example "primitive" is
    // triangle)
    using Prim = PG::template Primitive<TransContext>;
    // Hit record is the compiled data for embree which provides
    // SoA data of the primitive and transform groups,
    // keys etc.
    using HitRecord = EmbreeHitRecord<typename PG::DataSoA, typename TG::DataSoA>;
    //
    uint32_t primIndex = args->primID;
    // To access primitive data we need to get the index here as well
    const auto& geomData = *reinterpret_cast<const EmbreeGeomUserData*>(args->geometryUserPtr);
    const auto& recordIndex = geomData.recordIndexForBounds;
    const auto& geomGlobalData = *geomData.geomGlobalData;
    const HitRecord& record = reinterpret_cast<const HitRecord&>(geomGlobalData.hAllHitRecords[recordIndex]);
    PrimitiveKey pKey = record.dPrimKeys[primIndex];
    TransformKey tKey = record.transformKey;
    const auto& tgData = *record.tgData;
    const auto& pgData = *record.pgData;

    // All of these types are compile-time decutible
    // thus they should be optimizes out by the compiler.
    // For example IdentityTransformContext is just series of identity
    // functions and there is no polymorphism here so it should
    // be as efficient as if these functions do not exist.
    TransContext tContext = GenerateTransformContext(tgData, pgData, tKey, pKey);
    // Finally we have the primitive
    Prim prim = Prim(tContext, pgData, pKey);

    // All these work to call this function :)
    AABB3 aabb = prim.GetAABB();

    // TODO: We do not have support for motion-blur yet
    // so change this later
    assert(args->timeStep == 0);

    args->bounds_o->lower_x = aabb.Min()[0];
    args->bounds_o->lower_y = aabb.Min()[1];
    args->bounds_o->lower_z = aabb.Min()[2];

    args->bounds_o->upper_x = aabb.Max()[0];
    args->bounds_o->upper_y = aabb.Max()[1];
    args->bounds_o->upper_z = aabb.Max()[2];
}

template<PrimitiveGroupC PG, TransformGroupC TG = TransformGroupIdentity,
         auto GenerateTransformContext = MRAY_PRIM_TGEN_FUNCTION(PG, TG)>
void IntersectFuncEmbree(const RTCIntersectFunctionNArguments* args)
{
    // Check the "BoundsFuncEmbree" for the explanation
    using TransContext = typename PrimTransformContextType<PG, TG>::Result;
    using Prim = PG::template Primitive<TransContext>;
    using HitRecord = EmbreeHitRecord<typename PG::DataSoA, typename TG::DataSoA>;
    using HitRecordG = EmbreeHitRecord<>;
    const auto& geomData = *reinterpret_cast<const EmbreeGeomUserData*>(args->geometryUserPtr);
    const auto& groupData = *geomData.geomGlobalData;
    auto& embreeContext = *reinterpret_cast<EmbreeRayQueryContext*>(args->context);

    auto Intersect = [&]<size_t N>()
    {
        // TODO: AFAIK, this data is the exact data determined by the intersect calls?
        // If it is not, we fail spectecularly (we may check it with ubsan later)
        assert(args->N == N);
        using RayHitBatch = RTCRayHitNt<N>;
        auto& rh = *reinterpret_cast<RayHitBatch*>(args->rayhit);

        RTCHitNt<N> potentialHits;
        std::array<int, N> isValid;
        std::fill_n(isValid.data(), N, EMBREE_INVALID_RAY);
        std::array<Float, N> newTs;
        bool someHasAlphaMaps = false;
        for(uint32_t i = 0; i < N; i++)
        {
            if(args->valid[i] == EMBREE_INVALID_RAY) continue;

            // We set the same geometry user data pointer to all geometries in the group.
            // We utilize instId/primId fields to locate the actual primitive.
            // If instID is invalid, we are doing local ray cast
            // acceleratorKey in payload must be valid
            uint32_t localInstanceId = (args->context->instID[0] == RTC_INVALID_GEOMETRY_ID)
                ? embreeContext.localAccelKeys[rh.ray.id[i]].FetchIndexPortion()
                : args->context->instID[0] - groupData.globalToLocalOffset;
            uint32_t geomIndex = groupData.hInstanceHitRecordOffsets[localInstanceId] + args->geomID;
            const HitRecordG& recordGeneric = groupData.hAllHitRecords[geomIndex];
            const HitRecord& record = reinterpret_cast<const HitRecord&>(recordGeneric);
            //
            uint32_t primIndex = args->primID;
            TransformKey tKey = record.transformKey;
            PrimitiveKey pKey = record.dPrimKeys[primIndex];
            const auto& tgData = *record.tgData;
            const auto& pgData = *record.pgData;
            //
            TransContext tContext = GenerateTransformContext(tgData, pgData, tKey, pKey);
            Prim prim = Prim(tContext, pgData, pKey);
            //
            Vector3 dir = Vector3(rh.ray.dir_x[i], rh.ray.dir_y[i], rh.ray.dir_z[i]);
            Vector3 pos = Vector3(rh.ray.org_x[i], rh.ray.org_y[i], rh.ray.org_z[i]);
            Ray r = Ray(dir, pos);

            auto hitResult = prim.Intersects(r, record.cullFace);
            if(hitResult && (hitResult->t < rh.ray.tfar[i]))
            {
                isValid[i] = EMBREE_VALID_RAY;
                potentialHits.u[i] = hitResult->hit[0];
                potentialHits.v[i] = hitResult->hit[1];
                potentialHits.geomID[i] = args->geomID;
                potentialHits.primID[i] = args->primID;
                potentialHits.instID[0][i] = args->context->instID[0];
                potentialHits.instPrimID[0][i] = args->context->instPrimID[0];
                newTs[i] = hitResult->t;

                someHasAlphaMaps |= record.alphaMap.has_value();
            }
        }
        // Invoke alpha map if any rays requires it
        if(someHasAlphaMaps)
        {
            RTCFilterFunctionNArguments filterArgs;
            filterArgs.context = args->context;
            filterArgs.geometryUserPtr = args->geometryUserPtr;
            filterArgs.hit = reinterpret_cast<RTCHitN*>(&potentialHits);
            filterArgs.ray = reinterpret_cast<RTCRayN*>(&rh.ray);
            filterArgs.N = N;
            filterArgs.valid = isValid.data();
            rtcInvokeIntersectFilterFromGeometry(args, &filterArgs);
        }
        // Conditionally write the data
        for(uint32_t i = 0; i < N; i++)
        {
            if(isValid[i] == EMBREE_INVALID_RAY) continue;

            rh.ray.tfar[i] = newTs[i];
            rh.hit.primID[i] = args->primID;
            rh.hit.geomID[i] = potentialHits.geomID[i];
            rh.hit.instID[0][i] = potentialHits.instID[0][i];
            rh.hit.instPrimID[0][i] = potentialHits.instPrimID[0][i];
            rh.hit.Ng_x[i] = potentialHits.Ng_x[i];
            rh.hit.Ng_y[i] = potentialHits.Ng_y[i];
            rh.hit.Ng_z[i] = potentialHits.Ng_z[i];
            rh.hit.primID[i] = potentialHits.primID[i];
            rh.hit.u[i] = potentialHits.u[i];
            rh.hit.v[i] = potentialHits.v[i];
        }
    };

    switch(args->N)
    {
        case 1  : Intersect.template operator()<1 >(); break;
        case 4  : Intersect.template operator()<4 >(); break;
        case 8  : Intersect.template operator()<8 >(); break;
        case 16 : Intersect.template operator()<16>(); break;
        default : throw MRayError("[Embree]: Unknown N {}", args->N);
    }
}

template<PrimitiveGroupC PG, TransformGroupC TG = TransformGroupIdentity,
         auto GenerateTransformContext = MRAY_PRIM_TGEN_FUNCTION(PG, TG)>
void OccludedFuncEmbree(const RTCOccludedFunctionNArguments* args)
{
    // Check the "BoundsFuncEmbree" for the explanation
    using TransContext = typename PrimTransformContextType<PG, TG>::Result;
    using Prim = PG::template Primitive<TransContext>;
    using HitRecord = EmbreeHitRecord<typename PG::DataSoA, typename TG::DataSoA>;
    using HitRecordG = EmbreeHitRecord<>;
    const auto& geomData = *reinterpret_cast<const EmbreeGeomUserData*>(args->geometryUserPtr);
    const auto& groupData = *geomData.geomGlobalData;
    auto& embreeContext = *reinterpret_cast<EmbreeRayQueryContext*>(args->context);

    // TODO: AFAIK, this data is the exact data determined by the intersect calls?
    // If it is not, we fail spectecularly (we may check it with ubsan later)
    auto Occluded = [&]<size_t N>()
    {
        assert(args->N == N);
        using RayBatch = RTCRayNt<N>;
        auto& r = *reinterpret_cast<RayBatch*>(args->ray);

        RTCHitNt<N> potentialHits;
        std::array<int, N> isValid;
        std::fill_n(isValid.begin(), N, EMBREE_INVALID_RAY);
        bool someHasAlphaMaps = false;
        for(uint32_t i = 0; i < N; i++)
        {
            if(args->valid[i] == EMBREE_INVALID_RAY) continue;

            // We set the same geometry user data pointer to all geometries in the group.
            // We utilize instId/primId fields to locate the actual primitive.
            // If instID is invalid, we are doing local ray cast
            // acceleratorKey in payload must be valid
            uint32_t localInstanceId = (args->context->instID[0] == RTC_INVALID_GEOMETRY_ID)
                ? embreeContext.localAccelKeys[r.id[i]].FetchIndexPortion()
                : args->context->instID[0] - groupData.globalToLocalOffset;
            uint32_t geomIndex = groupData.hInstanceHitRecordOffsets[localInstanceId] + args->geomID;
            const HitRecordG& recordGeneric = groupData.hAllHitRecords[geomIndex];
            const HitRecord& record = reinterpret_cast<const HitRecord&>(recordGeneric);
            //
            uint32_t primIndex = args->primID;
            TransformKey tKey = record.transformKey;
            PrimitiveKey pKey = record.dPrimKeys[primIndex];
            const auto& tgData = *record.tgData;
            const auto& pgData = *record.pgData;

            TransContext tContext = GenerateTransformContext(tgData, pgData, tKey, pKey);
            Prim prim = Prim(tContext, pgData, pKey);
            //
            Vector3 dir = Vector3(r.dir_x[i], r.dir_y[i], r.dir_z[i]);
            Vector3 pos = Vector3(r.org_x[i], r.org_y[i], r.org_z[i]);
            Ray ray = Ray(dir, pos);

            auto hitResult = prim.Intersects(ray, record.cullFace);
            if(hitResult)
            {
                isValid[i] = EMBREE_VALID_RAY;
                potentialHits.u[i] = hitResult->hit[0];
                potentialHits.v[i] = hitResult->hit[1];
                //
                potentialHits.geomID[i] = args->geomID;
                potentialHits.primID[i] = args->primID;
                potentialHits.instID[0][i] = args->context->instID[0];
                potentialHits.instPrimID[0][i] = args->context->instPrimID[0];

                someHasAlphaMaps |= record.alphaMap.has_value();
            }
        }

        if(someHasAlphaMaps)
        {
            RTCFilterFunctionNArguments filterArgs;
            filterArgs.context = args->context;
            filterArgs.geometryUserPtr = args->geometryUserPtr;
            filterArgs.hit = reinterpret_cast<RTCHitN*>(&potentialHits);
            filterArgs.ray = reinterpret_cast<RTCRayN*>(&r);
            filterArgs.N = N;
            filterArgs.valid = isValid.data();
            rtcInvokeOccludedFilterFromGeometry(args, &filterArgs);
        }

        for(uint32_t i = 0; i < N; i++)
        {
            if(isValid[i] == EMBREE_INVALID_RAY) continue;

            r.tfar[i] = EMBREE_IS_OCCLUDED_RAY;
        }
    };

    switch(args->N)
    {
        case 1  : Occluded.template operator()<1 >(); break;
        case 4  : Occluded.template operator()<4 >(); break;
        case 8  : Occluded.template operator()<8 >(); break;
        case 16 : Occluded.template operator()<16>(); break;
        default : throw MRayError("[Embree]: Unknown N {}", args->N);
    }
}

template<PrimitiveGroupC PG, TransformGroupC TG = TransformGroupIdentity,
         auto GenerateTransformContext = MRAY_PRIM_TGEN_FUNCTION(PG, TG)>
void FilterFuncEmbree(const RTCFilterFunctionNArguments* args)
{
    // Check the "BoundsFuncEmbree" for the explanation
    using TransContext = typename PrimTransformContextType<PG, TG>::Result;
    using Prim = PG::template Primitive<TransContext>;
    using HitRecord = EmbreeHitRecord<typename PG::DataSoA, typename TG::DataSoA>;
    using HitRecordG = EmbreeHitRecord<>;
    const auto& geomData = *reinterpret_cast<const EmbreeGeomUserData*>(args->geometryUserPtr);
    const auto& groupData = *geomData.geomGlobalData;
    auto& embreeContext = *reinterpret_cast<EmbreeRayQueryContext*>(args->context);
    //
    auto Filter = [&]<size_t N>()
    {
        using HitBatch = RTCHitNt<N>;
        using RayBatch = RTCRayNt<N>;
        const HitBatch& h = *reinterpret_cast<HitBatch*>(args->hit);
        const RayBatch& r = *reinterpret_cast<RayBatch*>(args->ray);
        assert(args->N == N);
        for(uint32_t i = 0; i < N; i++)
        {
            if(args->valid[i] == EMBREE_INVALID_RAY) continue;
            // We need to fetch these from the hit in this case
            uint32_t localInstanceId = (args->context->instID[0] == RTC_INVALID_GEOMETRY_ID)
                ? embreeContext.localAccelKeys[r.id[i]].FetchIndexPortion()
                : args->context->instID[0] - groupData.globalToLocalOffset;
            uint32_t geomIndex = groupData.hInstanceHitRecordOffsets[localInstanceId] + h.geomID[i];
            const HitRecordG& recordGeneric = groupData.hAllHitRecords[geomIndex];
            const HitRecord& record = reinterpret_cast<const HitRecord&>(recordGeneric);

            // Embree default triangle routine does not have
            // runtime-enabled backface culling parameter.
            // Since we set filter function for all geometries
            // just check it here.
            // For user-defined geometries our intersection
            // routine checked it already so only compile for triangles
            Vector2 baryCoords(h.u[i], h.v[i]);
            if constexpr(TrianglePrimGroupC<PG>)
            {
                if(record.cullFace)
                {
                    // We lost primitive data etc. unlike intersection function
                    // So do ghetto check via normal and ray dir
                    Vector3 d = Vector3(r.dir_x[i], r.dir_y[i], r.dir_z[i]);
                    Vector3 n = Vector3(h.Ng_x[i], h.Ng_y[i], h.Ng_z[i]);

                    args->valid[i] = (Math::Dot(n, d) < Float{0})
                            ? EMBREE_VALID_RAY
                            : EMBREE_INVALID_RAY;

                    if(args->valid[i] == EMBREE_INVALID_RAY) continue;
                }
                // We might as well switch to MRay bary coords
                baryCoords = EmbreeBaryToMRay(baryCoords);
            }
            // Actual transparency
            if(!record.alphaMap.has_value()) continue;
            //
            const auto& tgData = *record.tgData;
            const auto& pgData = *record.pgData;
            uint32_t primIndex = h.primID[i];
            TransformKey tKey = record.transformKey;
            PrimitiveKey pKey = record.dPrimKeys[primIndex];
            //
            const auto& aMap = *record.alphaMap;
            TransContext tContext = GenerateTransformContext(tgData, pgData, tKey, pKey);
            Prim prim = Prim(tContext, pgData, pKey);
            Vector2 uv = prim.SurfaceParametrization(baryCoords);
            Float xi = embreeContext.rng[r.id[i]].NextFloat();
            Float alpha = aMap(uv);
            // Stochastic alpha cull
            if(xi >= alpha) args->valid[i] = EMBREE_INVALID_RAY;
        }
    };

    switch(args->N)
    {
        case 1  : Filter.template operator()<1 >(); break;
        case 4  : Filter.template operator()<4 >(); break;
        case 8  : Filter.template operator()<8 >(); break;
        case 16 : Filter.template operator()<16>(); break;
        default : throw MRayError("[Embree]: Unknown N {}", args->N);
    }
}

template<AccelGroupC AG, TransformGroupC TG>
RTCBoundsFunction
AcceleratorWorkEmbree<AG, TG>::AABBGenFunction() const
{
    return BoundsFuncEmbree<AG, TG>;
}

template<AccelGroupC AG, TransformGroupC TG>
RTCFilterFunctionN
AcceleratorWorkEmbree<AG, TG>::FilterFunction() const
{
    return FilterFuncEmbree<AG, TG>;
}

template<AccelGroupC AG, TransformGroupC TG>
RTCIntersectFunctionN
AcceleratorWorkEmbree<AG, TG>::IntersectionFunction() const
{
    return IntersectFuncEmbree<AG, TG>;
}

template<AccelGroupC AG, TransformGroupC TG>
RTCOccludedFunctionN
AcceleratorWorkEmbree<AG, TG>::OccludedFunction() const
{
    return OccludedFuncEmbree<AG, TG>;
}

template<PrimitiveGroupC PG>
std::string_view AcceleratorGroupEmbree<PG>::TypeName()
{
    using namespace TypeNameGen::CompTime;
    static const auto Name = AccelGroupTypeName(BaseAcceleratorEmbree::TypeName(),
                                                PG::TypeName());
    return Name;
}

template<PrimitiveGroupC PG>
AcceleratorGroupEmbree<PG>::AcceleratorGroupEmbree(uint32_t accelGroupId,
                                                   ThreadPool& tp,
                                                   const GPUSystem& sys,
                                                   const GenericGroupPrimitiveT& pg,
                                                   const AccelWorkGenMap& wMap)
    : Base(accelGroupId, tp, sys, pg, wMap)
    , mem(sys.AllGPUs(), 2_MiB, 64_MiB, true)
{}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::MultiBuildViaTriangle_CLT(const PreprocessResult& ppResult,
                                                           const GPUQueue&)
{
    // For constant local transforms, we can create an accelerator
    // over the base mesh and transform the traced ray beforehand.
    //
    // We can directly set the geometry buffers here.
    //
    // Create a scene for each MRay surface
    for(uint32_t i = 0; i < ppResult.concretePrimRanges.size(); i++)
    {
        const PrimRangeArray& primRanges = ppResult.concretePrimRanges[i];
        RTCScene s = rtcNewScene(rtcDevice);
        hConcreteScenes[i] = s;
        for(uint32_t j = 0; j < primRanges.size(); j++)
        {
            static constexpr auto INVALID_PRIM_RANGE = Vector2ui(std::numeric_limits<uint32_t>::max());
            if(primRanges[j] == INVALID_PRIM_RANGE) break;
            //
            Vector2ui primRange = primRanges[j];
            uint32_t primCount = primRange[1] - primRange[0];
            const auto& pgTri = static_cast<const PG&>(this->pg);
            Span<const Vector3> verts = pgTri.GetVertexPositionSpan();
            Span<const Vector3ui> indices = pgTri.GetIndexSpan();
            RTCGeometry g = rtcNewGeometry(rtcDevice, RTC_GEOMETRY_TYPE_TRIANGLE);
            rtcSetSharedGeometryBuffer(g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                       verts.data(), 0u, sizeof(Vector3), verts.size());
            rtcSetSharedGeometryBuffer(g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                                       indices.data(), sizeof(Vector3ui) * primRange[0],
                                       sizeof(Vector3ui), primCount);
            // Triangle does not need a specific user data, just set to the first
            // one
            rtcSetGeometryUserData(g, &geomUserData[0]);
            rtcSetGeometryIntersectFilterFunction(g, FilterFuncEmbree<PG>);
            rtcSetGeometryOccludedFilterFunction(g, FilterFuncEmbree<PG>);

            rtcCommitGeometry(g);
            [[maybe_unused]]
            auto geomId = rtcAttachGeometry(s, g);
            assert(geomId == j);
            rtcReleaseGeometry(g);
        }
        rtcSetSceneBuildQuality(s, RTC_BUILD_QUALITY_HIGH);
        rtcSetSceneFlags(s, RTC_SCENE_FLAG_COMPACT);
        rtcCommitScene(s);
        // We explicitly do not release here, we will release the scene's
        // when we attach to the base accelerator
    }

    // Distribute the concrete accelerators to instances
    for(uint32_t i = 0; i < this->InstanceCount(); i++)
    {
        uint32_t cIndex = this->concreteIndicesOfInstances[i];
        hInstanceScenes[i] = hConcreteScenes[cIndex];
        rtcRetainScene(hInstanceScenes[i]);
    }
    // Now we can release
    for(RTCScene s : hConcreteScenes)
        rtcReleaseScene(s);
}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::MultiBuildViaUser_CLT(const PreprocessResult& ppResult,
                                                       const GPUQueue&)
{
    // For constant local transforms, we can create an accelerator
    // over the base mesh and transform the traced ray beforehand.

    // Bounds function is little bit of hipster, it has its own pointer
    // since it does not have a geomID field we need to pass a concrete
    // hit record which we don't have
    // We can give one of the instance hit records that corresponds to
    // this accelerator, but we need to find one.
    // This search is required only for non-triangle primitives,
    // so we search here
    std::vector<uint32_t> instanceIndicesOfConcreteAccels;
    instanceIndicesOfConcreteAccels.resize(ppResult.concretePrimRanges.size());
    for(uint32_t i = 0; i < this->InstanceCount(); i++)
        instanceIndicesOfConcreteAccels[this->concreteIndicesOfInstances[i]] = i;

    uint32_t concreteGeomPtrIndex = 0;
    for(uint32_t i = 0; i < ppResult.concretePrimRanges.size(); i++)
    {
        const PrimRangeArray& primRanges = ppResult.concretePrimRanges[i];
        RTCScene s = rtcNewScene(rtcDevice);
        hConcreteScenes[i] = s;
        uint32_t instanceIndex = instanceIndicesOfConcreteAccels[i];
        uint32_t geomStart = geomGlobalData.hInstanceHitRecordOffsets[instanceIndex];
        for(uint32_t j = 0; j < uint32_t(primRanges.size()); j++)
        {
            static constexpr auto INVALID_PRIM_RANGE = Vector2ui(std::numeric_limits<uint32_t>::max());
            if(primRanges[j] == INVALID_PRIM_RANGE) break;

            // Find a geom user data location
            auto& curGeomUserData = geomUserData[concreteGeomPtrIndex++];
            curGeomUserData.recordIndexForBounds = geomStart + j;
            //
            Vector2ui primRange = primRanges[j];
            uint32_t primCount = primRange[1] - primRange[0];
            RTCGeometry g = rtcNewGeometry(rtcDevice, RTC_GEOMETRY_TYPE_USER);
            rtcSetGeometryUserPrimitiveCount(g, primCount);
            rtcSetGeometryUserData(g, &curGeomUserData);
            rtcSetGeometryBoundsFunction(g, BoundsFuncEmbree<PG>, nullptr);
            rtcSetGeometryIntersectFunction(g, IntersectFuncEmbree<PG>);
            rtcSetGeometryOccludedFunction(g, OccludedFuncEmbree<PG>);
            rtcSetGeometryIntersectFilterFunction(g, FilterFuncEmbree<PG>);
            rtcSetGeometryOccludedFilterFunction(g, FilterFuncEmbree<PG>);

            rtcCommitGeometry(g);
            [[maybe_unused]]
            auto geomId = rtcAttachGeometry(s, g);
            assert(geomId == j);
            rtcReleaseGeometry(g);
        }
        rtcSetSceneBuildQuality(s, RTC_BUILD_QUALITY_HIGH);
        rtcSetSceneFlags(s, RTC_SCENE_FLAG_COMPACT);
        rtcCommitScene(s);
        // We explicitly do not release here, we will release the scene's
        // when we attach to the base accelerator
    }

    // Distribute the concrete accelerators to instances
    for(uint32_t i = 0; i < this->InstanceCount(); i++)
    {
        uint32_t cIndex = this->concreteIndicesOfInstances[i];
        hInstanceScenes[i] = hConcreteScenes[cIndex];
        rtcRetainScene(hInstanceScenes[i]);
    }
    // Now we can release
    for(RTCScene s : hConcreteScenes)
        rtcReleaseScene(s);
}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::MultiBuildViaTriangle_PPT(const PreprocessResult&,
                                                           const GPUQueue&)
{
    // For PPT, we can utilize similar approach of OptiX
    // (As time of this writing OptiX portion is not implemented yet)
    // OptiX retains triangle data inside the accelerator structure
    // We can allocate a buffer and store transformed triangles there
    // to minimize transform cost.
    throw MRayError("NotImplemented!");
}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::MultiBuildViaUser_PPT(const PreprocessResult&,
                                                       const GPUQueue&)
{
    throw MRayError("NotImplemented!");
}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::PreConstruct(const BaseAcceleratorI* a)
{
    // Get the device from the base accelerator
    const auto* baseAccel = static_cast<const BaseAcceleratorEmbree*>(a);
    rtcDevice = baseAccel->GetRTCDeviceHandle();
}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::Construct(AccelGroupConstructParams p,
                                           const GPUQueue& queue)
{
    static constexpr bool PER_PRIM_TRANSFORM = TransformLogic == PrimTransformType::PER_PRIMITIVE_TRANSFORM;
    const PreprocessResult& ppResult = this->PreprocessConstructionParams(p);
    // Allocate buffers
    // Find Transform SoA offsets
    auto transformSoAOffsets = std::vector<size_t>(this->workInstances.size() + 1, 0);
    std::transform_inclusive_scan
    (
        this->workInstances.cbegin(), this->workInstances.cend(),
        transformSoAOffsets.begin() + 1, std::plus{},
        [](const auto& work)
        {
            return Math::NextMultiple(work.second->TransformSoAByteSize(),
                                      MemAlloc::DefaultSystemAlignment());
        }
    );
    transformSoAOffsets[0] = 0;
    // Calculate total subgeometry count and offsets
    // for subspan generation
    std::vector<uint32_t> hitRecordOffsets(this->InstanceCount() + 1);
    std::transform_inclusive_scan
    (
        ppResult.surfData.primRanges.begin(),
        ppResult.surfData.primRanges.end(),
        hitRecordOffsets.begin() + 1,
        std::plus{},
        [](const PrimRangeArray& primRanges)
        {
            constexpr Vector2ui INVALID_RANGE(std::numeric_limits<uint32_t>::max());
            uint32_t i = 0;
            while((i < primRanges.size()) && (primRanges[i] != INVALID_RANGE)) ++i;
            return i;
        },
        uint32_t(0)
    );
    hitRecordOffsets[0] = 0;
    // Find the geometry user ptr array size
    //uint32_t geomDataArraySize = (IsTriangle) ? 1 : std::transform_reduce
    uint32_t geomDataArraySize = std::transform_reduce
    (
        ppResult.concretePrimRanges.begin(),
        ppResult.concretePrimRanges.end(),
        uint32_t(0),
        std::plus{},
        [](const PrimRangeArray& primRanges)
        {
            constexpr Vector2ui INVALID_RANGE(std::numeric_limits<uint32_t>::max());
            uint32_t i = 0;
            while((i < primRanges.size()) && (primRanges[i] != INVALID_RANGE)) ++i;
            return i;
        }
    );
    //
    size_t hitRecordCount = hitRecordOffsets.back();
    size_t totalLeafCount = this->concreteLeafRanges.back()[1];
    size_t concreteAccelCount = this->concreteLeafRanges.size();
    // We do not instantiate per-primitive transformed primitives
    // So allocating extra buffer does not makes sense
    if constexpr(PER_PRIM_TRANSFORM)
        concreteAccelCount = 0;
    //
    MemAlloc::AllocateMultiData(Tie(hConcreteScenes, hInstanceScenes,
                                    hTransformKeys, hAllHitRecords,
                                    hInstanceHitRecordOffsets,
                                    hAllLeafs, hTransformGroupSoAList,
                                    pgSoA, geomUserData),
                                mem,
                                {concreteAccelCount, this->InstanceCount(),
                                 this->InstanceCount(), hitRecordCount,
                                 hitRecordOffsets.size(), totalLeafCount,
                                 transformSoAOffsets.back(), 1,
                                 geomDataArraySize});
    // Copy pgSoA to common buffer (easy)
    auto pgSoALocal = static_cast<const PG&>(this->pg).SoA();
    queue.MemcpyAsync(pgSoA, Span<const PGSoA>(&pgSoALocal, 1));
    queue.MemcpyAsync(hInstanceHitRecordOffsets, Span<const uint32_t>(hitRecordOffsets));
    // Copy TransformSoA's to local buffer
    {
        uint32_t i = 0;
        for(const auto& [_, wI] : this->workInstances)
        {
            size_t start = transformSoAOffsets[i];
            size_t end = transformSoAOffsets[i + 1];
            size_t size = end - start;
            i++;

            auto copyLoc = hTransformGroupSoAList.subspan(start, size);
            wI->CopyTransformSoA(copyLoc, queue);
        }
    }
    // Copy Leafs and generate primitive keys
    // Since we are CPU directly refer to the host allocated vectors.
    Span<const Vector2ui> hConcreteLeafRanges(this->concreteLeafRanges);
    Span<const PrimRangeArray> hConcretePrimRanges(ppResult.concretePrimRanges);
    uint32_t blockCount = static_cast<uint32_t>(this->concreteLeafRanges.size());
    using namespace std::string_literals;
    static const auto KernelName = "KCGeneratePrimitiveKeys-"s + std::string(TypeName());
    queue.IssueBlockKernel<KCGeneratePrimitiveKeys>
    (
        KernelName,
        DeviceBlockIssueParams
        {
            .gridSize = blockCount,
            .blockSize = StaticThreadPerBlock1D()
        },
        // Output
        hAllLeafs,
        // Input
        hConcretePrimRanges,
        hConcreteLeafRanges,
        // Constant
        this->pg.GroupId()
    );
    // Copy transform keys
    queue.MemcpyAsync(hTransformKeys, Span<const TransformKey>(ppResult.surfData.transformKeys));

    // Write the hit records
    for(size_t i = 0; i < this->InstanceCount(); i++)
    {
        // Set the subspan
        size_t hrStart = hitRecordOffsets[i];
        size_t hrSize = hitRecordOffsets[i + 1] - hrStart;
        auto hitRSpan = hAllHitRecords.subspan(hrStart, hrSize);

        size_t localPrimKeyOffset = 0;
        uint32_t concreteIndex = this->concreteIndicesOfInstances[i];
        Vector2ui leafRange = hConcreteLeafRanges[concreteIndex];

        // Find the actual range
        auto workOffset = std::upper_bound(this->workInstanceOffsets.cbegin(),
                                           this->workInstanceOffsets.cend(), i);
        assert(workOffset != this->workInstanceOffsets.cend());
        size_t workId = size_t(std::distance(this->workInstanceOffsets.cbegin(), workOffset - 1));
        size_t transSoASize = transformSoAOffsets[workId + 1] - transformSoAOffsets[workId];
        const Byte* dTransSoAPtr = hTransformGroupSoAList.subspan(transformSoAOffsets[workId],
                                                                  transSoASize).data();
        auto accKey = AcceleratorKey::CombinedKey(uint32_t(workId), uint32_t(i));
        for(uint32_t j = 0; j < hitRSpan.size(); j++)
        {
            const Vector2ui& primRange = hConcretePrimRanges[concreteIndex][j];
            uint32_t localSize = primRange[1] - primRange[0];

            hitRSpan[j] = EmbreeHitRecord<>
            {
                .tgData         = dTransSoAPtr,
                .pgData         = pgSoA.data(),
                .lmKey          = ppResult.surfData.lightOrMatKeys[i][j],
                .transformKey   = ppResult.surfData.transformKeys[i],
                .acceleratorKey = accKey,
                .dPrimKeys      = hAllLeafs.subspan(leafRange[0] + localPrimKeyOffset,
                                                    localSize),
                .alphaMap       = ppResult.surfData.alphaMaps[i][j],
                .cullFace       = ppResult.surfData.cullFaceFlags[i][j],
                .isTriangle     = IsTriangle
            };
            localPrimKeyOffset += localSize;
        }
        assert(localPrimKeyOffset == leafRange[1] - leafRange[0]);
    }

    // We set all the data we need.
    // Let embree to take the wheel
    geomGlobalData.hAllHitRecords = ToConstSpan(hAllHitRecords);
    geomGlobalData.hInstanceHitRecordOffsets = ToConstSpan(hInstanceHitRecordOffsets);
    for(auto& hr : geomUserData) hr.geomGlobalData = &geomGlobalData;
    // Embree may use its own MT environment
    queue.Barrier().Wait();
    // Create sub scenes
    if constexpr(IsTriangle && !PER_PRIM_TRANSFORM)
    {
        MultiBuildViaTriangle_CLT(ppResult, queue);
    }
    else if constexpr(!IsTriangle && !PER_PRIM_TRANSFORM)
    {
        MultiBuildViaUser_CLT(ppResult, queue);
    }
    else if constexpr(IsTriangle && PER_PRIM_TRANSFORM)
    {
        MultiBuildViaTriangle_PPT(ppResult, queue);
    }
    else if constexpr(!IsTriangle && PER_PRIM_TRANSFORM)
    {
        MultiBuildViaUser_PPT(ppResult, queue);
    }
    else static_assert(!IsTriangle && !PER_PRIM_TRANSFORM,
                       "Unknown params on Embree build!");
    //
    // We refer stack-local data so wait until threads are done
    queue.Barrier().Wait();
}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::WriteInstanceKeysAndAABBs(Span<AABB3>,
                                                           Span<AcceleratorKey>,
                                                           const GPUQueue&) const
{
    throw MRayError("For Embree, this function should not be called");
}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::CastLocalRays(// Output
                                               Span<HitKeyPack>,
                                               Span<MetaHit>,
                                               // I-O
                                               Span<BackupRNGState>,
                                               Span<RayGMem>,
                                               // Input
                                               Span<const RayIndex>,
                                               Span<const CommonKey>,
                                               // Constants
                                               CommonKey,
                                               const GPUQueue&)
{
    throw MRayError("For Embree, this function should not be called");
}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::CastVisibilityRays(// Output
                                                    Bitspan<uint32_t>,
                                                    // I-O
                                                    Span<BackupRNGState>,
                                                    // Input
                                                    Span<const RayGMem>,
                                                    Span<const RayIndex>,
                                                    Span<const CommonKey>,
                                                    // Constants
                                                    CommonKey,
                                                    const GPUQueue&)
{
    throw MRayError("For OptiX, this function should not be called");
}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::AcquireIASConstructionParams(Span<RTCScene> hSceneHandles,
                                                              Span<Matrix4x4> hInstanceMatrices,
                                                              Span<uint32_t> hInstanceHitRecordCounts,
                                                              Span<const EmbreeHitRecord<>*> dHitRecordPtrs,
                                                              const GPUQueue& queue) const
{
    queue.MemcpyAsync(hSceneHandles, ToConstSpan(hInstanceScenes));
    // The hard part
    for(const auto& work : this->workInstances)
    {
        CommonKey workId = work.first;
        uint32_t localStart = this->workInstanceOffsets[workId];
        uint32_t localEnd = this->workInstanceOffsets[workId + 1];
        uint32_t localCount = localEnd - localStart;
        //
        auto hTransformsLocal = hTransformKeys.subspan(localStart, localCount);
        auto hMatricesLocal = hInstanceMatrices.subspan(localStart, localCount);
        work.second->GetCommonTransforms(hMatricesLocal, ToConstSpan(hTransformsLocal),
                                         queue);
    }
    DeviceAlgorithms::AdjacentDifference(hInstanceHitRecordCounts,
                                         ToConstSpan(hInstanceHitRecordOffsets),
                                         queue, [](uint32_t l, uint32_t r) -> uint32_t
    {
        return r - l;
    });
    DeviceAlgorithms::Transform(dHitRecordPtrs, ToConstSpan(hAllHitRecords), queue,
                                [](const EmbreeHitRecord<>& r)
    {
        return &r;
    });
}

template<PrimitiveGroupC PG>
void AcceleratorGroupEmbree<PG>::OffsetAccelKeyInRecords(uint32_t instanceRecordStartOffset)
{
    assert(this->globalWorkIdToLocalOffset != std::numeric_limits<uint32_t>::max());
    // Calculate hit records for each build input
    for(auto& hr : hAllHitRecords)
    {
        CommonKey batch = hr.acceleratorKey.FetchBatchPortion();
        CommonKey index = hr.acceleratorKey.FetchIndexPortion();
        batch += this->globalWorkIdToLocalOffset;
        auto accKey = AcceleratorKey::CombinedKey(batch, index);
        hr.acceleratorKey = accKey;
    };
    geomGlobalData.globalToLocalOffset = instanceRecordStartOffset;
}

template<PrimitiveGroupC PG>
size_t AcceleratorGroupEmbree<PG>::HitRecordCount() const
{
    return hAllHitRecords.size();
}

template<PrimitiveGroupC PG>
typename AcceleratorGroupEmbree<PG>::DataSoA
AcceleratorGroupEmbree<PG>::SoA() const
{
    return EmptyType{};
}

template<PrimitiveGroupC PG>
size_t AcceleratorGroupEmbree<PG>::GPUMemoryUsage() const
{
    return mem.Size();
}