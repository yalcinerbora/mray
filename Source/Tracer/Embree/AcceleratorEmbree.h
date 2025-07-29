#pragma once

#include "AcceleratorC.h"
#include "AcceleratorWork.h"

#include <embree4/rtcore.h>

namespace EmbreeAccelDetail
{
    // Phony type to satisfy the concept
    template<PrimitiveGroupC PrimGroup,
             TransformGroupC TransGroup = TransformGroupIdentity>
    class AcceleratorEmbree
    {
        public:
        using PrimHit       = typename PrimGroup::Hit;
        using PrimDataSoA   = typename PrimGroup::DataSoA;
        using TransDataSoA  = typename TransGroup::DataSoA;
        using HitResult     = HitResultT<PrimHit>;
        using DataSoA       = EmptyType;

        public:
        MR_GF_DECL
        AcceleratorEmbree(TransDataSoA, PrimDataSoA,
                         DataSoA, AcceleratorKey) {}

        MR_GF_DECL
        Optional<HitResult>
        ClosestHit(BackupRNG&, const Ray&, const Vector2&) const
        { return std::nullopt; }

        MR_GF_DECL
        Optional<HitResult>
        FirstHit(BackupRNG&, const Ray&, const Vector2&) const
        { return std::nullopt; }

        MR_GF_DECL
        TransformKey
        GetTransformKey() const { return TransformKey::InvalidKey(); }
    };
}

static constexpr uint32_t EMBREE_BATCH_SIZE = 16;

// I am going insane here why valid ray is -1
// Abstracting that away
static constexpr int EMBREE_VALID_RAY = -1;
static constexpr int EMBREE_INVALID_RAY = 0;
static constexpr float EMBREE_IS_OCCLUDED_RAY = -std::numeric_limits<float>::infinity();
static constexpr unsigned int EMBREE_ALL_VALID_MASK = std::numeric_limits<unsigned int>::max();

static constexpr Vector2 EmbreeBaryToMRay(Vector2 ab)
{
    Float c = Float(1) - ab[0] - ab[1];
    return Vector2(c, ab[0]);
}

// Similar implementation like OptiX.
template<class PGSoA = void, class TGSoA = void>
struct EmbreeHitRecord
{
    // These two are duplicated for each geometry
    // unfortunately due to design of Embree
    const TGSoA* tgData;
    const PGSoA* pgData;
    // Each geometry accepts a single pointer
    // due to design.
    // Grouped geometries are not supported by this design
    // So for each geometry, we need to give non-batched
    // data, so all arrays of TracerConstants::MaxPrimPerBatch
    // will be segregated.
    LightOrMatKey       lmKey;
    // MRay accepts single transform per batch
    // However we need the data per geometry so this
    // is also duplicated
    TransformKey        transformKey;
    // Accelerator Key to use in SSS
    AcceleratorKey      acceleratorKey;
    // Actual keys of the primitives in the geometry
    // This can be accessed via primID field (hopefully)
    Span<const PrimitiveKey>    dPrimKeys;
    // Optional alpha map
    Optional<AlphaMap>          alphaMap;
    //
    // Also embree do not support cull back face
    // natively AFAIK. But we want to use cool AVX/SSE
    // intersection routines for triangles
    // so this will be ignored for triangles
    bool                        cullFace;
    // Embree barycentric coordinates are different from MRay
    // ray caster can check and reorder the coordinates via this
    bool                        isTriangle;
};

struct MRayEmbreeContext
{
    RTCDevice           device  = nullptr;
    RTCScene            scene   = nullptr;
    std::atomic_int64_t size;
    // Constructors & Destructor
                        MRayEmbreeContext();
                        MRayEmbreeContext(const MRayEmbreeContext&) = delete;
                        MRayEmbreeContext(MRayEmbreeContext&&) = delete;
    MRayEmbreeContext&  operator=(const MRayEmbreeContext&) = delete;
    MRayEmbreeContext&  operator=(MRayEmbreeContext&&) = delete;
                        ~MRayEmbreeContext();
    //
    static void         ErrorCallback(void* userPtr, const RTCError code, const char* str);

    static bool         AllocationCallback(void* userPtr, ssize_t bytes, bool post);
};

class AcceleratorWorkEmbreeI : public AcceleratorWorkI
{
    virtual RTCBoundsFunction       AABBGenFunction() const = 0;
    virtual RTCFilterFunctionN      FilterFunction() const = 0;
    virtual RTCIntersectFunctionN   IntersectionFunction() const = 0;
    virtual RTCOccludedFunctionN    OccludedFunction() const = 0;
};

class AcceleratorGroupEmbreeI : public AcceleratorGroupI
{
    public:
    virtual ~AcceleratorGroupEmbreeI() = default;

    virtual void AcquireIASConstructionParams(Span<RTCScene> hSceneHandles,
                                              Span<Matrix4x4> hInstanceMatrices,
                                              Span<uint32_t> hInstanceHitRecordCounts,
                                              Span<const EmbreeHitRecord<>*> dHitRecordPtrs,
                                              const GPUQueue& queue) const = 0;

    virtual void OffsetAccelKeyInRecords(uint32_t instanceRecordStartOffset) = 0;
    virtual size_t HitRecordCount() const = 0;
};

struct EmbreeRayQueryContext
{
    template<class T>
    using ArrayT = StaticVector<T, EMBREE_BATCH_SIZE>;

    RTCRayQueryContext  baseContext;
    ArrayT<BackupRNGState>  rngStates;
    ArrayT<BackupRNG>       rng;
    ArrayT<AcceleratorKey>  localAccelKeys;
};

struct EmbreGlobalUserData
{
    Span<const EmbreeHitRecord<>>   hAllHitRecords;
    Span<const uint32_t>            hInstanceHitRecordOffsets;
    //
    uint32_t globalToLocalOffset = std::numeric_limits<uint32_t>::max();
};

struct EmbreeGeomUserData
{
    const EmbreGlobalUserData*  geomGlobalData = nullptr;
    uint32_t                    recordIndexForBounds = std::numeric_limits<uint32_t>::max();
};

template<AccelGroupC AG, TransformGroupC TG>
struct AcceleratorWorkEmbree
    : public AcceleratorWork<AG, TG , AcceleratorWorkEmbreeI>
{
    RTCBoundsFunction       AABBGenFunction() const override;
    RTCFilterFunctionN      FilterFunction() const override;
    RTCIntersectFunctionN   IntersectionFunction() const override;
    RTCOccludedFunctionN    OccludedFunction() const override;
};

template<PrimitiveGroupC PrimitiveGroupType>
class AcceleratorGroupEmbree final
    : public AcceleratorGroupT<AcceleratorGroupEmbree<PrimitiveGroupType>, PrimitiveGroupType, AcceleratorGroupEmbreeI>
{
    using Base = AcceleratorGroupT<AcceleratorGroupEmbree<PrimitiveGroupType>,
                                   PrimitiveGroupType, AcceleratorGroupEmbreeI>;
    public:
    static std::string_view TypeName();
    using PrimitiveGroup    = PrimitiveGroupType;
    using DataSoA           = EmptyType;
    using PGSoA             = typename PrimitiveGroup::DataSoA;

    template<class TG = TransformGroupIdentity>
    using Accelerator = EmbreeAccelDetail::AcceleratorEmbree<PrimitiveGroup, TG>;

    static constexpr auto TransformLogic = PrimitiveGroup::TransformLogic;
    static constexpr auto IsTriangle = TrianglePrimGroupC<PrimitiveGroup>;

    private:
    RTCDevice               rtcDevice;
    DeviceMemory            mem;
    // Per-concrete data
    Span<RTCScene>          hConcreteScenes;
    // Per-instance data
    Span<RTCScene>          hInstanceScenes;
    Span<TransformKey>      hTransformKeys;
    // Per-instance per-prim batch data
    Span<EmbreeHitRecord<>> hAllHitRecords;
    Span<uint32_t>          hInstanceHitRecordOffsets;
    // Per-concrete per-prim data
    Span<PrimitiveKey>      hAllLeafs;
    // Per-transform group data (type ereased)
    Span<Byte>              hTransformGroupSoAList;
    //
    Span<PGSoA>             pgSoA;
    // Geometry User Pointer generic class.
    // Common for all accelerators in this group.
    // Inner data will be accessed via
    // instId/geomId/primId fields
    EmbreGlobalUserData         geomGlobalData;
    // This is needed for custom geometry
    // since bounds function does not have instId/geomId/primId
    // fields
    Span<EmbreeGeomUserData>    geomUserData;


    void MultiBuildViaTriangle_CLT(const PreprocessResult& ppResult,
                                  const GPUQueue& queue);
    void MultiBuildViaUser_CLT(const PreprocessResult& ppResult,
                               const GPUQueue& queue);
    void MultiBuildViaTriangle_PPT(const PreprocessResult& ppResult,
                                   const GPUQueue& queue);
    void MultiBuildViaUser_PPT(const PreprocessResult& ppResult,
                               const GPUQueue& queue);

    public:
    // Constructors & Destructor
    AcceleratorGroupEmbree(uint32_t accelGroupId,
                           ThreadPool&,
                           const GPUSystem&,
                           const GenericGroupPrimitiveT& pg,
                           const AccelWorkGenMap&);
    //
    void PreConstruct(const BaseAcceleratorI*) override;
    void Construct(AccelGroupConstructParams, const GPUQueue&) override;
    void WriteInstanceKeysAndAABBs(Span<AABB3> dAABBWriteRegion,
                                   Span<AcceleratorKey> dKeyWriteRegion,
                                   const GPUQueue&) const override;
    // Functionality
    void CastLocalRays(// Output
                       Span<HitKeyPack> dHitIds,
                       Span<MetaHit> dHitParams,
                       // I-O
                       Span<BackupRNGState> dRNGStates,
                       Span<RayGMem> dRays,
                       // Input
                       Span<const RayIndex> dRayIndices,
                       Span<const CommonKey> dAccelKeys,
                       // Constants
                       CommonKey workId,
                       const GPUQueue& queue) override;

    void CastVisibilityRays(// Output
                            Bitspan<uint32_t> dIsVisibleBuffer,
                            // I-O
                            Span<BackupRNGState> dRNGStates,
                            // Input
                            Span<const RayGMem> dRays,
                            Span<const RayIndex> dRayIndices,
                            Span<const CommonKey> dAccelKeys,
                            // Constants
                            CommonKey workId,
                            const GPUQueue& queue) override;

    // Embree Related
    void    AcquireIASConstructionParams(Span<RTCScene> hSceneHandles,
                                         Span<Matrix4x4> hInstanceMatrices,
                                         Span<uint32_t> hInstanceHitRecordCounts,
                                         Span<const EmbreeHitRecord<>*> dHitRecordPtrs,
                                         const GPUQueue& queue) const override;
    void    OffsetAccelKeyInRecords(uint32_t instanceRecordStartOffset) override;
    size_t  HitRecordCount() const override;

    DataSoA SoA() const;
    size_t  GPUMemoryUsage() const override;
};

class BaseAcceleratorEmbree final : public BaseAcceleratorT<BaseAcceleratorEmbree>
{
    public:
    static std::string_view TypeName();

    private:
    MRayEmbreeContext       embreeContext;
    RTCTraversable          baseTraversable;
    //
    DeviceMemory            allMem;
    // Per-AccelGroup
    Span<uint32_t>                  hInstanceHRStartOffsets;
    Span<const EmbreeHitRecord<>*>  hAllHitRecordPtrs;
    // For local ray casting
    Span<size_t>    hInstanceBatchStartOffsets;
    Span<Matrix4x4> hGlobalInstanceInvTransforms;
    Span<RTCScene>  hGlobalSceneHandles;

    protected:
    AABB3   InternalConstruct(const std::vector<size_t>& instanceOffsets) override;
    public:
    // Constructors & Destructor
            BaseAcceleratorEmbree(ThreadPool&, const GPUSystem&,
                                  const AccelGroupGenMap&,
                                  const AccelWorkGenMap&);

    //
    void    CastRays(// Output
                     Span<HitKeyPack> dHitIds,
                     Span<MetaHit> dHitParams,
                     // I-O
                     Span<BackupRNGState> dRNGStates,
                     Span<RayGMem> dRays,
                     // Input
                     Span<const RayIndex> dRayIndices,
                     const GPUQueue& queue) override;

    void    CastVisibilityRays(// Output
                               Bitspan<uint32_t> dIsVisibleBuffer,
                               // I-O
                               Span<BackupRNGState> dRNGStates,
                               // Input
                               Span<const RayGMem> dRays,
                               Span<const RayIndex> dRayIndices,
                               const GPUQueue& queue) override;

    void    CastLocalRays(// Output
                          Span<HitKeyPack> dHitIds,
                          Span<MetaHit> dHitParams,
                          // I-O
                          Span<BackupRNGState> dRNGStates,
                          Span<RayGMem> dRays,
                          // Input
                          Span<const RayIndex> dRayIndices,
                          Span<const AcceleratorKey> dAccelKeys,
                          CommonKey dAccelKeyBatchPortion,
                          const GPUQueue& queue) override;

    void    AllocateForTraversal(size_t maxRayCount) override;
    size_t  GPUMemoryUsage() const override;

    // Embree Related
    RTCDevice GetRTCDeviceHandle() const;
};

#include "AcceleratorEmbree.hpp"