#pragma once

#pragma once

#include <algorithm>
#include <bitset>

#include "Core/Types.h"
#include "Core/MemAlloc.h"
#include "Core/AABB.h"
#include "Core/GraphicsFunctions.h"

#include "Device/GPUAlgForward.h"

#include "TracerTypes.h"
#include "ParamVaryingData.h"
#include "AcceleratorC.h"
#include "PrimitiveC.h"
#include "TransformC.h"
#include "Random.h"
#include "RayPartitioner.h"
#include "AcceleratorWork.h"

namespace LBVHAccelDetail
{
    class BitStack
    {
        public:
        static constexpr uint32_t MAX_DEPTH = sizeof(uint64_t) * CHAR_BIT;

        enum TraverseState
        {
            FIRST_ENTRY = 0b00,
            U_TURN      = 0b01,
            GO_UP       = 0b10,
            // This shouldn't happen?
            INVALID     = 0b11
        };

        private:
        uint64_t    stack;
        uint32_t    depth;

        public:
        MRAY_HYBRID                 BitStack();
        MRAY_HYBRID                 BitStack(uint64_t state, uint32_t depth);
        MRAY_HYBRID
        MRAY_HYBRID void            WipeLowerBits();
        MRAY_HYBRID TraverseState   CurrentState() const;
        MRAY_HYBRID void            MarkAsTraversed();
        MRAY_HYBRID void            Descend();
        MRAY_HYBRID void            Ascend();
        // Access
        MRAY_HYBRID uint32_t        Depth() const;
        template<uint32_t SBits, uint32_t DBits>
        MRAY_HYBRID uint32_t        CompressState() const;
    };

    // Utilize "Key" type here to use the MSB as a flag
    using ChildIndex = KeyT<uint32_t, 1, 31>;
    static constexpr uint32_t IS_LEAF = 1;
    static constexpr uint32_t IS_INTERNAL = 0;

    class Delta
    {
        private:
        const Span<const uint64_t>& dMortonCodes;

        public:
        MRAY_HYBRID Delta(const Span<const uint64_t>&);

        MRAY_HYBRID
        int32_t operator()(int32_t i, int32_t j) const;
    };


    struct LBVHNode
    {
        ChildIndex  leftIndex;
        ChildIndex  rightIndex;
        uint32_t    parentIndex;
    };
    // This is seperate because of volatile writes
    // on construction, also it makes the alignment
    // better
    struct alignas(8) LBVHBoundingBox
    {
        Float min[3];
        Float max[3];
    };

    // SoA data of triangle group
    struct LBVHAcceleratorSoA
    {
        // Per accelerator instance stuff
        Span<const CullFaceFlagArray>           dCullFace;
        Span<const AlphaMapArray>               dAlphaMaps;
        Span<const LightOrMatKeyArray>          dLightOrMatKeys;
        Span<const PrimRangeArray>              dPrimitiveRanges;
        Span<const TransformKey>                dInstanceTransforms;
        Span<const Span<const PrimitiveKey>>    dLeafs;
        Span<const Span<const LBVHNode>>        dNodes;
        Span<const Span<const LBVHBoundingBox>> dBoundingBoxes;
    };

    template<PrimitiveGroupC PrimGroup,
             TransformGroupC TransGroup = TransformGroupIdentity>
    class AcceleratorLBVH
    {
        public:
        using PrimHit       = typename PrimGroup::Hit;
        using PrimDataSoA   = typename PrimGroup::DataSoA;
        using TransDataSoA  = typename TransGroup::DataSoA;
        using HitResult     = HitResultT<PrimHit>;
        using DataSoA       = LBVHAcceleratorSoA;

        private:
        // Accel Related
        // TODO: Check if expansion reduces performance
        //
        // Specifically load the range array by value,
        // we will do linear search over it
        // Maybe compiler put this on register space
        PrimRangeArray              primRanges;
        // Rest is by reference (except the tKey & cullFace, these are a single word)
        CullFaceFlagArray           cullFaceFlags;
        const AlphaMapArray&        alphaMaps;
        const LightOrMatKeyArray&   lmKeys;
        Span<const PrimitiveKey>    leafs;
        Span<const LBVHNode>        nodes;
        Span<const LBVHBoundingBox> boundingBoxes;
        // Primitive Related
        TransformKey                transformKey;
        const TransDataSoA&         transformSoA;
        const PrimDataSoA&          primitiveSoA;

        template<auto = MRAY_PRIM_TGEN_FUNCTION(PrimGroup, TransGroup)>
        MRAY_GPU
        Optional<HitResult>     IntersectionCheck(const Ray& ray,
                                                  const Vector2& tMinMax,
                                                  Float xi,
                                                  const PrimitiveKey& primKey) const;
        public:
        // Constructors & Destructor
        MRAY_GPU                AcceleratorLBVH(const TransDataSoA& tSoA,
                                                const PrimDataSoA& pSoA,
                                                const DataSoA& dataSoA,
                                                AcceleratorKey aId);
        MRAY_GPU
        TransformKey            GetTransformKey() const;
        MRAY_GPU
        Optional<HitResult>     ClosestHit(BackupRNG& rng, const Ray&, const Vector2&) const;
        MRAY_GPU
        Optional<HitResult>     FirstHit(BackupRNG& rng, const Ray&, const Vector2&) const;
    };
}

template<PrimitiveGroupC PrimitiveGroupType>
class AcceleratorGroupLBVH final : public AcceleratorGroupT<AcceleratorGroupLBVH<PrimitiveGroupType>, PrimitiveGroupType>
{
    using Base = AcceleratorGroupT<AcceleratorGroupLBVH<PrimitiveGroupType>, PrimitiveGroupType>;

    public:
    static std::string_view TypeName();

    using PrimitiveGroup    = PrimitiveGroupType;
    using DataSoA           = LBVHAccelDetail::LBVHAcceleratorSoA;
    using LBVHNode          = LBVHAccelDetail::LBVHNode;
    using LBVHBoundingBox   = LBVHAccelDetail::LBVHBoundingBox;

    template<class TG = TransformGroupIdentity>
    using Accelerator = LBVHAccelDetail::AcceleratorLBVH<PrimitiveGroup, TG>;

    static constexpr auto TransformLogic = PrimitiveGroup::TransformLogic;

    private:
    DeviceMemory                mem;
    DataSoA                     data;
    // Per-instance (All accelerators will have these)
    Span<CullFaceFlagArray>     dCullFaceFlags;
    Span<AlphaMapArray>         dAlphaMaps;
    Span<LightOrMatKeyArray>    dLightOrMatKeys;
    Span<TransformKey>          dTransformKeys;
    // These are duplicated since we will have only 8 prim batch per instance
    Span<PrimRangeArray>        dPrimitiveRanges;
    // These are not-duplicated, Instances have copy of the spans.
    // spans may be the same
    Span<Span<const PrimitiveKey>>      dLeafs;
    Span<Span<const LBVHNode>>          dNodes;
    Span<Span<const LBVHBoundingBox>>   dNodeAABBs;
    // Global data, all accelerator leafs are in here
    Span<PrimitiveKey>              dAllLeafs;
    Span<LBVHNode>                  dAllNodes;
    Span<LBVHBoundingBox>           dAllNodeAABBs;

    void    MultiBuildLBVH(Pair<const CommonKey, const AcceleratorWorkI*>* accelWork,
                           const std::vector<Vector2ui>& instanceNodeRanges,
                           const std::vector<Vector2ui>& concreteNodeRanges,
                           const GPUQueue& queue);

    public:
    // Constructors & Destructor
    AcceleratorGroupLBVH(uint32_t accelGroupId,
                         ThreadPool&,
                         const GPUSystem&,
                         const GenericGroupPrimitiveT& pg,
                         const AccelWorkGenMap&);
    //
    void    Construct(AccelGroupConstructParams, const GPUQueue&) override;
    void    WriteInstanceKeysAndAABBs(Span<AABB3> dAABBWriteRegion,
                                      Span<AcceleratorKey> dKeyWriteRegion,
                                      const GPUQueue&) const override;

    // Functionality
    void    CastLocalRays(// Output
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

    void    CastVisibilityRays(// Output
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


    DataSoA SoA() const;
    size_t  GPUMemoryUsage() const override;
};

class BaseAcceleratorLBVH final : public BaseAcceleratorT<BaseAcceleratorLBVH>
{
    public:
    static std::string_view TypeName();
    using LBVHNode          = LBVHAccelDetail::LBVHNode;
    using LBVHBoundingBox   = LBVHAccelDetail::LBVHBoundingBox;
    static constexpr uint32_t DepthBitCount = 5;
    static constexpr uint32_t StackBitCount = 27;
    static_assert(DepthBitCount + StackBitCount == sizeof(uint32_t) * CHAR_BIT);

    private:
    DeviceMemory            accelMem;
    Span<AcceleratorKey>    dLeafKeys;
    Span<AABB3>             dLeafAABBs;
    Span<LBVHNode>          dNodes;
    Span<LBVHBoundingBox>   dBoundingBoxes;
    //
    DeviceMemory            stackMem;
    Span<uint32_t>          dBitStacks;
    Span<uint32_t>          dPrevNodeIndices;
    RayPartitioner          rayPartitioner;
    size_t                  maxPartitionCount;

    protected:
    AABB3 InternalConstruct(const std::vector<size_t>& instanceOffsets) override;

    public:
    // Constructors & Destructor
    BaseAcceleratorLBVH(ThreadPool&, const GPUSystem&,
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
};

inline
BaseAcceleratorLBVH::BaseAcceleratorLBVH(ThreadPool& tp, const GPUSystem& sys,
                                         const AccelGroupGenMap& aGen,
                                         const AccelWorkGenMap& globalWorkMap)
    : BaseAcceleratorT<BaseAcceleratorLBVH>(tp, sys, aGen, globalWorkMap)
    , accelMem(sys.AllGPUs(), 8_MiB, 32_MiB, false)
    , stackMem(sys.AllGPUs(), 8_MiB, 32_MiB, false)
    , rayPartitioner(sys)
    , maxPartitionCount(0)
{}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCGenMortonCode(// Output
                     MRAY_GRID_CONSTANT const Span<uint64_t> dMortonCodes,
                     // Inputs
                     MRAY_GRID_CONSTANT const Span<const uint32_t> dSegmentRanges,
                     MRAY_GRID_CONSTANT const Span<const AABB3> dInstanceAABBs,
                     //
                     MRAY_GRID_CONSTANT const Span<const Vector3> dAllPrimCenters,
                     // Constants
                     MRAY_GRID_CONSTANT const uint32_t blockPerInstance);

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCConstructLBVHInternalNodes(// Output
                                  MRAY_GRID_CONSTANT const Span<LBVHAccelDetail::LBVHNode> dAllNodes,
                                  MRAY_GRID_CONSTANT const Span<uint32_t> dAllLeafParentIndices,
                                  // Inputs
                                  MRAY_GRID_CONSTANT const Span<const uint32_t> dLeafSegmentRanges,
                                  MRAY_GRID_CONSTANT const Span<const uint32_t> dNodeSegmentRanges,
                                  MRAY_GRID_CONSTANT const Span<const uint64_t> dAllMortonCodes,
                                  MRAY_GRID_CONSTANT const Span<const uint32_t> dAllLeafIndices,
                                  // Constants
                                  MRAY_GRID_CONSTANT const uint32_t blockPerInstance,
                                  MRAY_GRID_CONSTANT const uint32_t instanceCount);

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCUnionLBVHBoundingBoxes(// I-O
                              MRAY_GRID_CONSTANT const Span<LBVHAccelDetail::LBVHBoundingBox> dAllNodeAABBs,
                              MRAY_GRID_CONSTANT const Span<uint32_t> dAtomicCounters,
                              // Inputs
                              MRAY_GRID_CONSTANT const Span<const LBVHAccelDetail::LBVHNode> dAllNodes,
                              MRAY_GRID_CONSTANT const Span<const uint32_t> dAllLeafParentIndices,
                              MRAY_GRID_CONSTANT const Span<const uint32_t> dLeafSegmentRanges,
                              MRAY_GRID_CONSTANT const Span<const uint32_t> dNodeSegmentRanges,
                              MRAY_GRID_CONSTANT const Span<const AABB3> dAllLeafAABBs,
                              // Constants
                              MRAY_GRID_CONSTANT const uint32_t blockPerInstance,
                              MRAY_GRID_CONSTANT const uint32_t instanceCount);

#include "AcceleratorLBVH.hpp"