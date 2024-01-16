#pragma once

#include <type_traits>
#include <tuple>
#include <unordered_map>

#include "Core/MRayDataType.h"
#include "Core/MemAlloc.h"
#include "Device/GPUSystemForward.h"

#include "TracerTypes.h"

// Transform types
// This enumeration is tied to a primitive group
// a primitive group cannot have both constant-local and
// per-primitive transform.
enum class PrimTransformType : uint8_t
{
    // Single transform is applied to the group of primitives
    // Meaning that inverse of this can be applied to ray
    // and for instanced primitives, a single accelerator can be built
    // for group of primitives (meshes)
    LOCALLY_CONSTANT_TRANSFORM,
    // Each primitive in a group will different
    // and (maybe) multiple transforms.
    // This prevents the ray-casting to utilize a single local space
    // accelerator. Such primitive instantiations
    // will have unique accelerators.
    // As an example, skinned meshes and morph targets
    // will have such property.
    PER_PRIMITIVE_TRANSFORM
};

enum class PrimitiveAttributeLogic
{
    POSITION,
    INDEX,
    NORMAL,
    RADIUS,
    TANGENT,
    UV0,
    UV1,
    WEIGHT,
    WEIGHT_INDEX
};

class IdentityTransformContext;

template<class Surface, class Prim, class Hit>
using SurfaceGenFunc = Surface(Prim::*)(const Hit&,
                                        const DiffRay&,
                                        const Ray&);

template<class TransformContext, class PrimData, class TransformData>
using TransformContextGenFunc = TransformContext(*)(const TransformData&,
                                                    const PrimData&,
                                                    TransformId,
                                                    PrimitiveId);

template <class PrimType>
concept PrimC = requires(PrimType pt)
{
    typename PrimType::Hit;
    typename PrimType::DataSoA;

    // Has this specific constructor
    requires requires(const IdentityTransformContext& tc,
                      const typename PrimType::DataSoA& soaData)
    {
        PrimType(tc, soaData, PrimitiveId{});
    };

    // Intersect function
    {pt.Intersects(Ray{})
    } -> std::same_as<Optional<IntersectionT<typename PrimType::Hit>>>;

    // Parametrized sampling
    requires requires(Float f, Vector3 v3, RNGDispenser rng)
    {
        // Sample a position over the parametrized surface
        {pt.SamplePosition(rng)
        } -> std::same_as<SampleT<Vector3>>;

        // PDF of sampling such position
        // Conservative towards non-zero values
        // primitive may try to resolve position is on surface
        // but inherently assumes position is on surface
        {pt.PdfPosition(Vector3{})
        } -> std::same_as<Float>;

        {pt.SampleRNCount()
        } -> std::same_as<uint32_t>;
    };

    // Total surface area
    {pt.GetSurfaceArea()
    } -> std::same_as<Float>;

    // AABB of the primitive
    {pt.GetAABB()
    } -> std::same_as<AABB3>;

    // Geometric center of primitive
    {pt.GetCenter()
    } -> std::same_as<Vector3>;

    // Voxelization
    // Given a scene extent/resolution
    // Generate voxels for this primitive
    // voxels will be used for approixmate invariant scene representation
    // Many methods require discretized scene representation
    // for GPU-based renderer voxels are a good choice
    requires requires(Span<uint64_t> s0, Span<Vector2us> s1)
    {
        {pt.Voxelize(s0, s1, bool{},
                     VoxelizationParameters{})
        } -> std::same_as<uint32_t>;
    };

    // TODO: Better design maybe? // Only 2D surface parametrization?
    //
    // Accelerator "AnyHit" requirements
    // Acquire primitive's surface parametrization
    // (Aka. uv map)
    // Good: No data duplication no prinicpiled data for alpha testing
    //
    // Bad: All prims somehow has uv parametrization (if not available,
    // alpha maping cannot be done, but they can return garbage so its OK)
    {pt.SurfaceParametrization(typename PrimType::Hit{})
    } -> std::same_as<Vector2>;
};

template <class PGType>
concept PrimitiveGroupC = requires()
{
    // Mandatory Types
    // Primitive type satisfies its concept (at least on default form)
    requires PrimC<typename PGType::template Primitive<>>;

    // SoA fashion primitive data. This will be used to access internal
    // of the primitive with a given an index
    typename PGType::DataSoA;
    std::is_same_v<typename PGType::DataSoA,
                   typename PGType::template Primitive<>::DataSoA>;
    // Hit Data, ray will temporarily holds this information when ray casting is resolved
    // Delegates this to work kernel, work kernel will generate differential surface using this,
    // PrimSoA and TransformId,
    typename PGType::Hit;
    requires std::is_same_v<typename PGType::Hit,
                            typename PGType::template Primitive<>::Hit>;

    // Transform context generator list of the primitive is used to
    // statically select a appropirate function for given primitive and transform
    //
    // TransformTypes (identity, single, multi, morph target etc.)
    //
    // Primitive itself can support multiple of these, but also it needs to inject
    // its own information to the transform. Thus; it generates a transform context.
    // For example, a rigid-body primitive batch can support only identity and single transform
    // since the batch does not have per-primitive varying transforms.
    //
    // Skinned meshes may support additional multi-transform type. Then first two probably
    // does not makes sense (mesh will be stuck on a T-pose) but it can be provided by the
    // primitive implementor for debugging etc.
    //
    // If a generator is not on this list, that transform can not act on this primitive.
    // More importantly, one generator per transform type can be available on this TypeList.
    // This is a limitation but the entire system is complex as it is so...
    //
    // You can write completely new primitive type for that generation.
    PGType::TransContextGeneratorList;

    // Can query the type
    {PGType::TypeName()} -> std::same_as<std::string_view>;

    //// TODO: Some Functions
};

// Support Concepts
template <class PrimType, class PrimGroupType, class Surface>
concept PrimWithSurfaceC = requires(PrimType mg,
                                    Surface& surface)
{
    // Base concept
    requires PrimC<PrimType>;

    // TODO: Ask on stackoverflow how to
    // constrain the function to thave a specific
    // signature to prevent partial updates
    // when non-derived function exists.
    {mg.GenerateSurface(surface,
                        typename PrimType::Hit{},
                        Ray{},
                        DiffRay{})
    } -> std::same_as<void>;

};

// Some common types
using PrimAttributeInfo = std::pair<PrimitiveAttributeLogic, MRayDataTypeRT>;
using PrimBatchId = uint32_t;
using PrimCount = std::pair<uint32_t, uint32_t>;
using PrimRange = std::pair<Vector2ui, Vector2ui>;
using PrimitiveRangeMap = std::unordered_map<PrimBatchId, PrimRange>;
using PrimitiveCountMap = std::unordered_map<PrimBatchId, PrimCount>;

class PrimitiveGroupI
{
    public:
    virtual ~PrimitiveGroupI() = default;

    virtual PrimBatchId         ReservePrimitiveBatch(uint32_t primitiveCount,
                                                      uint32_t attributeCount) = 0;
    virtual bool                RemoveReservation(PrimBatchId batchId) = 0;
    virtual void                CommitReservations() = 0;
    virtual bool                IsInCommitState() const = 0;
    virtual uint32_t            GetAttributeCount() const = 0;
    virtual PrimAttributeInfo   GetAttributeInfo(uint32_t attributeIndex) const = 0;
    virtual void                PushAttributeData(PrimBatchId batchId, uint32_t attributeIndex,
                                                  std::vector<Byte> data) = 0;
    virtual void                PushAttributeData(PrimBatchId batchId, uint32_t attributeIndex,
                                                  Vector2ui subBatchRange, std::vector<Byte> data) = 0;
    virtual size_t              GPUMemoryUsage() const = 0;

    //virtual const PrimitiveRangeMap& GetCommitted
};

// Intermediate class that handles memory management
// Using CRTP for errors etc.
template<class Child>
class PrimitiveGroup : public PrimitiveGroupI
{
    static constexpr size_t HashTableReserveSize = 64;
    // Device Memory Related
    static constexpr size_t AllocGranularity = 64_MiB;
    static constexpr size_t InitialReservation = 256_MiB;

    private:
    PrimBatchId         batchCounter;
    protected:
    const GPUSystem&    gpuSystem;

    PrimitiveRangeMap   batchRanges;
    PrimitiveCountMap   batchCounts;
    bool                isCommitted;
    DeviceMemory        memory;

    template <class... Args>
    std::tuple<Span<Args>...>   GenericCommit(std::array<bool, sizeof...(Args)> isAttributeList);

    public:
                                PrimitiveGroup(const GPUSystem&);

    PrimBatchId                 ReservePrimitiveBatch(uint32_t primitiveCount,
                                                      uint32_t attributeCount) override;
    virtual bool                RemoveReservation(PrimBatchId batchId) override;
    virtual bool                IsInCommitState() const override;
    virtual bool                GPUMemoryUsage() const override;
};

template<class Child>
template <class... Args>
std::tuple<Span<Args>...> PrimitiveGroup<Child>::GenericCommit(std::array<bool, sizeof...(Args)> isAttributeList)
{
    assert(batchRanges.empty());
    if(isCommitted)
    {
        MRAY_WARNING_LOG("{:s} is in committed state, "
                         " you cannot re-commit!", Child::TypeName());
        return std::tuple<Span<Args>...>{};
    }
    // Cacluate offsets
    Vector2ui offsets = Vector2ui::Zero();
    for(const auto& c : batchCounts)
    {
        std::pair range(Vector2ui(offsets[0], offsets[0] + c.second.first),
                        Vector2ui(offsets[1], offsets[1] + c.second.second));
        [[maybe_unused]]
        auto r = batchRanges.emplace(c.first, range);
        assert(r.second);

        offsets = Vector2ui(range.first[1], range.second[1]);
    }
    // Rename for clarity
    Vector2ui totalSize = offsets;

    // Generate offsets etc
    constexpr size_t TotalElements = sizeof...(Args);
    std::array<size_t, TotalElements> sizes;
    for(size_t i = 0; i < TotalElements; i++)
    {
        bool isAttribute = isAttributeList[i];
        sizes[i] = (isAttribute) ? totalSize[1] : totalSize[0];
    }

    std::tuple<Span<Args>...> result;
    MemAlloc::AllocateMultiData<DeviceMemory, Args...>(result, memory, sizes);
    isCommitted = true;
    return result;
}

template<class Child>
PrimitiveGroup<Child>::PrimitiveGroup(const GPUSystem& s)
    : gpuSystem(s)
    , batchCounter(0)
    , isCommitted(false)
    , memory(gpuSystem.AllGPUs(), AllocGranularity, InitialReservation)
{
    batchRanges.reserve(HashTableReserveSize);
    batchCounts.reserve(HashTableReserveSize);
}

template<class Child>
PrimBatchId PrimitiveGroup<Child>::ReservePrimitiveBatch(uint32_t primitiveCount,
                                                         uint32_t attributeCount)
{
    if(isCommitted)
    {
        MRAY_WARNING_LOG("{:s} is in committed state, "
                         " you change cannot change reservations!",
                         Child::TypeName());
        return std::numeric_limits<PrimBatchId>::max();
    }

    [[maybe_unused]]
    auto r = batchCounts.emplace(batchCounter, std::pair(primitiveCount, attributeCount));
    assert(r.second);
    batchCounter++;
    return batchCounter - 1;
}

template<class Child>
bool PrimitiveGroup<Child>::RemoveReservation(PrimBatchId batchId)
{
    if(isCommitted)
    {
        MRAY_WARNING_LOG("{:s} is in committed state, "
                         " you change cannot change reservations!",
                         Child::TypeName());
        return false;
    }

    bool isRemoved = static_cast<bool>(batchCounts.erase(batchId));
    if(!isRemoved)
    {
        MRAY_WARNING_LOG("{:s}: unable to remove reservation for batch {}!",
                         Child::TypeName(), batchId);
    }
    return isRemoved;
}

template<class Child>
bool PrimitiveGroup<Child>::IsInCommitState() const
{
    return isCommitted;
}

template<class Child>
bool PrimitiveGroup<Child>::GPUMemoryUsage() const
{
    return memory.Size();
}