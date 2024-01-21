#pragma once

#include <type_traits>
#include <tuple>
#include <unordered_map>

#include "Core/MRayDataType.h"
#include "Core/MemAlloc.h"
#include "Device/GPUSystem.h"

#include "TracerTypes.h"
#include "TracerInterface.h"
#include "Transforms.h"

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
    WEIGHT_INDEX,

    END
};

struct PrimAttributeConverter
{
    using enum PrimitiveAttributeLogic;
    static constexpr std::array<std::string_view, static_cast<size_t>(END)> Names =
    {
        "Position",
        "Index",
        "Normal",
        "Radius",
        "Tangent",
        "UV0",
        "UV1",
        "Weight",
        "Weight Index"
    };
    static constexpr std::string_view           ToString(PrimitiveAttributeLogic e);
    static constexpr PrimitiveAttributeLogic    FromString(std::string_view e);
};

constexpr std::string_view PrimAttributeConverter::ToString(PrimitiveAttributeLogic e)
{
    return Names[static_cast<uint32_t>(e)];
}

constexpr PrimitiveAttributeLogic PrimAttributeConverter::FromString(std::string_view sv)
{
    using IntType = std::underlying_type_t<PrimitiveAttributeLogic>;
    IntType i = 0;
    for(const std::string_view& checkSV : Names)
    {
        if(checkSV == sv) return PrimitiveAttributeLogic(i);
        i++;
    }
    return PrimitiveAttributeLogic(END);
}

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
concept PrimitiveC = requires(PrimType pt, RNGDispenser& rng,
                              Span<uint64_t> mortonCodes,
                              Span<Vector2us> voxelNormals)
{
    typename PrimType::Hit;
    typename PrimType::DataSoA;

    // Has this specific constructor
    //requires requires(const TransformContextIdentity& tc,
    //                  const typename PrimType::DataSoA& soaData)
    PrimType(TransformContextIdentity{},
             typename PrimType::DataSoA{}, PrimitiveId{});

    // Intersect function
    {pt.Intersects(Ray{}, bool{})
    } -> std::same_as<Optional<IntersectionT<typename PrimType::Hit>>>;

    // Parametrized sampling
    // Sample a position over the parametrized surface
    {pt.SampleSurface(rng)
    } -> std::same_as<SampleT<BasicSurface>>;

    // PDF of sampling such position
    // Conservative towards non-zero values
    // primitive may try to resolve position is on surface
    // but inherently assumes position is on surface
    {pt.PdfSurface(typename PrimType::Hit{})
    } -> std::same_as<Float>;

    {pt.SampleRNCount()
    } -> std::same_as<uint32_t>;

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
    {pt.Voxelize(mortonCodes, voxelNormals, bool{},
                 VoxelizationParameters{})
    } -> std::same_as<uint32_t>;

    // Generate a basic surface from hit (utilized by light sampler)
    {pt.SurfaceFromHit(typename PrimType::Hit{})
    } ->std::same_as<Optional<BasicSurface>>;

    // Project a closeby surface and find the hit parameters
    {pt.ProjectedHit(Vector3{})
    } -> std::same_as<Optional<typename PrimType::Hit>>;

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
concept PrimitiveGroupC = requires(PGType pg)
{
    // Mandatory Types
    // Primitive type satisfies its concept (at least on default form)
    requires PrimitiveC<typename PGType::template Primitive<>>;

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

    // Compile-time constant of transform logic
    PGType::TransformLogic;

    // Can query the type
    {PGType::TypeName()} -> std::same_as<std::string_view>;

    {pg.ReservePrimitiveBatch(PrimCount{})} -> std::same_as<PrimBatchId>;
    {pg.CommitReservations()} -> std::same_as<void>;
    {pg.IsInCommitState()} -> std::same_as<bool>;
    {pg.AttributeInfo()} -> std::same_as<PrimAttributeInfoList>;
    {pg.PushAttribute(PrimBatchId{}, uint32_t{},
                      std::vector<Byte>{})} -> std::same_as<void>;
    {pg.PushAttribute(PrimBatchId{}, uint32_t{}, Vector2ui{},
                      std::vector<Byte>{})} ->std::same_as<void>;
    {pg.GPUMemoryUsage()} -> std::same_as<size_t>;
};

// Support Concepts
template <class PrimType, class PrimGroupType, class Surface>
concept PrimitiveWithSurfaceC = requires(PrimType mg,
                                    Surface& surface)
{
    // Base concept
    requires PrimitiveC<PrimType>;

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
struct PrimRange { Vector2ui primRange; Vector2ui attributeRange; };

using BatchIdType = typename PrimLocalBatchId::T;

using PrimitiveRangeMap = std::unordered_map<BatchIdType, PrimRange>;
using PrimitiveCountMap = std::unordered_map<BatchIdType, PrimCount>;

class PrimitiveGroupI
{
    public:
    virtual ~PrimitiveGroupI() = default;

    virtual PrimBatchId             ReservePrimitiveBatch(PrimCount) = 0;
    virtual void                    CommitReservations() = 0;
    virtual bool                    IsInCommitState() const = 0;
    virtual PrimAttributeInfoList   AttributeInfo() const = 0;
    virtual void                    PushAttribute(PrimBatchId batchId, uint32_t attributeIndex,
                                                  std::vector<Byte> data) = 0;
    virtual void                    PushAttribute(PrimBatchId batchId, uint32_t attributeIndex,
                                                  const Vector2ui& subBatchRange,
                                                  std::vector<Byte> data) = 0;
    virtual size_t                  GPUMemoryUsage() const = 0;
};

// Intermediate class that handles memory management
// Using CRTP for errors etc.
template<class Child>
class PrimitiveGroup : public PrimitiveGroupI
{
    static constexpr size_t     HashTableReserveSize = 64;
    // Device Memory Related
    static constexpr size_t     AllocGranularity = 64_MiB;
    static constexpr size_t     InitialReservation = 256_MiB;

    private:
    uint32_t            batchCounter;
    protected:
    const GPUSystem&    gpuSystem;

    PrimitiveRangeMap   batchRanges;
    PrimitiveCountMap   batchCounts;
    bool                isCommitted;
    DeviceMemory        deviceMem;
    uint32_t            primGroupId;

    template <class... Args>
    Tuple<Span<Args>...>        GenericCommit(std::array<bool, sizeof...(Args)> isAttributeList);

    template <class T>
    void                        GenericPushData(PrimBatchId batchId,
                                                const Span<T>& primData,
                                                std::vector<Byte> data,
                                                bool isPerPrimitive) const;
    template <class T>
    void                        GenericPushData(PrimBatchId batchId,
                                                const Span<T>& primData,
                                                const Vector2ui& subBatchRange,
                                                std::vector<Byte> data,
                                                bool isPerPrimitive) const;


    public:
                                PrimitiveGroup(uint32_t primGroupId, const GPUSystem&);

    PrimBatchId                 ReservePrimitiveBatch(PrimCount) override;
    virtual bool                IsInCommitState() const override;
    virtual size_t              GPUMemoryUsage() const override;
};

template<class TransContextType = TransformContextIdentity>
class EmptyPrimitive
{
    public:
    using DataSoA           = EmptyType;
    using Hit               = EmptyType;
    using Intersection      = Optional<IntersectionT<EmptyType>>;

    constexpr               EmptyPrimitive(const TransContextType&,
                                           const DataSoA&, PrimitiveId);
    constexpr Intersection  Intersects(const Ray&, bool) const;
    constexpr
    SampleT<BasicSurface>   SampleSurface(const RNGDispenser&) const;
    constexpr Float         PdfSurface(const Hit& hit) const;
    constexpr uint32_t      SampleRNCount() const;
    constexpr Float         GetSurfaceArea() const;
    constexpr AABB3         GetAABB() const;
    constexpr Vector3       GetCenter() const;
    constexpr uint32_t      Voxelize(Span<uint64_t>& mortonCodes,
                                     Span<Vector2us>& normals,
                                     bool onlyCalculateSize,
                                     const VoxelizationParameters& voxelParams) const;
    constexpr
    Optional<BasicSurface>  SurfaceFromHit(const Hit& hit) const;
    constexpr Optional<Hit> ProjectedHit(const Vector3& point) const;
    constexpr Vector2       SurfaceParametrization(const Hit& hit) const;
};

static_assert(PrimitiveC<EmptyPrimitive<>>,
              "Empty primitive does not satisfy Primitive concept!");

template<class Child>
template <class... Args>
Tuple<Span<Args>...> PrimitiveGroup<Child>::GenericCommit(std::array<bool, sizeof...(Args)> isPerPrimitiveList)
{
    assert(batchRanges.empty());
    if(isCommitted)
    {
        MRAY_WARNING_LOG("{:s} is in committed state, "
                         " you cannot re-commit!", Child::TypeName());
        return Tuple<Span<Args>...>{};
    }
    // Cacluate offsets
    Vector2ui offsets = Vector2ui::Zero();
    for(const auto& c : batchCounts)
    {
        PrimRange range
        {
            .primRange = Vector2ui(offsets[0], offsets[0] + c.second.primCount),
            .attributeRange = Vector2ui(offsets[1], offsets[1] + c.second.attributeCount)
        };
        [[maybe_unused]]
        auto r = batchRanges.emplace(c.first, range);
        assert(r.second);

        offsets = Vector2ui(range.primRange[1], range.attributeRange[1]);
    }
    // Rename for clarity
    Vector2ui totalSize = offsets;

    // Generate offsets etc
    constexpr size_t TotalElements = sizeof...(Args);
    std::array<size_t, TotalElements> sizes;
    for(size_t i = 0; i < TotalElements; i++)
    {
        bool isPerPrimitive = isPerPrimitiveList[i];
        sizes[i] = (isPerPrimitive) ? totalSize[0] : totalSize[1];
    }

    Tuple<Span<Args>...> result;
    MemAlloc::AllocateMultiData<DeviceMemory, Args...>(result, deviceMem, sizes);
    isCommitted = true;
    return result;
}

template<class Child>
template <class T>
void PrimitiveGroup<Child>::GenericPushData(PrimBatchId batchId,
                                            const Span<T>& primData,
                                            std::vector<Byte> data,
                                            bool isPerPrimitive) const
{
    // TODO: parallel issue maybe?
    // TODO: utilize multi device maybe
    const GPUQueue& deviceQueue = gpuSystem.BestDevice().GetQueue(0);

    const auto it = batchRanges.find(static_cast<BatchIdType>(batchId.localBatchId));
    Vector2ui attribRange = (isPerPrimitive)
                                ? it->second.primRange
                                : it->second.attributeRange;
    size_t count = attribRange[1] - attribRange[0];
    Span<T> dSubBatch = primData.subspan(attribRange[0], count);
    deviceQueue.MemcpyAsync(dSubBatch, ToSpan<const T>(data));
    deviceQueue.IssueBufferForDestruction(std::move(data));
}

template<class Child>
template <class T>
void PrimitiveGroup<Child>::GenericPushData(PrimBatchId batchId,
                                            const Span<T>& primData,
                                            const Vector2ui& subBatchRange,
                                            std::vector<Byte> data,
                                            bool isPerPrimitive) const
{
    // TODO: parallel issue maybe?
    // TODO: utilize multi device maybe
    const GPUQueue& deviceQueue = gpuSystem.BestDevice().GetQueue(0);

    const auto it = batchRanges.find(static_cast<BatchIdType>(batchId.localBatchId));
    Vector2ui attribRange = (isPerPrimitive)
                                ? it->second.primRange
                                : it->second.attributeRange;
    size_t count = attribRange[1] - attribRange[0];
    Span<T> dSubBatch = primData.subspan(attribRange[0], count);
    size_t subCount = subBatchRange[1] - subBatchRange[0];
    Span<T> dSubSubBatch = dSubBatch.subspan(subBatchRange[0], subCount);

    deviceQueue.MemcpyAsync(dSubBatch, ToSpan<const T>(data));
    deviceQueue.IssueBufferForDestruction(std::move(data));
}

template<class Child>
PrimitiveGroup<Child>::PrimitiveGroup(uint32_t primGroupId, const GPUSystem& s)
    : gpuSystem(s)
    , batchCounter(0)
    , isCommitted(false)
    , primGroupId(primGroupId)
    , deviceMem(s.AllGPUs(), AllocGranularity, InitialReservation, true)
{
    batchRanges.reserve(HashTableReserveSize);
    batchCounts.reserve(HashTableReserveSize);
}

template<class Child>
PrimBatchId PrimitiveGroup<Child>::ReservePrimitiveBatch(PrimCount primCount)
{
    if(isCommitted)
    {
        MRAY_WARNING_LOG("{:s} is in committed state, "
                         " you change cannot change reservations!",
                         Child::TypeName());
        return std::numeric_limits<PrimBatchId>::max();
    }
    [[maybe_unused]]
    auto r = batchCounts.emplace(batchCounter, primCount);
    assert(r.second);
    batchCounter++;

    return PrimBatchId
    {
        .primGroupId = PrimGroupId{primGroupId},
        .localBatchId = PrimLocalBatchId{batchCounter - 1}
    };
}

template<class Child>
bool PrimitiveGroup<Child>::IsInCommitState() const
{
    return isCommitted;
}

template<class Child>
size_t PrimitiveGroup<Child>::GPUMemoryUsage() const
{
    return deviceMem.Size();
}

template<class TC>
constexpr EmptyPrimitive<TC>::EmptyPrimitive(const TC&, const DataSoA&, PrimitiveId)
{}

template<class TC>
constexpr Optional<IntersectionT<EmptyType>> EmptyPrimitive<TC>::Intersects(const Ray&, bool) const
{
    return std::nullopt;
}

template<class TC>
constexpr SampleT<BasicSurface> EmptyPrimitive<TC>::SampleSurface(const RNGDispenser&) const
{
    return SampleT<BasicSurface>{};
}

template<class TC>
constexpr Float EmptyPrimitive<TC>::PdfSurface(const Hit& hit) const
{
    return Float{std::numeric_limits<Float>::signaling_NaN()};
}

template<class TC>
constexpr uint32_t EmptyPrimitive<TC>::SampleRNCount() const
{
    return 0;
}

template<class TC>
constexpr Float EmptyPrimitive<TC>::GetSurfaceArea() const
{
    return std::numeric_limits<Float>::signaling_NaN();
}

template<class TC>
constexpr AABB3 EmptyPrimitive<TC>::GetAABB() const
{
    return AABB3(Vector3(std::numeric_limits<Float>::signaling_NaN()),
                 Vector3(std::numeric_limits<Float>::signaling_NaN()));
}

template<class TC>
constexpr Vector3 EmptyPrimitive<TC>::GetCenter() const
{
    return Vector3(std::numeric_limits<Float>::signaling_NaN());
}

template<class TC>
constexpr uint32_t EmptyPrimitive<TC>::Voxelize(Span<uint64_t>& mortonCodes,
                                                Span<Vector2us>& normals,
                                                bool onlyCalculateSize,
                                                const VoxelizationParameters& voxelParams) const
{
    return 0;
}

template<class TC>
constexpr Optional<BasicSurface> EmptyPrimitive<TC>::SurfaceFromHit(const Hit& hit) const
{
    return std::nullopt;
}

template<class TC>
constexpr Optional<EmptyType> EmptyPrimitive<TC>::ProjectedHit(const Vector3& point) const
{
    return std::nullopt;
}

template<class TC>
constexpr Vector2 EmptyPrimitive<TC>::SurfaceParametrization(const Hit& hit) const
{
    return Vector2::Zero();
}