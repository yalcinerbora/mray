#pragma once

#include <type_traits>
#include <tuple>
#include <cstdint>

#include "Core/MRayDataType.h"
#include "Core/MemAlloc.h"
#include "Core/TypeFinder.h"

#include "Device/GPUSystem.h"

#include "TracerTypes.h"
#include "TransformsDefault.h"
#include "GenericGroup.h"

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
    // Each primitive in a group will have different
    // and (maybe) multiple transforms.
    // This prevents the ray-casting to utilize a single local space
    // accelerator. Such primitive instantiations
    // will have unique accelerators.
    // As an example, skinned meshes and morph targets
    // will have such property.
    PER_PRIMITIVE_TRANSFORM
};

template<class Surface, class Prim, class Hit>
using SurfaceGenFunc = Surface(Prim::*)(const Hit&,
                                        const DiffRay&,
                                        const Ray&);

template<class TransformContext, class PrimData, class TransformData>
using TransformContextGenFunc = TransformContext(*)(const TransformData&,
                                                    const PrimData&,
                                                    TransformKey,
                                                    PrimitiveKey);

template <class PrimType, class TransformContext = typename PrimType::TransformContext>
concept PrimitiveC = requires(PrimType pt,
                              TransformContext tc,
                              RNGDispenser& rng,
                              Span<uint64_t> mortonCodes,
                              Span<Vector2us> voxelNormals)
{
    typename PrimType::Hit;
    typename PrimType::DataSoA;
    typename PrimType::TransformContext;
    typename PrimType::Intersection;

    // Has this specific constructor
    PrimType(tc, typename PrimType::DataSoA{}, PrimitiveKey{});

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

    {pt.GetTransformContext()} -> std::same_as<const typename PrimType::TransformContext&>;

    // Type traits
    requires std::is_trivially_copyable_v<PrimType>;
    requires std::is_trivially_destructible_v<PrimType>;
    requires std::is_move_assignable_v<PrimType>;
    requires std::is_move_constructible_v<PrimType>;
};

template <class PGType>
concept PrimitiveGroupC = requires(PGType pg, TransientData input)
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
    // PrimSoA and TransformKey,
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

    // Acquire SoA struct of this primitive group
    {pg.SoA()} -> std::same_as<typename PGType::DataSoA>;

    requires GenericGroupC<PGType>;
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

// Compile-time transform context generator finder
template <PrimitiveGroupC PrimGroup, TransformGroupC TransGroup>
constexpr auto AcquireTransformContextGenerator();

// Alias some stuff to easily acquire the function and context type
// Using macro instead of "static constexpr auto" since it make
// GPU link errors
#define MRAY_PRIM_TGEN_FUNCTION(PG, TG) \
    decltype(AcquireTransformContextGenerator<typename PG, TG>())::Function

template<PrimitiveGroupC PG, TransformGroupC TG>
using PrimTransformContextType = typename decltype(AcquireTransformContextGenerator<PG, TG>())::ReturnType;

class GenericGroupPrimitiveT : public GenericGroupT<PrimBatchKey, PrimAttributeInfo>
{
    public:
                GenericGroupPrimitiveT(uint32_t groupId,
                                       const GPUSystem& sys,
                                       size_t allocationGranularity = 16_MiB,
                                       size_t initialReservartionSize = 64_MiB);

    virtual Vector2ui   BatchRange(PrimBatchKey) const = 0;
};

using PrimGroupPtr           = std::unique_ptr<GenericGroupPrimitiveT>;

template<class Child>
class GenericGroupPrimitive : public GenericGroupPrimitiveT
{
    public:
                     GenericGroupPrimitive(uint32_t groupId,
                                           const GPUSystem& sys,
                                           size_t allocationGranularity = 16_MiB,
                                           size_t initialReservartionSize = 64_MiB);
    std::string_view Name() const override;
};

template<TransformContextC TransContextType = TransformContextIdentity>
class EmptyPrimitive
{
    public:
    using DataSoA           = EmptyType;
    using Hit               = EmptyType;
    using Intersection      = Optional<IntersectionT<EmptyType>>;
    using TransformContext  = TransContextType;

    private:
    Ref<const TransformContext> transformContext;

    public:
    constexpr               EmptyPrimitive(const TransformContext&,
                                           const DataSoA&, PrimitiveKey);
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
    const TransformContext& GetTransformContext() const;
};

class PrimGroupEmpty final : public GenericGroupPrimitive<PrimGroupEmpty>
{
    public:
    using DataSoA       = EmptyType;
    using Hit           = EmptyType;

    template <TransformContextC TContext = TransformContextIdentity>
    using Primitive = EmptyPrimitive<TContext>;

    // Transform Context Generators
    static constexpr auto TransContextGeneratorList = std::make_tuple
    (
        TypeFinder::KeyTFuncPair<TransformGroupIdentity,
                                 TransformContextIdentity,
                                 &GenTContextIdentity<DataSoA>>{},
        TypeFinder::KeyTFuncPair<TransformGroupSingle,
                                 TransformContextSingle,
                                 &GenTContextSingle<DataSoA>>{}
    );

    // The actual name of the type
    static std::string_view TypeName();
    static constexpr auto   TransformLogic = PrimTransformType::LOCALLY_CONSTANT_TRANSFORM;

                            PrimGroupEmpty(uint32_t primGroupId,
                                           const GPUSystem& sys);

    void                    CommitReservations() override;
    PrimAttributeInfoList   AttributeInfo() const override;
    void                    PushAttribute(PrimBatchKey batchId,
                                          uint32_t attributeIndex,
                                          TransientData data,
                                          const GPUQueue&) override;
    void                    PushAttribute(PrimBatchKey batchId,
                                          uint32_t attributeIndex,
                                          const Vector2ui& subBatchRange,
                                          TransientData data,
                                          const GPUQueue&) override;
    void                    PushAttribute(PrimBatchKey idStart, PrimBatchKey idEnd,
                                          uint32_t attributeIndex,
                                          TransientData data,
                                          const GPUQueue&) override;

    Vector2ui               BatchRange(PrimBatchKey) const override;

    DataSoA                 SoA() const;
};

static_assert(PrimitiveC<EmptyPrimitive<>>,
              "Empty primitive does not satisfy Primitive concept!");
static_assert(PrimitiveGroupC<PrimGroupEmpty>,
              "Empty primitive group does not satisfy PrimitiveGroup concept!");

template <PrimitiveGroupC PrimGroup, TransformGroupC TransGroup>
constexpr auto AcquireTransformContextGenerator()
{
    using namespace TypeFinder;
    constexpr auto FList = PrimGroup::TransContextGeneratorList;
    return GetTupleElement<TransGroup>(std::forward<decltype(FList)>(FList));
}

inline
GenericGroupPrimitiveT::GenericGroupPrimitiveT(uint32_t groupId,
                                               const GPUSystem& sys,
                                               size_t allocationGranularity,
                                               size_t initialReservartionSize)
    : GenericGroupT<PrimBatchKey, PrimAttributeInfo>(groupId, sys,
                                                     allocationGranularity,
                                                     initialReservartionSize)
{}

template<class C>
GenericGroupPrimitive<C>::GenericGroupPrimitive(uint32_t groupId,
                                                const GPUSystem& sys,
                                                size_t allocationGranularity,
                                                size_t initialReservartionSize)
    : GenericGroupPrimitiveT(groupId, sys,
                             allocationGranularity,
                             initialReservartionSize)
{}

template<class C>
std::string_view GenericGroupPrimitive<C>::Name() const
{
    return C::TypeName();
}

template<TransformContextC TC>
constexpr EmptyPrimitive<TC>::EmptyPrimitive(const TC& tc, const DataSoA&, PrimitiveKey)
    : transformContext(tc)
{}

template<TransformContextC TC>
constexpr Optional<IntersectionT<EmptyType>> EmptyPrimitive<TC>::Intersects(const Ray&, bool) const
{
    return std::nullopt;
}

template<TransformContextC TC>
constexpr SampleT<BasicSurface> EmptyPrimitive<TC>::SampleSurface(const RNGDispenser&) const
{
    return SampleT<BasicSurface>{};
}

template<TransformContextC TC>
constexpr Float EmptyPrimitive<TC>::PdfSurface(const Hit& hit) const
{
    return Float{std::numeric_limits<Float>::signaling_NaN()};
}

template<TransformContextC TC>
constexpr uint32_t EmptyPrimitive<TC>::SampleRNCount() const
{
    return 0;
}

template<TransformContextC TC>
constexpr Float EmptyPrimitive<TC>::GetSurfaceArea() const
{
    return std::numeric_limits<Float>::signaling_NaN();
}

template<TransformContextC TC>
constexpr AABB3 EmptyPrimitive<TC>::GetAABB() const
{
    return AABB3(Vector3(std::numeric_limits<Float>::signaling_NaN()),
                 Vector3(std::numeric_limits<Float>::signaling_NaN()));
}

template<TransformContextC TC>
constexpr Vector3 EmptyPrimitive<TC>::GetCenter() const
{
    return Vector3(std::numeric_limits<Float>::signaling_NaN());
}

template<TransformContextC TC>
constexpr uint32_t EmptyPrimitive<TC>::Voxelize(Span<uint64_t>& mortonCodes,
                                                Span<Vector2us>& normals,
                                                bool onlyCalculateSize,
                                                const VoxelizationParameters& voxelParams) const
{
    return 0;
}

template<TransformContextC TC>
constexpr Optional<BasicSurface> EmptyPrimitive<TC>::SurfaceFromHit(const Hit& hit) const
{
    return std::nullopt;
}

template<TransformContextC TC>
constexpr Optional<EmptyType> EmptyPrimitive<TC>::ProjectedHit(const Vector3& point) const
{
    return std::nullopt;
}

template<TransformContextC TC>
constexpr Vector2 EmptyPrimitive<TC>::SurfaceParametrization(const Hit& hit) const
{
    return Vector2::Zero();
}

template<TransformContextC TC>
const TC& EmptyPrimitive<TC>::GetTransformContext() const
{
    return transformContext;
}

inline
std::string_view PrimGroupEmpty::TypeName()
{
    using namespace std::literals;
    static std::string_view name = "(P)Empty"sv;
    return name;
}

inline
PrimGroupEmpty::PrimGroupEmpty(uint32_t primGroupId,
                               const GPUSystem& sys)
    : GenericGroupPrimitive(primGroupId, sys)
{}

inline
void PrimGroupEmpty::CommitReservations()
{
    isCommitted = true;
}

inline
PrimAttributeInfoList PrimGroupEmpty::AttributeInfo() const
{
    static const PrimAttributeInfoList LogicList;
    return LogicList;
}

inline
void PrimGroupEmpty::PushAttribute(PrimBatchKey, uint32_t,
                                   TransientData, const GPUQueue&)
{}

inline
void PrimGroupEmpty::PushAttribute(PrimBatchKey, uint32_t,
                                   const Vector2ui&,
                                   TransientData, const GPUQueue&)
{}

inline
void PrimGroupEmpty::PushAttribute(PrimBatchKey, PrimBatchKey,
                                   uint32_t, TransientData, const GPUQueue&)
{}

inline
Vector2ui PrimGroupEmpty::BatchRange(PrimBatchKey) const
{
    return Vector2ui::Zero();
}

inline
typename PrimGroupEmpty::DataSoA PrimGroupEmpty::SoA() const
{
    return EmptyType{};
}