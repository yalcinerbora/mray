#pragma once

#include <type_traits>
#include <tuple>

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
    CONSTANT_LOCAL_TRANSFORM,
    // Each primitive in a group will different (maybe)
    // and multiple transforms.
    // This prevents the ray-casting to utilize a single local space
    // accelerator. Such primitive instantiations
    // will have unique accelerators.
    // As an example, skinned meshes and morph targets
    // will have such property.
    PER_PRIMITIVE_TRANSFORM
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

template <class PrimType, class PrimGroupType>
concept PrimC = requires(PrimType pt)
{
    typename PrimType::AcceleratorLeaf;
    typename PrimType::Hit;

    // Has this specific constructor
    requires requires(const IdentityTransformContext& tc,
                      const typename PrimGroupType::DataSoA & soaData)
    {PrimType(tc, soaData, PrimitiveId{});};


    // Mandatory member functions
    // Generate a accelerator leaf with this key
    // Default implementations probably just writes key and primitive id
    {pt.GenerateLeaf()
    } -> std::same_as<typename PrimType::AcceleratorLeaf>;

    // Intersect function
    requires requires(Float f, typename PrimType::Hit h)
    {
        {pt.Intersects(Ray{},
                       typename PrimType::AcceleratorLeaf{})
        } -> std::same_as<Optional<Intersection<typename PrimType::Hit>>>;
    };

    // Parametrized sampling
    requires requires(Float f, Vector3 v3)
    {
        // Sample a position over the parametrized surface
        {pt.SamplePosition(RNGDispenser{})
        } -> std::same_as<Sample<Vector3>>;

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

    // Accelerator "AnyHit" requirements
    // I
    // TODO: **Design**
    //
    // Design 1 attach "map" alpha/displacement
    // to the primitive (sub-primitive-bath)
    // Good: Prims will not require to have uv parametrization
    //
    // Bad: Same primitive having/not having
    //   alpha map would require a data duplication
    //
    //      Sthocastic parametrization (rng) is mandated
    //  what if user had different technique?
    //
    //      What about displacement maps?
    //
    // Design 2 is better, I think...
    //{pt.AlphaTest(typename PrimGroupType::Hit{},
    //              typename PrimGroupType::AcceleratorLeaf{},
    //              Float{})
    //} -> std::same_as<bool>;

    // Design 2 Acquire primitives surface parametrization
    // (Aka. uv map)
    // Good: No data duplication no prinicpiled data for alpha testing
    //
    // Bad: All prims somehow has uv parametrization (if not available,
    // alpha maping cannot be done, but they can return garbage so its OK)
    //
    //      Only 2D surface parametrization?
    {pt.SurfaceParametrization(typename PrimGroupType::Hit{},
                               typename PrimGroupType::AcceleratorLeaf{})
    } -> std::same_as<Vector2>;

    // Still we will do these tests on anyhit shader (given hw acceleration)
    // how to attach maps to any hit shader?
};

template <class PGType>
concept PrimitiveGroupC = requires()
{
    // Mandatory Types
    // Primitive type satisfies its concept (at least on default form)
    requires PrimC<typename PGType::template Primitive<>, PGType>;

    // SoA fashion primitive data. This will be used to access internal
    // of the primitive with a given an index
    typename PGType::DataSoA;
    // Hit Data, ray will temporarily holds this information when ray casting is resolved
    // Delegates this to work kernel, work kernel will generate differential surface using this,
    // PrimSoA and TransformId,
    typename PGType::Hit;
    requires std::is_same_v<typename PGType::Hit,
                            typename PGType::template Primitive<>::Hit>;
    // What should this primitive's accelerator hold on its "leafs". Most of the time this is default
    // type (work key and primitive id is present on the leaves).
    typename PGType::AcceleratorLeaf;
    requires std::is_same_v<typename PGType::AcceleratorLeaf,
                            typename PGType::template Primitive<>::AcceleratorLeaf>;
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

    //// List of differential surface
    //// generator functions that this primitive can generate.
    //// A material will require one of these.
    //PGType::SurfaceGeneratorList;

    //// TODO: Some Functions
};

// Support Concepts
template <class PrimType, class PrimGroupType, class Surface>
concept PrimWithSurfaceC = requires(PrimType mg,
                                    Surface& surface)
{
    // Base concept
    requires PrimC<PrimType, PrimGroupType>;

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