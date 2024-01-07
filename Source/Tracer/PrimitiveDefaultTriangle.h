#pragma once

#include <limits>

#include "Core/TypeFinder.h"
#include "Core/Vector.h"
#include "Core/Ray.h"
#include "Core/Quaternion.h"
#include "Core/Matrix.h"
#include "Core/DataStructures.h"

#include "PrimitiveC.h"
#include "ShapeFunctions.h"
#include "GraphicsFunctions.h"
#include "Transforms.h"

namespace DefaultTriangleDetail
{
    using LookupTable = StratifiedIntegerAliasTable<PrimitiveId::Type>;

    // SoA data of triangle group
    struct TriangleData
    {
        // Per sub-batch attributes
        LookupTable         subBatchTable;
        const bool*         cullFace;

        // Per vertex attributes
        const Vector3*      positions;
        const Quaternion*   tbnRotations;
        const Vector2*      uvs;
        // Single-indexed vertices
        const PrimitiveId*  indexList;
    };

    using TriHit            = Vector2;
    using TriIntersection   = Optional<Intersection<TriHit>>;
    using TriLeaf           = DefaultLeaf;

    template<class TransContextType = IdentityTransformContext>
    class Triangle
    {
        private:
        const TriangleData&         data;
        PrimitiveId                 id;
        Vector3                     positions[ShapeFunctions::Triangle::TRI_VERTEX_COUNT];

        const TransContextType&    transformContext;

        public:
        using AcceleratorLeaf   = TriLeaf;
        using Hit               = TriHit;
        using IntersectionT     = TriIntersection;

        MRAY_HYBRID
                            Triangle(const TransContextType& transform,
                                     const TriangleData& data, PrimitiveId id);

        MRAY_HYBRID
        AcceleratorLeaf     GenerateLeaf() const;

        MRAY_HYBRID
        IntersectionT       Intersects(const Ray& ray,
                                       const AcceleratorLeaf& leaf) const;

        MRAY_HYBRID
        Sample<Vector3>     SamplePosition(const RNGDispenser& rng) const;

        MRAY_HYBRID
        Float               PdfPosition(const Vector3& position) const;

        MRAY_HYBRID
        uint32_t            SampleRNCount() const;

        MRAY_HYBRID
        Float               GetSurfaceArea() const;

        MRAY_HYBRID
        AABB3               GetAABB() const;

        MRAY_HYBRID
        Vector3             GetCenter() const;

        MRAY_HYBRID
        uint32_t            Voxelize(Span<uint64_t>& mortonCodes,
                                     Span<Vector2us>& normals,
                                     bool onlyCalculateSize,
                                     const VoxelizationParameters& voxelParams) const;

        MRAY_HYBRID
        Vector2             SurfaceParametrization(const Hit& hit,
                                                   const AcceleratorLeaf& leaf) const;

        // Surface Generation
        MRAY_HYBRID
        void                GenerateSurface(EmptySurface&,
                                            // Inputs
                                            const Hit&,
                                            const Ray&,
                                            const DiffRay&) const;

        MRAY_HYBRID
        void                GenerateSurface(BasicSurface&,
                                            // Inputs
                                            const Hit&,
                                            const Ray&,
                                            const DiffRay&) const;

        MRAY_HYBRID
        void                GenerateSurface(BarycentricSurface&,
                                            // Inputs
                                            const Hit&,
                                            const Ray&,
                                            const DiffRay&) const;

        MRAY_HYBRID
        void                GenerateSurface(DefaultSurface&,
                                            // Inputs
                                            const Hit&,
                                            const Ray&,
                                            const DiffRay&) const;
    };

}

namespace DefaultSkinnedTriangleDetail
{
    // SoA data of triangle group
    struct SkinnedTriangleData : public DefaultTriangleDetail::TriangleData
    {
        // Extra per-primitive attributes
        const UNorm4x8*     skinWeights;
        const Vector4uc*    skinIndices;
    };

    // Default skinned triangle has 4 transform per primitive
    static constexpr int TRANSFORM_PER_PRIMITIVE = 4;

    using TriHit            = DefaultTriangleDetail::TriHit;
    using TriIntersection   = DefaultTriangleDetail::TriIntersection;
    using TriLeaf           = DefaultTriangleDetail::TriLeaf;

    template<class TransContextType = IdentityTransformContext>
    using SkinnedTriangle   = DefaultTriangleDetail::Triangle<TransContextType>;

    struct SkinnedTransformContext
    {
        private:
        // These are generated on the fly (so no reference)
        Matrix4x4 transform;
        Matrix4x4 invTransform;

        public:
        MRAY_HYBRID
        SkinnedTransformContext(const typename MultiTransformGroup::DataSoA& transformData,
                                const SkinnedTriangleData& triData,
                                TransformId tId,
                                PrimitiveId pId);

        Vector3 ApplyP(const Vector3& point) const
        {
            return point;
        }

        Vector3 ApplyV(const Vector3& vec) const
        {
            return vec;
        }

        Vector3 ApplyN(const Vector3& norm) const
        {
            return norm;
        }

        AABB3 Apply(const AABB3& aabb) const
        {
            return aabb;
        }

        Ray Apply(const Ray& ray) const
        {
            return ray;
        }

        Vector3 InvApplyP(const Vector3& point) const
        {
            return point;
        }

        Vector3 InvApplyV(const Vector3& vec) const
        {
            return vec;
        }

        Vector3 InvApplyN(const Vector3& norm) const
        {
            return norm;
        }

        AABB3 InvApply(const AABB3& aabb) const
        {
            return aabb;
        }

        Ray InvApply(const Ray& ray) const
        {
            return ray;
        }
    };

    // Transform Context Generators
    MRAY_HYBRID
    SkinnedTransformContext GenTContextSkinned(const typename MultiTransformGroup::DataSoA&,
                                               const SkinnedTriangleData&,
                                               TransformId,
                                               PrimitiveId);

    static_assert(TransformContextC<SkinnedTransformContext>);
}

struct PrimGroupTriangle
{
    using DataSoA = DefaultTriangleDetail::TriangleData;

    template <class TContext = IdentityTransformContext>
    using Primitive = DefaultTriangleDetail:: template Triangle<TContext>;
    // Hit info is barycentrics for triangle
    using Hit = typename DefaultTriangleDetail::TriHit;
    using AcceleratorLeaf = typename DefaultTriangleDetail::TriLeaf;

    // Transform Context Generators
    static constexpr auto TransContextGeneratorList = std::make_tuple
    (
        TypeFinder::KeyFuncT<IdentityTransformGroup,
                             IdentityTransformContext,
                             &GenTContextIdentity<DataSoA>>{},
        TypeFinder::KeyFuncT<SingleTransformGroup,
                             SingleTransformContext,
                             &GenTContextSingle<DataSoA>>{}
    );

    //// Surface Generators
    //static constexpr auto SurfaceGeneratorList = std::make_tuple
    //(
    //    TypeFinderFinder::KeyFuncT<EmptySurface, & GenTContextIdentity<DataSoA>>{},
    //    TypeFinderFinder::KeyFuncT<BasicSurface, & GenTContextSingle<DataSoA>>{},
    //    TypeFinderFinder::KeyFuncT<BarycentricSurface, &GenTContextSingle<DataSoA>>{},
    //    TypeFinderFinder::KeyFuncT<DefaultSurface, &GenTContextSingle<DataSoA>>{}
    //);
};

struct PrimGroupSkinnedTriangle
{
    using DataSoA = DefaultSkinnedTriangleDetail::SkinnedTriangleData;

    template <class TContext = IdentityTransformContext>
    using Primitive = DefaultTriangleDetail:: template Triangle<TContext>;
    // Hit info is barycentrics for triangle
    using Hit = typename DefaultTriangleDetail::TriHit;
    using AcceleratorLeaf = typename DefaultTriangleDetail::TriLeaf;

    // Transform Context Generators
    static constexpr auto TransContextGeneratorList = std::make_tuple
    (
        TypeFinder::KeyFuncT<IdentityTransformGroup,
                             IdentityTransformContext,
                             &GenTContextIdentity<DataSoA>>{},
        TypeFinder::KeyFuncT<SingleTransformGroup,
                             SingleTransformContext,
                             &GenTContextSingle<DataSoA>>{},
        TypeFinder::KeyFuncT<MultiTransformGroup,
                             DefaultSkinnedTriangleDetail::SkinnedTransformContext,
                             &DefaultSkinnedTriangleDetail::GenTContextSkinned>{}
    );
};

#include "PrimitiveDefaultTriangle.hpp"

static_assert(PrimitiveGroupC<PrimGroupTriangle>);
//static_assert(PrimitiveGroupC<PrimGroupSkinnedTriangle>);