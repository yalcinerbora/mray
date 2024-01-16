#pragma once

#include <limits>
#include <string_view>
#include <unordered_map>

#include "Core/TypeFinder.h"
#include "Core/MRayDataType.h"
#include "Core/Vector.h"
#include "Core/Ray.h"
#include "Core/Quaternion.h"
#include "Core/Matrix.h"
#include "Core/DataStructures.h"

#include "PrimitiveC.h"
#include "ShapeFunctions.h"
#include "GraphicsFunctions.h"
#include "Transforms.h"
#include "Random.h"

namespace DefaultTriangleDetail
{
    using LookupTable = StratifiedIntegerAliasTable<PrimitiveId::Type>;

    // SoA data of triangle group
    struct TriangleData
    {
        // Per vertex attributes
        Span<const Vector3>     positions;
        Span<const Quaternion>  tbnRotations;
        Span<const Vector2>     uvs;
        // Single-indexed vertices
        Span<const Vector3ui>   indexList;
    };

    using TriHit            = Vector2;
    using TriIntersection   = Optional<IntersectionT<TriHit>>;
    using TriLeaf           = DefaultLeaf;

    template<class TransContextType = IdentityTransformContext>
    class Triangle
    {
        public:
        using DataSoA           = TriangleData;
        using AcceleratorLeaf   = TriLeaf;
        using Hit               = TriHit;
        using Intersection      = TriIntersection;

        private:
        const TriangleData&         data;
        PrimitiveId                 id;
        Vector3                     positions[ShapeFunctions::Triangle::TRI_VERTEX_COUNT];
        const TransContextType&     transformContext;


        public:
        MRAY_HYBRID         Triangle(const TransContextType& transform,
                                     const TriangleData& data, PrimitiveId id);

        MRAY_HYBRID
        Intersection        Intersects(const Ray& ray) const;

        MRAY_HYBRID
        SampleT<Vector3>    SamplePosition(const RNGDispenser& rng) const;

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
        Vector2             SurfaceParametrization(const Hit& hit) const;

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
        Span<const UNorm4x8>     skinWeights;
        Span<const Vector4uc>    skinIndices;
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

class PrimGroupTriangle : public PrimitiveGroup<PrimGroupTriangle>
{
    public:
    using DataSoA       = DefaultTriangleDetail::TriangleData;
    using Hit           = typename DefaultTriangleDetail::TriHit;

    template <class TContext = IdentityTransformContext>
    using Primitive = DefaultTriangleDetail:: template Triangle<TContext>;

    // Transform Context Generators
    static constexpr auto TransContextGeneratorList = std::make_tuple
    (
        TypeFinder::KeyTFuncPair<IdentityTransformGroup,
                                 IdentityTransformContext,
                                 &GenTContextIdentity<DataSoA>>{},
        TypeFinder::KeyTFuncPair<SingleTransformGroup,
                                 SingleTransformContext,
                                 &GenTContextSingle<DataSoA>>{}
    );
    // The actual name of the type
    static std::string_view TypeName();
    static constexpr size_t AttributeCount = 4;

    private:
    Span<Vector3>       dPositions;
    Span<Quaternion>    dTBNRotations;
    Span<Vector2>       dUVs;
    Span<Vector3ui>     dIndexList;
    DataSoA             soa;

    public:

    void                    CommitReservations() override;
    uint32_t                GetAttributeCount() const override;
    PrimAttributeInfo       GetAttributeInfo(uint32_t attributeIndex) const override;
    void                    PushAttributeData(uint32_t batchId, uint32_t attributeIndex,
                                              std::vector<Byte> data) override;
    void                    PushAttributeData(PrimBatchId batchId, uint32_t attributeIndex,
                                              Vector2ui subBatchRange, std::vector<Byte> data) override;
};

class PrimGroupSkinnedTriangle : public PrimitiveGroup<PrimGroupSkinnedTriangle>
{
    public:
    using DataSoA       = DefaultSkinnedTriangleDetail::SkinnedTriangleData;
    using Hit           = typename DefaultTriangleDetail::TriHit;

    template <class TContext = IdentityTransformContext>
    using Primitive     = DefaultTriangleDetail:: template Triangle<TContext>;

    // Transform Context Generators
    static constexpr auto TransContextGeneratorList = std::make_tuple
    (
        TypeFinder::KeyTFuncPair<IdentityTransformGroup,
                                 IdentityTransformContext,
                                 &GenTContextIdentity<DataSoA>>{},
        TypeFinder::KeyTFuncPair<SingleTransformGroup,
                                 SingleTransformContext,
                                 &GenTContextSingle<DataSoA>>{},
        TypeFinder::KeyTFuncPair<MultiTransformGroup,
                                 DefaultSkinnedTriangleDetail::SkinnedTransformContext,
                                 &DefaultSkinnedTriangleDetail::GenTContextSkinned>{}
    );
    // Actual Name of the Type
    static std::string_view TypeName();
    static constexpr size_t AttributeCount = 6;

    private:
    Span<Vector3>       dPositions;
    Span<Quaternion>    dTBNRotations;
    Span<Vector2>       dUVs;
    Span<Vector3ui>     dIndexList;
    Span<UNorm4x8>      dSkinWeights;
    Span<Vector4uc>     dSkinIndices;
    DataSoA             soa;

    public:
    //

    void                    CommitReservations() override;
    uint32_t                GetAttributeCount() const override;
    PrimAttributeInfo       GetAttributeInfo(uint32_t attributeIndex) const override;
    void                    PushAttributeData(uint32_t batchId, uint32_t attributeIndex,
                                              std::vector<Byte> data) override;
    void                    PushAttributeData(PrimBatchId batchId, uint32_t attributeIndex,
                                              Vector2ui subBatchRange, std::vector<Byte> data) override;
};

#include "PrimitiveDefaultTriangle.hpp"

static_assert(PrimitiveGroupC<PrimGroupTriangle>);
static_assert(PrimitiveGroupC<PrimGroupSkinnedTriangle>);