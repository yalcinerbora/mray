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
#include "Core/GraphicsFunctions.h"
#include "Core/TypeNameGenerators.h"

#include "PrimitiveC.h"
#include "ShapeFunctions.h"
#include "TransformsDefault.h"
#include "Random.h"

namespace DefaultTriangleDetail
{
    constexpr size_t DeviceMemAllocationGranularity = 32_MiB;
    constexpr size_t DeviceMemReservationSize = 256_MiB;

    using LookupTable = StratifiedIntegerAliasTable<PrimitiveKey::Type>;

    // SoA data of triangle group
    struct alignas(64) TriangleData
    {
        // Per vertex attributes
        Span<const Vector3>     positions;
        Span<const Quaternion>  tbnRotations;
        Span<const Vector2>     uvs;
        // Single-indexed vertices
        Span<const Vector3ui>   indexList;
    };

    using TriHit            = Vector2;
    using TriIntersection   = IntersectionT<TriHit>;

    template<TransformContextC TransContextType = TransformContextIdentity>
    class Triangle
    {
        public:
        using DataSoA           = TriangleData;
        using Hit               = TriHit;
        using Intersection      = Optional<TriIntersection>;
        using TransformContext  = TransContextType;

        private:
        Ref<const TriangleData>     data;
        Ref<const TransContextType> transformContext;
        PrimitiveKey                key;
        Vector3                     positions[ShapeFunctions::Triangle::TRI_VERTEX_COUNT];

        public:
        MRAY_HYBRID             Triangle(const TransContextType& transform,
                                         const TriangleData& data, PrimitiveKey key);
        MRAY_HYBRID
        Intersection            Intersects(const Ray& ray, bool cullBackface) const;
        MRAY_HYBRID
        SampleT<BasicSurface>   SampleSurface(RNGDispenser& rng) const;
        MRAY_HYBRID Float       PdfSurface(const Hit& hit) const;
        MRAY_HYBRID uint32_t    SampleRNCount() const;
        MRAY_HYBRID Float       GetSurfaceArea() const;
        MRAY_HYBRID AABB3       GetAABB() const;
        MRAY_HYBRID Vector3     GetCenter() const;
        MRAY_HYBRID uint32_t    Voxelize(Span<uint64_t>& mortonCodes,
                                         Span<Vector2us>& normals,
                                         bool onlyCalculateSize,
                                         const VoxelizationParameters& voxelParams) const;
        MRAY_HYBRID
        Optional<BasicSurface>  SurfaceFromHit(const Hit& hit) const;
        MRAY_HYBRID
        Optional<Hit>           ProjectedHit(const Vector3& point) const;
        MRAY_HYBRID Vector2     SurfaceParametrization(const Hit& hit) const;

        const TransContextType& GetTransformContext() const;

        // Surface Generation
        MRAY_HYBRID void        GenerateSurface(EmptySurface&,
                                                // Inputs
                                                const Hit&,
                                                const Ray&,
                                                const DiffRay&) const;

        MRAY_HYBRID void        GenerateSurface(BasicSurface&,
                                                // Inputs
                                                const Hit&,
                                                const Ray&,
                                                const DiffRay&) const;

        MRAY_HYBRID void        GenerateSurface(BarycentricSurface&,
                                                // Inputs
                                                const Hit&,
                                                const Ray&,
                                                const DiffRay&) const;

        MRAY_HYBRID void        GenerateSurface(DefaultSurface&,
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

    template<class TransContextType = TransformContextIdentity>
    using SkinnedTriangle   = DefaultTriangleDetail::Triangle<TransContextType>;

    struct TransformContextSkinned
    {
        private:
        // These are generated on the fly (so no reference)
        Matrix4x4 transform;
        Matrix4x4 invTransform;

        public:
        MRAY_HYBRID
        TransformContextSkinned(const typename TransformGroupMulti::DataSoA& transformData,
                                const SkinnedTriangleData& triData,
                                TransformKey tId,
                                PrimitiveKey pId);

        MRAY_HYBRID MRAY_CGPU_INLINE
        Vector3 ApplyP(const Vector3& point) const
        {
            return Vector3(transform *  Vector4(point, 1));
        }

        MRAY_HYBRID MRAY_CGPU_INLINE
        Vector3 ApplyV(const Vector3& vec) const
        {
            return transform * vec;
        }

        MRAY_HYBRID MRAY_CGPU_INLINE
        Vector3 ApplyN(const Vector3& norm) const
        {
            return invTransform.LeftMultiply(norm);
        }

        MRAY_HYBRID MRAY_CGPU_INLINE
        AABB3 Apply(const AABB3& aabb) const
        {
            return transform.TransformAABB(aabb);
        }

        MRAY_HYBRID MRAY_CGPU_INLINE
        Ray Apply(const Ray& ray) const
        {
            return transform.TransformRay(ray);
        }

        MRAY_HYBRID MRAY_CGPU_INLINE
        Vector3 InvApplyP(const Vector3& point) const
        {
            return Vector3(invTransform * Vector4(point, 1));
        }

        MRAY_HYBRID MRAY_CGPU_INLINE
        Vector3 InvApplyV(const Vector3& vec) const
        {
            return invTransform * vec;
        }

        MRAY_HYBRID MRAY_CGPU_INLINE
        Vector3 InvApplyN(const Vector3& norm) const
        {
            return transform.LeftMultiply(norm);
        }

        MRAY_HYBRID MRAY_CGPU_INLINE
        AABB3 InvApply(const AABB3& aabb) const
        {
            return invTransform.TransformAABB(aabb);
        }

        MRAY_HYBRID MRAY_CGPU_INLINE
        Ray InvApply(const Ray& ray) const
        {
            return invTransform.TransformRay(ray);
        }
    };

    // Transform Context Generators
    MRAY_HYBRID
    TransformContextSkinned GenTContextSkinned(const typename TransformGroupMulti::DataSoA&,
                                               const SkinnedTriangleData&,
                                               TransformKey,
                                               PrimitiveKey);

    static_assert(TransformContextC<TransformContextSkinned>);
}

class PrimGroupTriangle final : public GenericGroupPrimitive<PrimGroupTriangle>
{
    static constexpr size_t POSITION_ATTRIB_INDEX   = 0;
    static constexpr size_t NORMAL_ATTRIB_INDEX     = 1;
    static constexpr size_t UV_ATTRIB_INDEX         = 2;
    static constexpr size_t INDICES_ATTRIB_INDEX    = 3;

    public:
    using DataSoA       = DefaultTriangleDetail::TriangleData;
    using Hit           = typename DefaultTriangleDetail::TriHit;

    template <TransformContextC TContext = TransformContextIdentity>
    using Primitive = DefaultTriangleDetail:: template Triangle<TContext>;

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
    static constexpr size_t AttributeCount = 4;
    static constexpr auto TransformLogic = PrimTransformType::LOCALLY_CONSTANT_TRANSFORM;

    private:
    Span<Vector3>       dPositions;
    Span<Quaternion>    dTBNRotations;
    Span<Vector2>       dUVs;
    Span<Vector3ui>     dIndexList;
    DataSoA             soa;

    public:
                            PrimGroupTriangle(uint32_t primGroupId,
                                              const GPUSystem& sys);

    void                    CommitReservations() override;
    PrimAttributeInfoList   AttributeInfo() const override;
    void                    PushAttribute(PrimBatchKey batchKey,
                                          uint32_t attributeIndex,
                                          TransientData data,
                                          const GPUQueue& queue) override;
    void                    PushAttribute(PrimBatchKey batchKey,
                                          uint32_t attributeIndex,
                                          const Vector2ui& subRange,
                                          TransientData data,
                                          const GPUQueue& queue) override;
    void                    PushAttribute(PrimBatchKey idStart, PrimBatchKey idEnd,
                                          uint32_t attributeIndex,
                                          TransientData data,
                                          const GPUQueue& queue) override;

    void                    CopyPrimIds(Span<PrimitiveKey>,
                                        PrimBatchKey,
                                        const GPUQueue&) const override;
    Vector2ui               BatchRange(PrimBatchKey id) const override;

    DataSoA                 SoA() const;
};

class PrimGroupSkinnedTriangle final : public GenericGroupPrimitive<PrimGroupSkinnedTriangle>
{
    static constexpr size_t POSITION_ATTRIB_INDEX   = 0;
    static constexpr size_t NORMAL_ATTRIB_INDEX     = 1;
    static constexpr size_t UV_ATTRIB_INDEX         = 2;
    static constexpr size_t SKIN_W_ATTRIB_INDEX     = 3;
    static constexpr size_t SKIN_I_ATTRIB_INDEX     = 4;
    static constexpr size_t INDICES_ATTRIB_INDEX    = 5;

    public:
    using DataSoA       = DefaultSkinnedTriangleDetail::SkinnedTriangleData;
    using Hit           = typename DefaultTriangleDetail::TriHit;

    template <class TContext = TransformContextIdentity>
    using Primitive     = DefaultTriangleDetail:: template Triangle<TContext>;

    // Transform Context Generators
    static constexpr auto TransContextGeneratorList = std::make_tuple
    (
        TypeFinder::KeyTFuncPair<TransformGroupIdentity,
                                 TransformContextIdentity,
                                 &GenTContextIdentity<DataSoA>>{},
        TypeFinder::KeyTFuncPair<TransformGroupSingle,
                                 TransformContextSingle,
                                 &GenTContextSingle<DataSoA>>{},
        TypeFinder::KeyTFuncPair<TransformGroupMulti,
                                 DefaultSkinnedTriangleDetail::TransformContextSkinned,
                                 &DefaultSkinnedTriangleDetail::GenTContextSkinned>{}
    );
    // Actual Name of the Type
    static std::string_view TypeName();
    static constexpr size_t AttributeCount = 6;
    static constexpr auto TransformLogic = PrimTransformType::PER_PRIMITIVE_TRANSFORM;

    private:
    Span<Vector3>       dPositions;
    Span<Quaternion>    dTBNRotations;
    Span<Vector2>       dUVs;
    Span<Vector3ui>     dIndexList;
    Span<UNorm4x8>      dSkinWeights;
    Span<Vector4uc>     dSkinIndices;
    DataSoA             soa;

    public:
                            PrimGroupSkinnedTriangle(uint32_t primGroupId,
                                                     const GPUSystem& sys);

    void                    CommitReservations() override;
    PrimAttributeInfoList   AttributeInfo() const override;
    void                    PushAttribute(PrimBatchKey batchKey,
                                          uint32_t attributeIndex,
                                          TransientData data,
                                          const GPUQueue& queue) override;
    void                    PushAttribute(PrimBatchKey batchKey,
                                          uint32_t attributeIndex,
                                          const Vector2ui& subRange,
                                          TransientData data,
                                          const GPUQueue& queue) override;
    void                    PushAttribute(PrimBatchKey idStart,
                                          PrimBatchKey idEnd,
                                          uint32_t attributeIndex,
                                          TransientData data,
                                          const GPUQueue& queue) override;

    void                    CopyPrimIds(Span<PrimitiveKey>,
                                        PrimBatchKey,
                                        const GPUQueue&) const override;
    Vector2ui               BatchRange(PrimBatchKey id) const override;

    DataSoA                 SoA() const;
};

inline std::string_view PrimGroupTriangle::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Triangle"sv;
    return PrimTypeName<Name>;
}

inline std::string_view PrimGroupSkinnedTriangle::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "TriangleSkinned"sv;
    return PrimTypeName<Name>;
}

#include "PrimitiveDefaultTriangle.hpp"

static_assert(PrimitiveGroupC<PrimGroupTriangle>);
static_assert(PrimitiveGroupC<PrimGroupSkinnedTriangle>);