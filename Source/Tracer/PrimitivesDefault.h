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
#include "Core/ShapeFunctions.h"

#include "PrimitiveC.h"
#include "TransformsDefault.h"
#include "Random.h"

namespace DefaultSphereDetail
{
    constexpr size_t DeviceMemAllocationGranularity = 2_MiB;
    constexpr size_t DeviceMemReservationSize = 4_MiB;

    // SoA data of triangle group
    struct SphereData
    {
        // Per vertex attributes
        Span<const Vector3>     centers;
        Span<const Float>       radius;
    };

    using SphereHit             = Vector2;
    using SphereIntersection    = IntersectionT<SphereHit>;

    template<TransformContextC TransContextType = TransformContextIdentity>
    class Sphere
    {
        public:
        using DataSoA           = SphereData;
        using Hit               = SphereHit;
        using Intersection      = Optional<SphereIntersection>;
        using TransformContext  = TransContextType;
        //
        static constexpr uint32_t SampleRNCount = 2;

        private:
        Vector3                     center;
        Float                       radius;
        Ref<const TransContextType> transformContext;

        public:
        MRAY_HYBRID             Sphere(const TransContextType& transform,
                                       const SphereData& data, PrimitiveKey key);
        MRAY_HYBRID
        Intersection            Intersects(const Ray& ray, bool cullBackface) const;
        MRAY_HYBRID
        SampleT<BasicSurface>   SampleSurface(RNGDispenser& rng) const;
        MRAY_HYBRID Float       PdfSurface(const Hit& hit) const;
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

        MRAY_HYBRID
        const TransContextType& GetTransformContext() const;

        // Surface Generation
        MRAY_HYBRID void        GenerateSurface(EmptySurface&,
                                                RayConeSurface&,
                                                // Inputs
                                                const Hit&,
                                                const Ray&,
                                                const RayCone&) const;

        MRAY_HYBRID void        GenerateSurface(BasicSurface&,
                                                RayConeSurface&,
                                                // Inputs
                                                const Hit&,
                                                const Ray&,
                                                const RayCone&) const;

        MRAY_HYBRID void        GenerateSurface(DefaultSurface&,
                                                RayConeSurface&,
                                                // Inputs
                                                const Hit&,
                                                const Ray&,
                                                const RayCone&) const;
    };

}

class PrimGroupSphere final : public GenericGroupPrimitive<PrimGroupSphere>
{
    static constexpr size_t CENTER_ATTRIB_INDEX = 0;
    static constexpr size_t RADIUS_ATTRIB_INDEX = 1;

    public:
    using DataSoA   = DefaultSphereDetail::SphereData;
    using Hit       = typename DefaultSphereDetail::SphereHit;

    template <TransformContextC TContext = TransformContextIdentity>
    using Primitive = DefaultSphereDetail:: template Sphere<TContext>;

    // Transform Context Generators
    using TransContextGeneratorList = TypeFinder::T_VMapper:: template Map
    <
        TypeFinder::T_VMapper::template TVPair<TransformGroupIdentity,
                                               &GenTContextIdentity<DataSoA>>,
        TypeFinder::T_VMapper::template TVPair<TransformGroupSingle,
                                               &GenTContextSingle<DataSoA>>
    >;

    // The actual name of the type
    static std::string_view TypeName();
    static constexpr size_t AttributeCount = 2;
    static constexpr auto TransformLogic = PrimTransformType::LOCALLY_CONSTANT_TRANSFORM;

    private:
    Span<Vector3>   dCenters;
    Span<Float>     dRadius;
    DataSoA         soa;

    public:
                            PrimGroupSphere(uint32_t primGroupId,
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
    void                    Finalize(const GPUQueue& queue) override;

    Vector2ui               BatchRange(PrimBatchKey id) const override;
    size_t                  TotalPrimCount() const override;
    DataSoA                 SoA() const;
};

inline std::string_view PrimGroupSphere::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Sphere"sv;
    return PrimTypeName<Name>;
}

#include "PrimitivesDefault.hpp"

static_assert(PrimitiveGroupC<PrimGroupSphere>);
static_assert(!TrianglePrimGroupC<PrimGroupSphere>);
