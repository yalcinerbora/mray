#pragma once

#include <string_view>

#include "Core/TypeFinder.h"
#include "Core/TypeNameGenerators.h"

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
        MR_HF_DECL              Sphere(const TransContextType& transform,
                                       const SphereData& data, PrimitiveKey key);
        MR_HF_DECL
        Intersection            Intersects(const Ray& ray, bool cullBackface) const;
        MR_HF_DECL
        SampleT<BasicSurface>   SampleSurface(RNGDispenser& rng) const;
        MR_HF_DECL Float        PdfSurface(const Hit& hit) const;
        MR_HF_DECL Float        GetSurfaceArea() const;
        MR_HF_DECL AABB3        GetAABB() const;
        MR_HF_DECL Vector3      GetCenter() const;
        MR_HF_DECL uint32_t     Voxelize(Span<uint64_t>& mortonCodes,
                                         Span<Vector2us>& normals,
                                         bool onlyCalculateSize,
                                         const VoxelizationParameters& voxelParams) const;
        MR_HF_DECL
        Optional<BasicSurface>  SurfaceFromHit(const Hit& hit) const;
        MR_HF_DECL
        Optional<Hit>           ProjectedHit(const Vector3& point) const;
        MR_HF_DECL Vector2      SurfaceParametrization(const Hit& hit) const;

        MR_HF_DECL
        const TransContextType& GetTransformContext() const;

        // Surface Generation
        MR_HF_DECL void GenerateSurface(EmptySurface&,
                                        RayConeSurface&,
                                        // Inputs
                                        const Hit&,
                                        const Ray&,
                                        const RayCone&) const;

        MR_HF_DECL void GenerateSurface(BasicSurface&,
                                        RayConeSurface&,
                                        // Inputs
                                        const Hit&,
                                        const Ray&,
                                        const RayCone&) const;

        MR_HF_DECL void GenerateSurface(DefaultSurface&,
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
