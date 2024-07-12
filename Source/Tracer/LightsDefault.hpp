#pragma once

namespace LightDetail
{

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
LightPrim<P, SC>::LightPrim(const typename SpectrumConverter& specTransformer,
                            const P& p, const LightData& soa, LightKey key)
    : prim(p)
    , radiance(specTransformer, soa.dRadiances[key.FetchIndexPortion()])
    , initialMedium(soa.dMediumIds[key.FetchIndexPortion()])
    , isTwoSided(soa.dIsTwoSidedFlags[key.FetchIndexPortion()])
{}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> LightPrim<P, SC>::SampleSolidAngle(RNGDispenser& rng,
                                                    const Vector3& distantPoint) const
{
    SampleT<BasicSurface> surfaceSample = prim.SampleSurface(rng);
    Vector3 sampledDir = (surfaceSample.value.position - distantPoint);

    Float NdL = surfaceSample.value.normal.Dot(-sampledDir.Normalize());
    NdL = (isTwoSided) ? abs(NdL) : max(Float{0}, NdL);
    // Get projected area
    Float pdf = (NdL == 0) ? Float{0.0} : surfaceSample.pdf / NdL;
    pdf *= sampledDir.LengthSqr();

    return SampleT<Vector3>
    {
        .value = surfaceSample.value.position,
        .pdf = pdf
    };
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Float LightPrim<P, SC>::PdfSolidAngle(const typename P::Hit& hit,
                                      const Vector3& distantPoint,
                                      const Vector3& dir) const
{
    // Project point to surface (function assumes
    Optional<BasicSurface> surfaceOpt = prim.SurfaceFromHit(hit);
    if(!surfaceOpt) return Float{0};

    BasicSurface surface = surfaceOpt.value();

    Float pdf = prim.PdfSurface(hit);
    Float NdL = surface.normal.Dot(-dir);
    NdL = (isTwoSided) ? abs(NdL) : max(Float{0}, NdL);
    // Get projected area
    pdf = (NdL == 0) ? Float{0.0} : pdf / NdL;
    pdf *= (distantPoint - surface.position).LengthSqr();
    return pdf;
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t LightPrim<P, SC>::SampleSolidAngleRNCount() const
{
    return prim.SampleRNCount();
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Ray> LightPrim<P, SC>::SampleRay(RNGDispenser& rng) const
{
    // What is the probability?
    //return SampleT<Ray>{};
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Float LightPrim<P, SC>::PdfRay(const Ray&) const
{
    //return 0;
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t LightPrim<P, SC>::SampleRayRNCount() const
{
    return 4;
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum LightPrim<P, SC>::EmitViaHit(const Vector3& wO,
                                      const typename P::Hit& hitParams) const
{
    // Find
    Vector2 uv = radiance.Constant()
                    ? Vector2::Zero()
                    : prim.ProjectedHit(hitParams);
    return radiance(uv);
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum LightPrim<P, SC>::EmitViaSurfacePoint(const Vector3& wO,
                                               const Vector3& surfacePoint) const
{
    using Hit = typename P::Hit;
    Optional<Hit> hit = prim.ProjectedHit(surfacePoint);
    if(!hit) return Spectrum::Zero();

    Vector2 uv = radiance.Constant()
                    ? Vector2::Zero()
                    : hit.value();
    return radiance(uv);
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
bool LightPrim<P, SC>::IsPrimitiveBackedLight() const
{
    return true;
}

}

namespace LightDetail
{

MRAY_HYBRID MRAY_CGPU_INLINE
Vector2 SphericalCoordConverter::DirToUV(const Vector3& dirYUp)
{
    Vector3 dirZUp = TransformGen::YUpToZUp(dirYUp);
    Vector2 thetaPhi = Graphics::CartesianToUnitSpherical(dirZUp);
    Vector2 uv = Graphics::SphericalAnglesToUV(thetaPhi);
    return uv;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 SphericalCoordConverter::UVToDir(const Vector2& uv)
{
    Vector2 thetaPhi = Graphics::UVToSphericalAngles(uv);
    Vector3 dirZUp = Graphics::UnitSphericalToCartesian(thetaPhi);
    Vector3 dirYUp = TransformGen::ZUpToYUp(dirZUp);
    return dirYUp;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float SphericalCoordConverter::ToSolidAnglePdf(Float pdf, const Vector3& dirYUp)
{
    using namespace MathConstants;
    // There is code duplication here hopefully this will optimized out
    Vector3 dirZUp = TransformGen::YUpToZUp(dirYUp);
    Vector2 thetaPhi = Graphics::CartesianToUnitSpherical(dirZUp);

    // Convert to solid angle pdf
    // http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources.html
    Float sinPhi = sin(thetaPhi[1]);
    pdf = (sinPhi == 0) ? 0 : pdf / (2 * PiSqr<Float>() * sinPhi);
    return pdf;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float SphericalCoordConverter::ToSolidAnglePdf(Float pdf, const Vector2& uv)
{
    using namespace MathConstants;
    // Similar to the direction version, code duplication here
    Vector2 thetaPhi = Graphics::UVToSphericalAngles(uv);
    // Convert to solid angle pdf
    // http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources.html
    Float sinPhi = sin(thetaPhi[1]);
    pdf = (sinPhi == 0) ? 0 : pdf / (2 * PiSqr<Float>() * sinPhi);
    return pdf;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector2 CoOctoCoordConverter::DirToUV(const Vector3& dirYUp)
{
    Vector3 dirZUp = TransformGen::YUpToZUp(dirYUp);
    Vector2 uv = Graphics::DirectionToCocentricOctohedral(dirZUp);
    return uv;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 CoOctoCoordConverter::UVToDir(const Vector2& uv)
{
    Vector3 dirZUp = Graphics::CocentricOctohedralToDirection(uv);
    Vector3 dirYUp = TransformGen::ZUpToYUp(dirZUp);
    return dirYUp;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float CoOctoCoordConverter::ToSolidAnglePdf(Float pdf, const Vector3&)
{
    using namespace MathConstants;
    return pdf * Float(0.25) * InvPi<Float>();
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float CoOctoCoordConverter::ToSolidAnglePdf(Float pdf, const Vector2&)
{
    using namespace MathConstants;
    return pdf * Float(0.25) * InvPi<Float>();
}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
LightSkysphere<CC, TC, SC>::LightSkysphere(const SpectrumConverter& specTransformer,
                                           const Primitive& p, const LightSkysphereData& soa, LightKey key)
    : prim(p)
    , radiance(specTransformer, soa.dRadiances[key.FetchIndexPortion()])
    , initialMedium(soa.dMediumIds[key.FetchIndexPortion()])
    , dist2D(soa.dDistributions[key.FetchIndexPortion()])
    , sceneDiameter(soa.sceneDiameter)
{}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> LightSkysphere<CC, TC, SC>::SampleSolidAngle(RNGDispenser& rng,
                                                              const Vector3& distantPoint) const
{
    Vector2 xi = rng.NextFloat2D<0>();
    SampleT<Vector2> sampledUV = dist2D.SampleUV(xi);
    Vector3 dirYUp = CC::UVToDir(sampledUV.value);
    Float pdf = CC::ToSolidAnglePDF(sampledUV.pdf, sampledUV.value);
    // Transform Direction to World Space
    Vector3 worldDir = prim.get().GetTransformContext().InvApplyV(dirYUp);

    // Now add the extent of the scene
    Vector3 sampledPoint = distantPoint + worldDir * sceneDiameter;
    return SampleT<Vector3>
    {
        .value = sampledPoint,
        .pdf = pdf
    };
}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Float LightSkysphere<CC, TC, SC>::PdfSolidAngle(const typename EmptyPrimitive<TC>::Hit& hit,
                                                const Vector3& distantPoint,
                                                const Vector3& dir) const
{
    Vector3 dirYUp = prim.get().GetTransformContext().ApplyV(dir);
    Vector2 uv = CC::DirToUV(dirYUp);
    Float pdf = dist2D.PdfUV(uv);
    pdf = CC::ToSolidAnglePdf(pdf, dirYUp);
    return pdf;
}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t LightSkysphere<CC, TC, SC>::SampleSolidAngleRNCount() const
{
    return 2;
}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Ray> LightSkysphere<CC, TC, SC>::SampleRay(RNGDispenser& rng) const
{
    // What is the probability?
    Vector2 xi = rng.NextFloat2D<0>();
    SampleT<Vector2> sampledUV = dist2D.SampleUV(xi);
    Vector3 dirYUp = CC::UVToDir(sampledUV.value);
    Float pdf = CC::ToSolidAnglePDF(sampledUV.pdf, sampledUV.value);

    // Transform Direction to World Space
    Vector3 worldDir = prim.get().GetTransformContext().InvApplyV(dirYUp);

    // Now add the extent of the scene
    Vector3 sampledPoint = worldDir * sceneDiameter;
    return SampleT<Ray>
    {
        .value = Ray(-worldDir, sampledPoint),
        .pdf = pdf
    };
}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Float LightSkysphere<CC, TC, SC>::PdfRay(const Ray& ray) const
{
    Vector3 dirYUp = prim.get().GetTransformContext().ApplyV(-ray.Dir());
    Vector2 uv = CC::DirToUV(dirYUp);
    Float pdf = dist2D.PdfUV(uv);
    pdf = CC::ToSolidAnglePDF(pdf, uv);
    return pdf;
}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t LightSkysphere<CC, TC, SC>::SampleRayRNCount() const
{
    return 2;
}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum LightSkysphere<CC, TC, SC>::EmitViaHit(const Vector3& wO,
                                                const typename EmptyPrimitive<TC>::Hit&) const
{
    Vector3 dirYUp = prim.get().GetTransformContext().ApplyV(-wO);
    // TODO: How to incorporate differentials here?
    Vector2 uv = radiance.Constant()
                    ? CC::DirToUV(dirYUp)
                    : Vector2::Zero();
    return radiance(uv);
}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum LightSkysphere<CC, TC, SC>::EmitViaSurfacePoint(const Vector3& wO,
                                                         const Vector3&) const
{
    // Distant light do not have surfaces, this function
    // should not be called.
    assert(false);
    return Spectrum::Zero();
}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
bool LightSkysphere<CC, TC, SC>::IsPrimitiveBackedLight() const
{
    return false;
}

static_assert(LightC<LightSkysphere<SphericalCoordConverter>>);
static_assert(LightC<LightSkysphere<CoOctoCoordConverter>>);

}

namespace LightDetail
{

    template<TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
LightNull<TC, SC>::LightNull(const SpectrumConverter&,
                            const Primitive&,
                            const DataSoA&, LightKey)
{}

template<TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> LightNull<TC, SC>::SampleSolidAngle(RNGDispenser&,
                                             const Vector3&) const
{
    return SampleT<Vector3>
    {
        Vector3::Zero(),
        Float(0.0)
    };

}

template<TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Float LightNull<TC, SC>::PdfSolidAngle(const typename Primitive::Hit&,
                                       const Vector3&,
                                       const Vector3&) const
{
    return Float(0.0);
}

template<TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t LightNull<TC, SC>::SampleSolidAngleRNCount() const
{
    return 0;
}

template<TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Ray> LightNull<TC, SC>::SampleRay(RNGDispenser&) const
{
    return SampleT<Ray>
    {
        Ray(Vector3::Zero(), Vector3::Zero()),
        Float(0.0)
    };
}

template<TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Float LightNull<TC, SC>::PdfRay(const Ray&) const
{
    return Float(0.0);
}

template<TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t LightNull<TC, SC>::SampleRayRNCount() const
{
    return 0;
}

template<TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum LightNull<TC, SC>::EmitViaHit(const Vector3&,
                                       const typename Primitive::Hit&) const
{
    return Spectrum::Zero();
}

template<TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum LightNull<TC, SC>::EmitViaSurfacePoint(const Vector3&,
                                                const Vector3&) const
{
    return Spectrum::Zero();
}

template<TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
bool LightNull<TC, SC>::IsPrimitiveBackedLight() const
{
    return false;
}

}

template <PrimitiveGroupC PG>
std::string_view LightGroupPrim<PG>::TypeName()
{
    using namespace TypeNameGen::CompTime;
    static const auto Name = PrimLightTypeName(PG::TypeName());
    return Name;
}

template <PrimitiveGroupC PG>
LightGroupPrim<PG>::LightGroupPrim(uint32_t groupId,
                                   const GPUSystem& system,
                                   const TextureViewMap& map,
                                   const GenericGroupPrimitiveT& primGroup)
    : Parent(groupId, system, map)
    , primGroup(static_cast<const PrimGroup&>(primGroup))
{}

template <PrimitiveGroupC PG>
void LightGroupPrim<PG>::CommitReservations()
{
}

template <PrimitiveGroupC PG>
LightAttributeInfoList LightGroupPrim<PG>::AttributeInfo() const
{
    return LightAttributeInfoList{};
}

template <PrimitiveGroupC PG>
void LightGroupPrim<PG>::PushAttribute(LightKey id,
                                       uint32_t attributeIndex,
                                       TransientData data,
                                       const GPUQueue& queue)
{

}

template <PrimitiveGroupC PG>
void LightGroupPrim<PG>::PushAttribute(LightKey id,
                                       uint32_t attributeIndex,
                                       const Vector2ui& subRange,
                                       TransientData data,
                                       const GPUQueue& queue)
{

}

template <PrimitiveGroupC PG>
void LightGroupPrim<PG>::PushAttribute(LightKey idStart, LightKey idEnd,
                                       uint32_t attributeIndex,
                                       TransientData data,
                                       const GPUQueue& queue)
{

}

template <PrimitiveGroupC PG>
void LightGroupPrim<PG>::PushTexAttribute(LightKey idStart, LightKey idEnd,
                                          uint32_t attributeIndex,
                                          TransientData,
                                          std::vector<Optional<TextureId>>,
                                          const GPUQueue& queue)
{

}

template <PrimitiveGroupC PG>
void LightGroupPrim<PG>::PushTexAttribute(LightKey idStart, LightKey idEnd,
                                          uint32_t attributeIndex,
                                          std::vector<Optional<TextureId>>,
                                          const GPUQueue& queue)
{

}

template <PrimitiveGroupC PG>
void LightGroupPrim<PG>::PushTexAttribute(LightKey idStart, LightKey idEnd,
                                          uint32_t attributeIndex,
                                          std::vector<TextureId>,
                                          const GPUQueue& queue)
{

}

template <PrimitiveGroupC PG>
typename LightGroupPrim<PG>::DataSoA LightGroupPrim<PG>::SoA() const
{
    return soa;
}

template <PrimitiveGroupC PG>
const PG& LightGroupPrim<PG>::PrimitiveGroup() const
{
    return primGroup;
}

template <PrimitiveGroupC PG>
const GenericGroupPrimitiveT& LightGroupPrim<PG>::GenericPrimGroup() const
{
    return primGroup;
}

template <PrimitiveGroupC PG>
bool LightGroupPrim<PG>::IsPrimitiveBacked() const
{
    return true;
}

template <CoordConverterC CC>
std::string_view LightGroupSkysphere<CC>::TypeName()
{
    // using namespace TypeNameGen::CompTime;
    // using namespace std::string_view_literals;
    // static constexpr auto Name = "Skysphere"sv;
    // return LightTypeName<Name>;
    if constexpr(std::is_same_v<CC, LightDetail::CoOctoCoordConverter>)
        return "(L)Skysphere_CoOcto";

    if constexpr(std::is_same_v<CC, LightDetail::SphericalCoordConverter>)
        return "(L)Skysphere_Spherical";

    return "(L)Skysphere";
}

template <CoordConverterC CC>
LightGroupSkysphere<CC>::LightGroupSkysphere(uint32_t groupId,
                                             const GPUSystem& system,
                                             const TextureViewMap& map,
                                             const GenericGroupPrimitiveT& primGroup)
    : Parent(groupId, system, map)
    , primGroup(static_cast<const PrimGroup&>(primGroup))
{}

template <CoordConverterC CC>
void LightGroupSkysphere<CC>::CommitReservations()
{

}

template <CoordConverterC CC>
LightAttributeInfoList LightGroupSkysphere<CC>::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeOptionality;
    using enum AttributeTexturable;
    using enum AttributeIsArray;
    using enum AttributeIsColor;
    static const MatAttributeInfoList LogicList =
    {
        MatAttributeInfo("radiance", MRayDataType<MR_VECTOR_3>(), IS_SCALAR,
                         MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_COLOR)
    };
    return LogicList;
}

template <CoordConverterC CC>
void LightGroupSkysphere<CC>::PushAttribute(LightKey id,
                                            uint32_t attributeIndex,
                                            TransientData data,
                                            const GPUQueue& queue)
{

}

template <CoordConverterC CC>
void LightGroupSkysphere<CC>::PushAttribute(LightKey id,
                                            uint32_t attributeIndex,
                                            const Vector2ui& subRange,
                                            TransientData data,
                                            const GPUQueue& queue)
{

}

template <CoordConverterC CC>
void LightGroupSkysphere<CC>::PushAttribute(LightKey idStart, LightKey idEnd,
                                            uint32_t attributeIndex,
                                            TransientData data,
                                            const GPUQueue& queue)
{

}

template <CoordConverterC CC>
void LightGroupSkysphere<CC>::PushTexAttribute(LightKey idStart, LightKey idEnd,
                                               uint32_t attributeIndex,
                                               TransientData,
                                               std::vector<Optional<TextureId>>,
                                               const GPUQueue& queue)
{

}

template <CoordConverterC CC>
void LightGroupSkysphere<CC>::PushTexAttribute(LightKey idStart, LightKey idEnd,
                                               uint32_t attributeIndex,
                                               std::vector<Optional<TextureId>>,
                                               const GPUQueue& queue)
{

}

template <CoordConverterC CC>
void LightGroupSkysphere<CC>::PushTexAttribute(LightKey idStart, LightKey idEnd,
                                               uint32_t attributeIndex,
                                               std::vector<TextureId>,
                                               const GPUQueue& queue)
{

}

template <CoordConverterC CC>
typename LightGroupSkysphere<CC>::DataSoA LightGroupSkysphere<CC>::SoA() const
{
    return soa;
}

template <CoordConverterC CC>
const typename LightGroupSkysphere<CC>::PrimGroup& LightGroupSkysphere<CC>::PrimitiveGroup() const
{
    return primGroup;
}

template <CoordConverterC CC>
const GenericGroupPrimitiveT& LightGroupSkysphere<CC>::GenericPrimGroup() const
{
    return primGroup;
}

template <CoordConverterC CC>
bool LightGroupSkysphere<CC>::IsPrimitiveBacked() const
{
    return false;
}