#pragma once

namespace LightDetail
{

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
LightPrim<P, SC>::LightPrim(const typename SpectrumConverter& specTransformer,
                            const P& p, const LightData& soa, LightKey key)
    : prim(p)
    , radiance(specTransformer, soa.dRadiances[key.FetchIndexPortion()])
    , isTwoSided(soa.dIsTwoSidedFlags[key.FetchIndexPortion()])
{}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> LightPrim<P, SC>::SampleSolidAngle(RNGDispenser& rng,
                                                    const Vector3& distantPoint) const
{
    const P& primitive = prim.get();
    SampleT<BasicSurface> surfaceSample = primitive.SampleSurface(rng);
    Vector3 sampledDir = (distantPoint - surfaceSample.value.position);
    Float distSqr = sampledDir.LengthSqr();
    sampledDir.NormalizeSelf();

    Float NdL = surfaceSample.value.normal.Dot(sampledDir);
    NdL = (isTwoSided) ? abs(NdL) : max(Float{0}, NdL);
    // Get projected area
    Float pdf = (NdL == 0) ? Float{0.0} : surfaceSample.pdf / NdL;
    pdf *= distSqr;

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
    const P& primitive = prim.get();
    // Project point to surface
    Optional<BasicSurface> surfaceOpt = primitive.SurfaceFromHit(hit);
    if(!surfaceOpt) return Float{0};
    const BasicSurface& surface = surfaceOpt.value();

    Float pdf = primitive.PdfSurface(hit);
    Float NdL = surface.normal.Dot(-dir);
    NdL = (isTwoSided) ? abs(NdL) : max(Float{0}, NdL);
    // Get projected area
    pdf = (NdL == 0) ? Float{0} : pdf / NdL;
    pdf *= (distantPoint - surface.position).LengthSqr();
    return pdf;
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Ray> LightPrim<P, SC>::SampleRay(RNGDispenser& rng) const
{
    using Distribution::Common::SampleUniformDirection;
    const P& primitive = prim.get();
    SampleT<BasicSurface> surfaceSample = primitive.SampleSurface(rng);

    Vector2 xi = rng.NextFloat2D<Primitive::SampleRNCount>();
    SampleT<Vector3> dirSample;
    if(isTwoSided)
    {
        using Distribution::Common::BisectSample2;
        // TODO: We bisect one dimension of 2D sample.
        // Is this wrong?
        auto bisection = BisectSample2(xi[0], Vector2(0.5), true);
        dirSample = SampleUniformDirection(Vector2(bisection.second, xi[1]));

        if(bisection.first == 0)
            dirSample.value = -dirSample.value;
    }
    else dirSample = SampleUniformDirection(xi);

    return SampleT<Ray>
    {
        .value = Ray(dirSample.value, surfaceSample.value.position),
        .pdf = surfaceSample.pdf * dirSample.pdf
    };
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Float LightPrim<P, SC>::PdfRay(const Ray& ray) const
{
    using Hit = typename P::Hit;
    using Distribution::Common::PDFUniformDirection;

    const P& primitive = prim.get();
    Optional<Hit> hit = primitive.ProjectedHit(ray.Pos());
    if(!hit.has_value()) return Float(0);

    Optional<BasicSurface> surf = primitive.SurfaceFromHit(*hit);
    if(!surf.has_value()) return Float(0);

    Float NdL = (*surf).normal.Dot(ray.Dir());
    if(!isTwoSided && NdL <= Float(0))
        return Float(0);

    Float pdfDir = PDFUniformDirection();
    if(isTwoSided) pdfDir *= Float(2);

    Float pdfSurface = primitive.PdfSurface(*hit);
    return pdfDir * pdfSurface;
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum LightPrim<P, SC>::EmitViaHit(const Vector3& wO,
                                      const typename P::Hit& hit) const
{

    const P& primitive = prim.get();
    Vector2 uv = (radiance.IsConstant())
        ? Vector2::Zero()
        : primitive.SurfaceParametrization(hit);

    Optional<BasicSurface> surf = primitive.SurfaceFromHit(hit);
    if(!surf.has_value()) return Spectrum::Zero();

    Float NdL = (*surf).normal.Dot(wO);
    if(!isTwoSided && NdL <= Float(0))
        return Spectrum::Zero();

    // TODO: How to incorporate differentials here?
    return radiance(uv).value();
}

template<PrimitiveC P, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum LightPrim<P, SC>::EmitViaSurfacePoint(const Vector3& wO,
                                               const Vector3& surfacePoint) const
{
    const P& primitive = prim.get();
    using Hit = typename P::Hit;
    Optional<Hit> hit = primitive.ProjectedHit(surfacePoint);
    if(!hit) return Spectrum::Zero();
    Vector2 uv = radiance.IsConstant()
                ? Vector2::Zero()
                : primitive.SurfaceParametrization(*hit);

    Optional<BasicSurface> surf = primitive.SurfaceFromHit(*hit);
    if(!surf.has_value()) return Spectrum::Zero();

    Float NdL = (*surf).normal.Dot(wO);
    if(!isTwoSided && NdL <= Float(0))
        return Spectrum::Zero();

    // TODO: How to incorporate differentials here?
    return radiance(uv).value();
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
Vector2 CoOctaCoordConverter::DirToUV(const Vector3& dirYUp)
{
    Vector3 dirZUp = TransformGen::YUpToZUp(dirYUp);
    Vector2 uv = Graphics::DirectionToCocentricOctahedral(dirZUp);
    return uv;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 CoOctaCoordConverter::UVToDir(const Vector2& uv)
{
    Vector3 dirZUp = Graphics::CocentricOctahedralToDirection(uv);
    Vector3 dirYUp = TransformGen::ZUpToYUp(dirZUp);
    return dirYUp;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float CoOctaCoordConverter::ToSolidAnglePdf(Float pdf, const Vector3&)
{
    using namespace MathConstants;
    return pdf * Float(0.25) * InvPi<Float>();
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float CoOctaCoordConverter::ToSolidAnglePdf(Float pdf, const Vector2&)
{
    using namespace MathConstants;
    return pdf * Float(0.25) * InvPi<Float>();
}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector2> LightSkysphere<CC, TC, SC>::SampleUV(Vector2 xi) const
{
    if(radiance.IsConstant())
        return SampleT<Vector2>{.value = xi, .pdf = Float(1)};
    else
        return dist2D.SampleUV(xi);
}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Float LightSkysphere<CC, TC, SC>::PdfUV(Vector2 uv) const
{
    if(radiance.IsConstant())
        return Float(1);
    else
        return dist2D.PdfUV(uv);
}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
LightSkysphere<CC, TC, SC>::LightSkysphere(const SpectrumConverter& specTransformer,
                                           const Primitive& p, const LightSkysphereData& soa, LightKey key)
    : prim(p)
    , radiance(specTransformer, soa.dRadiances[key.FetchIndexPortion()])
    , dist2D(soa.dDistributions[key.FetchIndexPortion()])
    , sceneDiameter(soa.sceneDiameter)
{}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> LightSkysphere<CC, TC, SC>::SampleSolidAngle(RNGDispenser& rng,
                                                              const Vector3& distantPoint) const
{
    Vector2 xi = rng.NextFloat2D<0>();
    SampleT<Vector2> sampledUV = SampleUV(xi);
    Vector3 dirYUp = CC::UVToDir(sampledUV.value);
    Float pdf = CC::ToSolidAnglePdf(sampledUV.pdf, sampledUV.value);
    // Transform Direction to World Space
    Vector3 worldDir = prim.get().GetTransformContext().ApplyV(dirYUp);
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
Float LightSkysphere<CC, TC, SC>::PdfSolidAngle(const typename EmptyPrimitive<TC>::Hit&,
                                                const Vector3& distantPoint,
                                                const Vector3& dir) const
{
    Vector3 dirYUp = prim.get().GetTransformContext().InvApplyV(dir);
    Vector2 uv = CC::DirToUV(dirYUp.Normalize());
    Float pdf = PdfUV(uv);
    pdf = CC::ToSolidAnglePdf(pdf, dirYUp);
    return pdf;
}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Ray> LightSkysphere<CC, TC, SC>::SampleRay(RNGDispenser& rng) const
{
    // What is the probability?
    Vector2 xi = rng.NextFloat2D<0>();
    SampleT<Vector2> sampledUV = SampleUV(xi);
    //
    Vector3 dirYUp = CC::UVToDir(sampledUV.value);
    Float pdf = CC::ToSolidAnglePdf(sampledUV.pdf, sampledUV.value);

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
    Float pdf = PdfUV(uv);
    pdf = CC::ToSolidAnglePdf(pdf, uv);
    return pdf;
}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum LightSkysphere<CC, TC, SC>::EmitViaHit(const Vector3& wO,
                                                const typename EmptyPrimitive<TC>::Hit&) const
{
    Vector3 dirYUp = prim.get().GetTransformContext().InvApplyV(-wO).Normalize();
    Vector2 uv = radiance.IsConstant()
                    ? Vector2::Zero()
                    : CC::DirToUV(dirYUp);
    // TODO: How to incorporate differentials here?
    return radiance(uv).value();
}

template<CoordConverterC CC, TransformContextC TC, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum LightSkysphere<CC, TC, SC>::EmitViaSurfacePoint(const Vector3& wO,
                                                         const Vector3&) const
{
    Vector3 dirYUp = prim.get().GetTransformContext().ApplyV(-wO).Normalize();
    Vector2 uv = radiance.IsConstant()
                    ? Vector2::Zero()
                    : CC::DirToUV(dirYUp);
    // TODO: How to incorporate differentials here?
    return radiance(uv).value();
}

static_assert(LightC<LightSkysphere<SphericalCoordConverter>>);
static_assert(LightC<LightSkysphere<CoOctaCoordConverter>>);

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
        Vector3(1e10),
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
SampleT<Ray> LightNull<TC, SC>::SampleRay(RNGDispenser&) const
{
    return SampleT<Ray>
    {
        Ray(Vector3(1e10), Vector3::Zero()),
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
                                   const TextureMap&,
                                   const GenericGroupPrimitiveT& primGroup)
    : Parent(groupId, system, map)
    , primGroup(static_cast<const PrimGroup&>(primGroup))
{}

template <PrimitiveGroupC PG>
void LightGroupPrim<PG>::CommitReservations()
{
    // TODO: Wasting 32x memory cost due to "Bit" is not a type
    // Change this later
    this->GenericCommit(std::tie(dRadiances, dPrimRanges,
                                 dIsTwoSidedFlags),
                        {0, 0, 0});

    auto dIsTwoSidedFlagsIn = Bitspan<uint32_t>(dIsTwoSidedFlags);

    soa.dRadiances = ToConstSpan(dRadiances);
    soa.dIsTwoSidedFlags = ToConstSpan(dIsTwoSidedFlagsIn);
    // TODO:
    //soa.dPrimRanges = ToConstSpan(dPrimRanges);
}

template <PrimitiveGroupC PG>
LightAttributeInfoList LightGroupPrim<PG>::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeOptionality;
    using enum AttributeTexturable;
    using enum AttributeIsArray;
    using enum AttributeIsColor;
    static const LightAttributeInfoList LogicList =
    {
        LightAttributeInfo("radiance", MRayDataType<MR_VECTOR_3>(), IS_SCALAR,
                           MR_MANDATORY, MR_TEXTURE_OR_CONSTANT, IS_COLOR)
    };
    return LogicList;
}

template <PrimitiveGroupC PG>
void LightGroupPrim<PG>::PushAttribute(LightKey,
                                       uint32_t attributeIndex,
                                       TransientData,
                                       const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"ConstantOnly\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

template <PrimitiveGroupC PG>
void LightGroupPrim<PG>::PushAttribute(LightKey,
                                       uint32_t attributeIndex,
                                       const Vector2ui&,
                                       TransientData,
                                       const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"ConstantOnly\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

template <PrimitiveGroupC PG>
void LightGroupPrim<PG>::PushAttribute(LightKey, LightKey,
                                       uint32_t attributeIndex,
                                       TransientData,
                                       const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"ConstantOnly\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

template <PrimitiveGroupC PG>
void LightGroupPrim<PG>::PushTexAttribute(LightKey idStart, LightKey idEnd,
                                          uint32_t attributeIndex,
                                          TransientData data,
                                          std::vector<Optional<TextureId>> texIds,
                                          const GPUQueue& queue)
{
    if(attributeIndex == 0)
    {
        this->template GenericPushTexAttribute<2, Vector3>
        (
            dRadiances,
            //
            idStart, idEnd,
            attributeIndex,
            std::move(data),
            std::move(texIds),
            queue
        );
    }
    else throw MRayError("{:s}: Attribute {:d} is not \"ParamVarying\", wrong "
                         "function is called", TypeName(), attributeIndex);
}

template <PrimitiveGroupC PG>
void LightGroupPrim<PG>::PushTexAttribute(LightKey, LightKey,
                                          uint32_t attributeIndex,
                                          std::vector<Optional<TextureId>>,
                                          const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"Optional Texture\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

template <PrimitiveGroupC PG>
void LightGroupPrim<PG>::PushTexAttribute(LightKey, LightKey,
                                          uint32_t attributeIndex,
                                          std::vector<TextureId>,
                                          const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"Mandatory Texture\", wrong "
                    "function is called", TypeName(), attributeIndex);
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

template <PrimitiveGroupC PG>
void LightGroupPrim<PG>::Finalize(const GPUQueue& q)
{
    std::vector<Vector2ui> hPrimRanges;
    hPrimRanges.reserve(dPrimRanges.size());

    for(const auto& [_, batchKey] : this->primMappings)
        hPrimRanges.push_back(primGroup.BatchRange(batchKey));

    // TODO: Add is two sided flags
    q.MemsetAsync(dIsTwoSidedFlags, 0x00);
    q.MemcpyAsync(dPrimRanges, Span<const Vector2ui>(hPrimRanges));
    q.Barrier().Wait();
}

template <CoordConverterC CC>
std::string_view LightGroupSkysphere<CC>::TypeName()
{
    // using namespace TypeNameGen::CompTime;
    // using namespace std::string_view_literals;
    // static constexpr auto Name = "Skysphere"sv;
    // return LightTypeName<Name>;
    if constexpr(std::is_same_v<CC, LightDetail::CoOctaCoordConverter>)
        return "(L)Skysphere_CoOcta";
    else if constexpr(std::is_same_v<CC, LightDetail::SphericalCoordConverter>)
        return "(L)Skysphere_Spherical";
    else
        return "(L)Skysphere";
}

template <CoordConverterC CC>
LightGroupSkysphere<CC>::LightGroupSkysphere(uint32_t groupId,
                                             const GPUSystem& system,
                                             const TextureViewMap& map,
                                             const TextureMap& texMapIn,
                                             const GenericGroupPrimitiveT& primGroup)
    : Parent(groupId, system, map)
    , primGroup(static_cast<const PrimGroup&>(primGroup))
    , pwcDistributions(system)
    , texMap(texMapIn)
{}

template <CoordConverterC CC>
void LightGroupSkysphere<CC>::CommitReservations()
{
    this->GenericCommit(std::tie(dRadiances), {0});
    radianceFieldTextures = std::vector<const GenericTexture*>(this->TotalItemCount(), nullptr);
    soa.dRadiances = ToConstSpan(dRadiances);
    soa.sceneDiameter = sceneDiameter;
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
void LightGroupSkysphere<CC>::PushAttribute(LightKey,
                                            uint32_t attributeIndex,
                                            TransientData,
                                            const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"ConstantOnly\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

template <CoordConverterC CC>
void LightGroupSkysphere<CC>::PushAttribute(LightKey,
                                            uint32_t attributeIndex,
                                            const Vector2ui&,
                                            TransientData,
                                            const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"ConstantOnly\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

template <CoordConverterC CC>
void LightGroupSkysphere<CC>::PushAttribute(LightKey, LightKey,
                                            uint32_t attributeIndex,
                                            TransientData,
                                            const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"ConstantOnly\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

template <CoordConverterC CC>
void LightGroupSkysphere<CC>::PushTexAttribute(LightKey idStart, LightKey idEnd,
                                               uint32_t attributeIndex,
                                               TransientData data,
                                               std::vector<Optional<TextureId>> texIds,
                                               const GPUQueue& queue)
{
    if(attributeIndex != 0)
        throw MRayError("{:s}: Attribute {:d} is not \"ParamVarying\", wrong "
                        "function is called", TypeName(), attributeIndex);
    //
    CommonKey i = std::bit_cast<CommonKey>(idStart.FetchIndexPortion());
    for(const auto& texId : texIds)
    {
        if(!texId.has_value()) continue;

        auto optTex = texMap.at(*texId);
        if(!optTex)
        {
            throw MRayError("{:s}: Given texture({:d}) is not found",
                            TypeName(), static_cast<CommonKey>(*texId));
        }
        const GenericTexture& tex = optTex.value();
        radianceFieldTextures[i] = &tex;
        i++;
    }
    //
    this->template GenericPushTexAttribute<2, Vector3>
    (
        dRadiances,
        //
        idStart, idEnd,
        attributeIndex,
        std::move(data),
        std::move(texIds),
        queue
    );
}

template <CoordConverterC CC>
void LightGroupSkysphere<CC>::PushTexAttribute(LightKey, LightKey,
                                               uint32_t attributeIndex,
                                               std::vector<Optional<TextureId>>,
                                               const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"Optional Texture\", wrong "
                    "function is called", TypeName(), attributeIndex);
}

template <CoordConverterC CC>
void LightGroupSkysphere<CC>::PushTexAttribute(LightKey, LightKey,
                                               uint32_t attributeIndex,
                                               std::vector<TextureId>,
                                               const GPUQueue&)
{
    throw MRayError("{:s}: Attribute {:d} is not \"Mandatory Texture\", wrong "
                    "function is called", TypeName(), attributeIndex);
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

template <CoordConverterC CC>
void LightGroupSkysphere<CC>::SetSceneDiameter(Float d)
{
    sceneDiameter = d;
    soa.sceneDiameter = sceneDiameter;
}

template <CoordConverterC CC>
void LightGroupSkysphere<CC>::Finalize(const GPUQueue& q)
{
    //
    std::vector<size_t> radianceFieldSizesLinear;
    radianceFieldSizesLinear.reserve(radianceFieldTextures.size());
    // Find sizes and allocate a temp memory
    for(const GenericTexture* t : radianceFieldTextures)
    {
        Vector2ui size = (t) ? Vector2ui(t->Extents()) : Vector2ui(1);
        pwcDistributions.Reserve(size);
        radianceFieldSizesLinear.push_back(size.Multiply());
    }
    pwcDistributions.Commit();

    // Allocate the buffer
    DeviceLocalMemory tempMem(*q.Device());
    std::vector<Span<Float>> dLuminanceData =
        MemAlloc::AllocateSegmentedData<Float>(tempMem, radianceFieldSizesLinear);

    // For non-texture types just put 1.
    auto allMem = Span<Byte>(static_cast<Byte*>(tempMem), tempMem.Size());
    Span<Float> allMemFloat =  MemAlloc::RepurposeAlloc<Float>(allMem);
    DeviceAlgorithms::InPlaceTransform
    (
        allMemFloat, q,
        [] MRAY_HYBRID (Float) -> Float
        {
            return Float(1);
        }
    );

    // Convert to Luminance
    ColorConverter converter(this->gpuSystem);
    converter.ExtractLuminance(dLuminanceData, radianceFieldTextures, q);
    // Generate cdf/pdf for sampling
    for(size_t i = 0; i < radianceFieldTextures.size(); i++)
    {
        if(!radianceFieldTextures[i]) continue;

        pwcDistributions.Construct(uint32_t(i), dLuminanceData[i], q);
    }
    soa.dDistributions = pwcDistributions.DeviceDistributions();
    // All Done!
    q.Barrier().Wait();
}

template <CoordConverterC CC>
size_t LightGroupSkysphere<CC>::GPUMemoryUsage() const
{
    return (Parent::GPUMemoryUsage() +
            pwcDistributions.GPUMemoryUsage());

}