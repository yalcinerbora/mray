#pragma once

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
MetaLightViewT<CH, ML, SC>::MetaLightViewT(const SpectrumConverter& sConverter,
                                           const ML& l)
    : light(l)
    , sConverter(sConverter)
{}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> MetaLightViewT<CH, ML, SC>::SampleSolidAngle(RNGDispenser& rng,
                                                              const Vector3& distantPoint) const
{
    return DeviceVisit(light, [&](auto&& l) -> Float
    {
        return l.SampleSolidAngle(rng, distantPoint);
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Float MetaLightViewT<CH, ML, SC>::PdfSolidAngle(const CH& hit,
                                                const Vector3& distantPoint,
                                                const Vector3& dir) const
{
    return DeviceVisit(light, [=](auto&& l) -> Float
    {
        using HitType = decltype(l)::Primitive::Hit;
        HitType hitIn = hit.template AsVector<HitType::Dims>();
        return l.PdfSolidAngle(hitIn, distantPoint, dir);
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t MetaLightViewT<CH, ML, SC>::SampleSolidAngleRNCount() const
{
    return DeviceVisit(light, [&](auto&& l) -> Float
    {
        return l.SampleSolidAngleRNCount();
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Ray> MetaLightViewT<CH, ML, SC>::SampleRay(RNGDispenser& rng) const
{
    return DeviceVisit(light, [&](auto&& l) -> SampleT<Ray>
    {
        return l.SampleRay(rng);
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Float MetaLightViewT<CH, ML, SC>::PdfRay(const Ray& ray) const
{
    return DeviceVisit(light, [&](auto&& l) -> SampleT<Ray>
    {
        return l.SampleRay(ray);
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t MetaLightViewT<CH, ML, SC>::SampleRayRNCount() const
{
    return DeviceVisit(light, [&](auto&& l) -> SampleT<Ray>
    {
        return l.SampleRayRNCount();
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MetaLightViewT<CH, ML, SC>::EmitViaHit(const Vector3& wO, const CH& hit) const
{
    return DeviceVisit(light, [=](auto&& l) -> Spectrum
    {
        using HitType = decltype(l)::Primitive::Hit;
        return l.Emit(wO, HitType(hit));
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MetaLightViewT<CH, ML, SC>::EmitViaSurfacePoint(const Vector3& wO,
                                                         const Vector3& surfacePoint) const
{
    return DeviceVisit(light, [&](auto&& l) -> SampleT<Ray>
    {
        return specConverter(l.SampleRay(wO, surfacePoint));
    });
}

template<class CH, class ML, class SC>
MRAY_HYBRID MRAY_CGPU_INLINE
bool MetaLightViewT<CH, ML, SC>::IsPrimitiveBackedLight() const
{
    return DeviceVisit(light, [&](auto&& l) -> bool
    {
        return IsPrimitiveBackedLight();
    });
}

template<TransformContextC... TC, LightC... L>
MetaLightArray<Variant<TC...>, Variant<L...>>::MetaLightArray(const GPUSystem& s)
    : system(s)
    , memory(system.AllGPUs(), 2_MiB, 16_MiB)
{}

template<TransformContextC... TC, LightC... L>
template<LightGroupC LightGroup, TransformGroupC TransformGroup>
void MetaLightArray<Variant<TC...>, Variant<L...>>::AddBatch(const LightGroup& lg, const TransformGroup& tg,
                                                             const Span<const PrimitiveKey>& primitiveKeys,
                                                             const Span<const LightKey>& lightKeys,
                                                             const Span<const TransformKey>& transformKeys,
                                                             const Vector2ui& batchRange)
{
    const GPUQueue& queue = system.BestDevice().GetQueue(0);

    using TGSoA = typename TransformGroup::DataSoA;
    using LGSoA = typename LightGroup::DataSoA;
    using PGSoA = typename LightGroup::PrimGroup::DataSoA;

    PGSoA pgData = lg.PrimitiveGroup().SoA();
    LGSoA lgData = lg.SoA();
    TGSoA tgData = tg.SoA();

    uint32_t lightCount = (batchRange[1] - batchRange[0]);
    assert(lightKeys.size() == lightCount);

    // Given light construct the transformed light
    // This means having a primitive context
    auto ConstructKernel = [=, this] MRAY_HYBRID(KernelCallParams kp)
    {
        for(uint32_t i = kp.GlobalId(); i < lightCount; i += kp.TotalSize())
        {
            // Determine types, transform context primitive etc.
            // Compile-time find the transform generator function and return type
            using PrimGroup = typename LightGroup::PrimGroup;
            constexpr auto TContextGen = AcquireTransformContextGenerator<PrimGroup, TransformGroup>();
            constexpr auto TGenFunc = decltype(TContextGen)::Function;
            // Define the types
            // First, this kernel uses a transform context
            // that this primitive group provides to generate a surface
            using TContextType = typename decltype(TContextGen)::ReturnType;
            // Assert that this context is either single-transform or identity-transform
            // Currently, each light can only be transformed via
            // single or or identity transform (no skinned meshes :/)
            static_assert(std::is_same_v<TContextType, TransformContextIdentity> ||
                          std::is_same_v<TContextType, TransformContextSingle>);

            // Second, we are using this primitive
            using Primitive = typename PrimGroup:: template Primitive<TContextType>;

            // Light type has to be with identity spectrum conversion
            // meta light will handle the spectrum conversion instead of the light
            using Light = typename LightGroup:: template Light<TContextType>;
            // Check if the light type is in variant list
            static_assert((std::is_same_v<Light, L> || ...),
                          "This light type is not in variant list!");

            // Find the lights starting location
            uint32_t index = batchRange[0] + i;

            // Primitives do not own the transform contexts,
            // save it to global memory.
            dTContextList[index] = TGenFunc(tgData, pgData,
                                            transformKeys[i],
                                            primitiveKeys[i]);
            // Now construct the primitive, it refers to the tc on global memory
            auto& p = dLightPrimitiveList[index];
            p.template emplace<Primitive>(std::get<TContextType>(dTContextList[index]),
                                          pgData, primitiveKeys[i]);

            // And finally construct the light, and this also refers to primitive
            // on the global memory.
            // Construct the lights with identity spectrum transform
            // context since it depends on per-ray data.
            auto& l = dLightList[index];
            l.template emplace<Light>(dSConverter[0],
                                      std::get<Primitive>(dLightPrimitiveList[index]),
                                      lgData, lightKeys[i]);
        }
    };

    using namespace std::literals;
    queue.IssueSaturatingLambda
    (
        "KCConstructMetaLight"sv,
        KernelIssueParams{.workCount = lightCount},
        //
        std::move(ConstructKernel)
    );
}

template<TransformContextC... TC, LightC... L>
Span<const Variant<std::monostate, L...>>
MetaLightArray<Variant<TC...>, Variant<L...>>::Array() const
{
    return ToConstSpan(dLightList);
}
