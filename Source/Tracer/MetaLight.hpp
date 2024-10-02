#pragma once

template<class V, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
MetaLightViewT<V, ST>::MetaLightViewT(const V& v, const SpectrumConverter& sc)
    : light(v)
    , sConverter(sc)
{}

template<class V, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> MetaLightViewT<V, ST>::SampleSolidAngle(RNGDispenser& rng,
                                                              const Vector3& distantPoint) const
{
    return DeviceVisit(light, [&](auto&& l) -> Float
    {
        return l.SampleSolidAngle(rng, distantPoint);
    });
}

template<class V, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Float MetaLightViewT<V, ST>::PdfSolidAngle(const MetaHit& hit,
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

template<class V, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t MetaLightViewT<V, ST>::SampleSolidAngleRNCount() const
{
    return DeviceVisit(light, [&](auto&& l) -> Float
    {
        return l.SampleSolidAngleRNCount();
    });
}

template<class V, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Ray> MetaLightViewT<V, ST>::SampleRay(RNGDispenser& rng) const
{
    return DeviceVisit(light, [&](auto&& l) -> SampleT<Ray>
    {
        return l.SampleRay(rng);
    });
}

template<class V, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Float MetaLightViewT<V, ST>::PdfRay(const Ray& ray) const
{
    return DeviceVisit(light, [&](auto&& l) -> SampleT<Ray>
    {
        return l.SampleRay(ray);
    });
}

template<class V, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t MetaLightViewT<V, ST>::SampleRayRNCount() const
{
    return DeviceVisit(light, [&](auto&& l) -> SampleT<Ray>
    {
        return l.SampleRayRNCount();
    });
}

template<class V, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MetaLightViewT<V, ST>::EmitViaHit(const Vector3& wO,
                                           const MetaHit& hit) const
{
    return DeviceVisit(light, [=](auto&& l) -> Spectrum
    {
        using HitType = decltype(l)::Primitive::Hit;
        return specConverter(l.Emit(wO, HitType(hit)));
    });
}

template<class V, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MetaLightViewT<V, ST>::EmitViaSurfacePoint(const Vector3& wO,
                                                    const Vector3& surfacePoint) const
{
    return DeviceVisit(light, [&](auto&& l) -> SampleT<Ray>
    {
        return specConverter(l.SampleRay(wO, surfacePoint));
    });
}

template<class V, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
bool MetaLightViewT<V, ST>::IsPrimitiveBackedLight() const
{
    return DeviceVisit(light, [&](auto&& l) -> bool
    {
        return IsPrimitiveBackedLight();
    });
}

template<LightTransPairC... TLT>
MetaLightArrayT<TLT...>::View::View(Span<const MetaLight> dLights)
    : dMetaLights(dLights)
{}

template<LightTransPairC... TLT>
template<class STransformer>
MRAY_HYBRID MRAY_CGPU_INLINE
typename MetaLightArrayT<TLT...>::MetaLightView<STransformer>
MetaLightArrayT<TLT...>::View::operator()(const typename STransformer::Converter& sc,
                                          uint32_t i) const
{
    return MetaLightView(dMetaLights[i], sc);
}

template<LightTransPairC... TLT>
MRAY_HYBRID
uint32_t MetaLightArrayT<TLT...>::View::Size() const
{
    return static_cast<uint32_t>(dMetaLights.size());
}

template<LightTransPairC... TLT>
MetaLightArrayT<TLT...>::MetaLightArrayT(const GPUSystem& s)
    : system(s)
    , memory(system.AllGPUs(), 2_MiB, 16_MiB)
{}

template<LightTransPairC... TLT>
template<LightGroupC LightGroup, TransformGroupC TransformGroup>
void MetaLightArrayT<TLT...>::AddBatch(const LightGroup& lg, const TransformGroup& tg,
                                       const Span<const PrimitiveKey>& primitiveKeys,
                                       const Span<const LightKey>& lightKeys,
                                       const Span<const TransformKey>& transformKeys,
                                       const Vector2ui& lightKeyRange,
                                       const GPUQueue& queue)
{
    using TGSoA = typename TransformGroup::DataSoA;
    using LGSoA = typename LightGroup::DataSoA;
    using PGSoA = typename LightGroup::PrimGroup::DataSoA;

    // Copy the SoA's to gpu memory
    PGSoA pgData = lg.PrimitiveGroup().SoA();
    LGSoA lgData = lg.SoA();
    TGSoA tgData = tg.SoA();
    Span<Byte> dPrimSoAWriteRegion(dPrimSoA.subspan(soaCounter, 1)[0].data(),
                                   sizeof(PGSoA));
    Span<const Byte> dPrimSoAReadRegion(reinterpret_cast<const Byte*>(&pgData),
                                        sizeof(PGSoA));
    //
    Span<Byte> dLightSoAWriteRegion(dLightSoA.subspan(soaCounter, 1)[0].data(),
                                    sizeof(LGSoA));
    Span<const Byte> dLightSoAReadRegion(reinterpret_cast<const Byte*>(&lgData),
                                        sizeof(LGSoA));
    //
    Span<Byte> dTransSoAWriteRegion(dTransSoA.subspan(soaCounter, 1)[0].data(),
                                    sizeof(TGSoA));
    Span<const Byte> dTransSoAReadRegion(reinterpret_cast<const Byte*>(&tgData),
                                         sizeof(TGSoA));
    queue.MemcpyAsync(dPrimSoAWriteRegion, dPrimSoAReadRegion);
    queue.MemcpyAsync(dTransSoAWriteRegion, dTransSoAReadRegion);
    queue.MemcpyAsync(dLightSoAWriteRegion, dLightSoAReadRegion);

    uint32_t lightCount = (lightKeyRange[1] - lightKeyRange[0]);
    assert(lightKeys.size() == lightCount);

    // Given light construct the transformed light
    // This means having a primitive context
    auto ConstructKernel = [=, this] MRAY_HYBRID(KernelCallParams kp)
    {
        // Determine types, transform context primitive etc.
        // Compile-time find the transform generator function and return type
        using PrimGroup = typename LightGroup::PrimGroup;
        using Primitive = MetaLightDetail::PrimType<LightGroup, TransformGroup>;
        using Light     = MetaLightDetail::LightType<LightGroup, TransformGroup>;
        using TContext  = MetaLightDetail::TContextType<LightGroup, TransformGroup>;
        // Context generator of the prim group
        constexpr auto GenerateTContext = AcquireTransformContextGenerator<PrimGroup, TransformGroup>();
        // SoA Ptrs
        const PGSoA* dPGData = reinterpret_cast<const PGSoA*>(dPrimSoA.subspan(soaCounter, 1)[0].data());
        const LGSoA* dLGData = reinterpret_cast<const LGSoA*>(dLightSoA.subspan(soaCounter, 1)[0].data());
        const TGSoA* dTGData = reinterpret_cast<const TGSoA*>(dTransSoA.subspan(soaCounter, 1)[0].data());

        for(uint32_t i = kp.GlobalId(); i < lightCount; i += kp.TotalSize())
        {
            // Find the lights starting location
            uint32_t index = lightCounter + i;
            // Primitives do not own the transform contexts,
            // save it to global memory.
            Byte* tContextLocation = dMetaTContexts[index].data();
            TContext* tContext = new(tContextLocation) TContext(GenerateTContext(*dTGData, *dPGData,
                                                                                  transformKeys[i],
                                                                                  primitiveKeys[i]));
            // Now construct the primitive, it refers to the tc on global memory
            Byte* primLocation = dMetaPrims[index].data();
            Primitive* prim = new(primLocation) Primitive(*tContext, *dPGData, primitiveKeys[i]);

            // And finally construct the light, and this also refers to primitive
            // on the global memory. This will be the variant
            // unlike the other two, we will use this to call member function
            // Construct the lights with identity spectrum transform
            // context since it depends on per-ray data.
            auto& l = dMetaLights[index];
            l.template emplace<Light>(dSpectrumConverter[0],
                                      *prim, *dLGData, lightKeys[i]);
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

    // Need to wait for async memcopies
    queue.Barrier().Wait();
}

template<LightTransPairC... TLT>
void MetaLightArrayT<TLT...>::AddBatchGeneric(const GenericGroupLightT& lg,
                                              const GenericGroupTransformT& tg,
                                              const Span<const PrimitiveKey>& dPrimitiveKeys,
                                              const Span<const LightKey>& dLightKeys,
                                              const Span<const TransformKey>& dTransformKeys,
                                              const Vector2ui& lightKeyRange,
                                              const GPUQueue& queue)
{
    // https://stackoverflow.com/questions/16387354/template-tuple-calling-a-function-on-each-element/37100197#37100197
    // Except that implementation can be optimized out?
    // Instead we use parameter pack expansion.
    uint32_t uncalled = 0;
    auto Call = [&, this](auto* tuple) -> void
    {
        using TupleType = std::remove_pointer_t<decltype(tuple)>;
        using LGType = std::tuple_element_t<0, TupleType>;
        using TGType = std::tuple_element_t<1, TupleType>;

        if(LGType::TypeName() == lg.Name() &&
           TGType::TypeName() == tg.Name())
        {
            AddBatch(dynamic_cast<const LGType&>(lg),
                     dynamic_cast<const TGType&>(tg),
                     dPrimitiveKeys, dLightKeys, dTransformKeys,
                     lightKeyRange, queue);
        }
        else uncalled++;
    };

    std::apply([&](auto... x)
    {
        // Parameter pack expansion
        (
            (void)Call(x),
            ...
        );
    }, TLGroupPtrTuple{});

    if(uncalled == GroupCount)
    {
        throw MRayError("Unkown light/transform group pair (Id:{}/{}) is given to MetaLightArray",
                        lg.GroupId(), tg.GroupId());
    }
}

template<LightTransPairC... TLT>
typename MetaLightArrayT<TLT...>::View
MetaLightArrayT<TLT...>::Array() const
{
    return ToConstSpan(dMetaLights);
}
