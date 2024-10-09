#pragma once

#include "Core/Algorithm.h"

inline std::vector<LightPartition>
MetaLightListConstructionParams::Partition() const
{
    std::vector<LightPartition> partitionList;
    partitionList.reserve(lightGroups.size());

    auto start = lSurfList.begin();
    while(start != lSurfList.end())
    {
        CommonKey lGroupId = LightGroupIdFetcher()(start->second.lightId);
        auto end = std::upper_bound(start, lSurfList.end(), lGroupId,
        [](CommonKey value, const LightSurfPair& surf)
        {
            CommonKey batchPortion = LightGroupIdFetcher()(surf.second.lightId);
            return value < batchPortion;
        });

        auto& slot = partitionList.emplace_back(LightPartition
        {
            .lgId = LightGroupId(lGroupId),
            .ltPartitions = {}
        });
        slot.ltPartitions.reserve(transformGroups.size());

        // Sub-partition wrt. transform
        auto innerStart = start;
        while(innerStart != end)
        {
            TransformId tId = innerStart->second.transformId;
            CommonKey tGroupId = TransGroupIdFetcher()(tId);
            auto innerEnd = std::upper_bound(innerStart, end, tGroupId,
            [](CommonKey value, const LightSurfPair& surf) -> bool
            {
                auto tId = surf.second.transformId;
                return (value < TransGroupIdFetcher()(tId));
            });

            size_t elemCount = static_cast<size_t>(std::distance(innerStart, innerEnd));
            size_t startDistance = static_cast<size_t>(std::distance(lSurfList.begin(), innerStart));
            slot.ltPartitions.emplace_back(TransGroupId(tGroupId),
                                           lSurfList.subspan(startDistance, elemCount));
            innerStart = innerEnd;
        }
        start = end;
    };
    return partitionList;
}

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
    return DeviceVisit(light, [&](auto&& l) -> SampleT<Vector3>
    {
        using T = std::remove_cvref_t<decltype(l)>;
        if constexpr(std::is_same_v<T, std::monostate>)
            return SampleT<Vector3>{};
        else return l.SampleSolidAngle(rng, distantPoint);
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
        using T = std::remove_cvref_t<decltype(l)>;
        if constexpr(std::is_same_v<T, std::monostate>)
            return Float(0);
        else
        {
            using HitType = decltype(l)::Primitive::Hit;
            HitType hitIn = hit.template AsVector<HitType::Dims>();
            return l.PdfSolidAngle(hitIn, distantPoint, dir);
        }
    });
}

template<class V, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t MetaLightViewT<V, ST>::SampleSolidAngleRNCount() const
{
    return DeviceVisit(light, [&](auto&& l) -> uint32_t
    {
        using T = std::remove_cvref_t<decltype(l)>;
        if constexpr(std::is_same_v<T, std::monostate>)
            return 0;
        else return l.SampleSolidAngleRNCount();
    });
}

template<class V, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Ray> MetaLightViewT<V, ST>::SampleRay(RNGDispenser& rng) const
{
    return DeviceVisit(light, [&](auto&& l) -> SampleT<Ray>
    {
        using T = std::remove_cvref_t<decltype(l)>;
        if constexpr(std::is_same_v<T, std::monostate>)
            return SampleT<Ray>{};
        else return l.SampleRay(rng);
    });
}

template<class V, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Float MetaLightViewT<V, ST>::PdfRay(const Ray& ray) const
{
    return DeviceVisit(light, [&](auto&& l) -> Float
    {
        using T = std::remove_cvref_t<decltype(l)>;
        if constexpr(std::is_same_v<T, std::monostate>)
            return Float(0);
        else return l.SampleRay(ray);
    });
}

template<class V, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t MetaLightViewT<V, ST>::SampleRayRNCount() const
{
    return DeviceVisit(light, [&](auto&& l) -> uint32_t
    {
        using T = std::remove_cvref_t<decltype(l)>;
        if constexpr(std::is_same_v<T, std::monostate>)
            return 0;
        else return l.SampleRayRNCount();
    });
}

template<class V, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MetaLightViewT<V, ST>::EmitViaHit(const Vector3& wO,
                                           const MetaHit& hit) const
{
    return DeviceVisit(light, [=](auto&& l) -> Spectrum
    {
        using T = std::remove_cvref_t<decltype(l)>;
        if constexpr(std::is_same_v<T, std::monostate>)
            return Spectrum::Zero();
        else
        {
            using HitType = decltype(l)::Primitive::Hit;
            return specConverter(l.Emit(wO, HitType(hit)));
        }
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
typename MetaLightArrayT<TLT...>::View::MetaLightView<STransformer>
MetaLightArrayT<TLT...>::View::operator()(const typename STransformer::Converter& sc,
                                          uint32_t i) const
{
    return MetaLightView<STransformer>(dMetaLights[i], sc);
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
                                       const GPUQueue& queue)
{
    uint32_t lightCount = static_cast<uint32_t>(lightKeys.size());
    assert(primitiveKeys.size() == lightKeys.size() &&
           lightKeys.size() == transformKeys.size());
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
                     queue);
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
void MetaLightArrayT<TLT...>::Construct(MetaLightListConstructionParams params,
                                        const GPUQueue& queue)
{
    assert(std::is_sorted(params.lSurfList.begin(), params.lSurfList.end(),
                          LightSurfaceLessThan));
    std::vector<LightPartition> partitions = params.Partition();
    // Find allocation size
    std::vector<LightKey> hLKList;
    std::vector<PrimitiveKey> hPKList;
    std::vector<TransformKey> hTKList;
    std::vector<Vector2ui> primExpandedRanges;
    hLKList.reserve(1024);
    hPKList.reserve(1024);
    hTKList.reserve(1024);

    uint32_t offset = 0;
    for(const auto& p : partitions)
    {
        LightGroupId lgId = p.lgId;
        for(const auto& ltP : p.ltPartitions)
        {
            primExpandedRanges.push_back(Vector2ui(offset, 0));
            for(const auto& surf : ltP.second)
            {
                LightKey lKey = std::bit_cast<LightKey>(surf.second.lightId);
                TransformKey tKey = std::bit_cast<TransformKey>(surf.second.transformId);
                //
                const auto& lg = *params.lightGroups.at(lgId).value().get().get();
                PrimBatchKey primBatchKey = lg.LightPrimBatch(lKey);
                if(lg.IsPrimitiveBacked())
                {
                    Vector2ui batchRange = lg.GenericPrimGroup().BatchRange(primBatchKey);
                    uint32_t primCount = batchRange[1] - batchRange[0];
                    offset += primCount;
                    // Add keys
                    for(uint32_t i = 0; i < primCount; i++)
                    {
                        hLKList.push_back(lKey);
                        hTKList.push_back(tKey);
                        auto pKey = PrimitiveKey::CombinedKey(primBatchKey.FetchBatchPortion(),
                                                              batchRange[0] + i);
                        hPKList.push_back(pKey);
                    }
                }
                else
                {
                    hLKList.push_back(lKey);
                    hTKList.push_back(tKey);
                    CommonKey emptyGroupKey = static_cast<CommonKey>(TracerConstants::EmptyPrimGroupId);
                    auto pKey = PrimitiveKey::CombinedKey(emptyGroupKey, 0u);
                    hPKList.push_back(pKey);
                }
            }
            primExpandedRanges.back()[1] = offset;
        }
    }
    size_t totalLightCount = offset;

    // Allocate for keys
    Span<LightKey>        dLKList;
    Span<PrimitiveKey>    dPKList;
    Span<TransformKey>    dTKList;
    DeviceLocalMemory tempMem(*queue.Device());
    MemAlloc::AllocateMultiData(std::tie(dLKList, dPKList, dTKList),
                                tempMem,
                                {totalLightCount,
                                 totalLightCount,
                                 totalLightCount});
    // Do the internal allocation as well
    MemAlloc::AllocateMultiData(std::tie(dPrimSoA, dLightSoA, dTransSoA,
                                         dMetaPrims, dMetaTContexts, dMetaLights),
                                memory,
                                {primExpandedRanges.size(),
                                 primExpandedRanges.size(),
                                 primExpandedRanges.size(),
                                 totalLightCount,
                                 totalLightCount,
                                 totalLightCount});

    queue.MemcpyAsync(dLKList, Span<const LightKey>(hLKList));
    queue.MemcpyAsync(dPKList, Span<const PrimitiveKey>(hPKList));
    queue.MemcpyAsync(dTKList, Span<const TransformKey>(hTKList));

    for(Vector2ui range : primExpandedRanges)
    {
        LightGroupId lgId = LightGroupId(hLKList[range[0]].FetchBatchPortion());
        TransGroupId tgId = TransGroupId(hTKList[range[0]].FetchBatchPortion());
        const auto& lg = *params.lightGroups.at(lgId).value().get().get();
        const auto& tg = *params.transformGroups.at(tgId).value().get().get();

        AddBatchGeneric(lg, tg,
                        dPKList.subspan(range[0], range[1] - range[0]),
                        dLKList.subspan(range[0], range[1] - range[0]),
                        dTKList.subspan(range[0], range[1] - range[0]),
                        queue);
    }
    queue.Barrier().Wait();
}

template<LightTransPairC... TLT>
typename MetaLightArrayT<TLT...>::View
MetaLightArrayT<TLT...>::Array() const
{
    return ToConstSpan(dMetaLights);
}
