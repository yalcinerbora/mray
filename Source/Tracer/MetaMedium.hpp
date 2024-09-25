#pragma once

template<class MM, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
MetaMediumViewT<MM, ST>::MetaMediumViewT(const MM& m, const SpectrumConverter& sc)
    : medium(m)
    , sConverter(sc)
{}

template<class MM, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
ScatterSample MetaMediumViewT<MM, ST>::SampleScattering(const Vector3& wO, RNGDispenser& rng) const
{
    return DeviceVisit(medium, [&](auto&& m) -> ScatterSample
    {
        return m.SampleScattering(wO);
    });
}

template<class MM, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Float MetaMediumViewT<MM, ST>::PdfScattering(const Vector3& wI, const Vector3& wO) const
{
    return DeviceVisit(medium, [&](auto&& m) -> Float
    {
        return m.PdfScattering(wI, wO);
    });
}

template<class MM, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t MetaMediumViewT<MM, ST>::SampleScatteringRNCount() const
{
    return DeviceVisit(medium, [&](auto&& m) -> uint32_t
    {
        return decltype(m)::SampleScatteringRNCount;
    });
}

template<class MM, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MetaMediumViewT<MM, ST>::IoR(const Vector3& uv) const
{
    return DeviceVisit(medium, [&](auto&& m) -> Spectrum
    {
        return m.IoR(uv);
    });
}

template<class MM, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MetaMediumViewT<MM, ST>::SigmaA(const Vector3& uv) const
{
    return DeviceVisit(medium, [&](auto&& m) -> Spectrum
    {
        return specConverter((m.SigmaA(uv));
    });
}

template<class MM, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MetaMediumViewT<MM, ST>::SigmaS(const Vector3& uv) const
{
    return DeviceVisit(medium, [&](auto&& m) -> Spectrum
    {
        return specConverter((m.SigmaS(uv));
    });
}

template<class MM, class ST>
MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MetaMediumViewT<MM, ST>::Emission(const Vector3& uv) const
{
    return DeviceVisit(medium, [&](auto&& m) -> Spectrum
    {
        return specConverter(m.Emission(uv));
    });
}

template<MediumGroupC... MG>
MetaMediumArrayT< Tuple<MG...>>::MetaMediumArrayT(const GPUSystem& sys,
                                                  size_t maximumMediums,
                                                  size_t maximumMediumGroups)
    : gpuSystem(sys)
    , memory(gpuSystem.AllGPUs(),
             MemAlloc::RequiredAllocation({maximumMediums * sizeof(MetaMedium),
                                           maximumMediumGroups * sizeof(uint32_t)}),
             MemAlloc::RequiredAllocation({maximumMediums * sizeof(MetaMedium),
                                           maximumMediumGroups * sizeof(uint32_t)}))
    , hKeys(MediumKey::BatchMask)
{
    MemAlloc::AllocateMultiData(std::tie(dMediums, dGroupStartOffsets),
                                memory, {maximumMediums, maximumMediumGroups});
}

template<MediumGroupC... MG>
template<MediumGroupC MediumGroup>
requires((std::is_same_v<MediumGroup, MG> || ...))
void MetaMediumArrayT< Tuple<MG...>>::AddBatchTyped(const MediumGroup& mg,
                                                    const GPUQueue& queue)
{
    using MGData = typename MediumGroup::DataSoA;
    using Medium = typename MediumGroup::Medium<>;

    MediumGroupId mgId = mg.GroupId();
    size_t totalMedium = mg.TotalItemCount();
    MGData mgData = mg.SoA();

    //
    auto ConstructKernel = [=, this] MRAY_HYBRID(KernelCallParams kp)
    {
        for(uint32_t i = kp.GlobalId(); i < totalMedium; i += kp.TotalSize())
        {
            // Construct the key by hand
            MediumKey key = MediumKey::CombinedKey(mgId, i);
            MetaMedium& m = dMediums[i];
            m.template emplace<Medium>(SpectrumConverterIdentity{},
                                       mgData, key);
        }
    };
}

template<MediumGroupC... MG>
void MetaMediumArrayT< Tuple<MG...>>::AddBatch(const GenericGroupMediumT& mg,
                                               const GPUQueue& queue)
{
    uint32_t uncalled = 0;
    std::apply([&, this](const auto* mgType) -> void
    {
        using CurType = std::remove_pointer_t<decltype(mgType)>;
        if(mg.Name() == CurType::TypeName())
        {
            AddBatchTyped<CurType>(dynamic_cast<CurType&>(mg), queue);
        }
        else uncalled++;
    }, Tuple<const MG*...>);

    if(uncalled == GroupCount)
    {
        throw MRayError("Unkown generic medium (Id:{}) is given to MetaMedium",
                        mg.GroupId());
    }
}