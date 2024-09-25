#pragma once

#include <unordered_map>

#include "MediumC.h"
#include "ParamVaryingData.h"

#include "Core/DeviceVisit.h"

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"

// Easy part, just wrap the MetaMediumT variant
// on the same interface of material
template<class MetaMediumT,
         class SpectrumTransformer = SpectrumConverterContextIdentity>
class MetaMediumViewT
{
    using SpectrumConverter = typename SpectrumTransformer::Converter;

    private:
    const MetaMediumT&          medium;
    const SpectrumConverter&    sConverter;

    MRAY_HYBRID     MetaMediumViewT(const MetaMediumT& medium,
                                    const SpectrumConverter& specConverter);

    MRAY_HYBRID
    ScatterSample   SampleScattering(const Vector3& wO, RNGDispenser& rng) const;
    MRAY_HYBRID
    Float           PdfScattering(const Vector3& wI, const Vector3& wO) const;
    MRAY_HYBRID
    uint32_t        SampleScatteringRNCount() const;

    MRAY_HYBRID
    Spectrum        IoR(const Vector3& uv) const;
    MRAY_HYBRID
    Spectrum        SigmaA(const Vector3& uv) const;
    MRAY_HYBRID
    Spectrum        SigmaS(const Vector3& uv) const;
    MRAY_HYBRID
    Spectrum        Emission(const Vector3& uv) const;
};

// Hard part, utilize an array that can accept
// arbitrary mediums (given medium is in the type pack of the variant)
// First, create a machinery that get the Medium types from the groups
// we specifically ask for "IdentitySpectrumConverter"
// we we will create the array in initialization time,
// and this will not have a notion of wavelengths.
//
// "MetaMediumView" will transform the item.
//template<MediumC... Media>
//using MediumGroupOf = Tuple<Media::template Medium<>...>;

template<class... >
class MetaMediumArrayT;

// Specialize the array
// Accept a tuple of groups, since user will define a generic
// tuple of groups to transfer the type.
template<MediumGroupC... MediumGroups>
class MetaMediumArrayT<Tuple<MediumGroups...>>
{
    // Fetch the medium of the group and make a variant
    // Also add monostate to make it default constructible.
    using MediumVariant = Variant<std::monostate, typename MediumGroups::template Medium<>...>;
    //using HostKeyMap = std::unordered_map<MediumGroupId, std::vector<MediumKey>>
    static constexpr auto GroupCount = sizeof...(MediumGroups);

    public:
    using MetaMedium = MediumVariant;

    private:
    // Actual backing data, now do we
    const GPUSystem&    gpuSystem;
    DeviceMemory        memory;
    Span<uint32_t>      dGroupStartOffsets;
    Span<MetaMedium>    dMediums;
    //
    uint32_t currentOffset = 0;
    //HostKeyMap          hKeys;

    template<MediumGroupC MediumGroup>
    requires((std::is_same_v<MediumGroup, MediumGroups> || ...))
    void    AddBatchTyped(const MediumGroup& mg,
                          const GPUQueue& queue);

    public:
            MetaMediumArrayT(const GPUSystem&,
                             size_t maximumMediums,
                             size_t maximumMediumGroups);

    void    AddBatch(const GenericGroupMediumT& mg,
                     const GPUQueue& queue);
};

#include "MetaMedium.hpp"