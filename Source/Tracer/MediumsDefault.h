#pragma once

#include "MediumC.h"


namespace MediumDetail
{
    class MediumVacuum
    {
        public:
        using DataSoA = EmptyType;
        using SpectrumConverter = typename SpectrumConverterContextIdentity::Converter;

        private:
        public:
        MRAY_HYBRID MediumVacuum(const SpectrumConverter& sTransContext,
                                 const DataSoA&, MediumKey);


        // TODO: Designing this requires paper reading
        // Currently I've no/minimal information about this topic.
        // We need to do a ray marching style approach probably.
        // Instead of doing a single thread per scatter,
        // We can do single warp (or even block?, prob too much)
        // per ray.
        //
        // I've checked the PBRT book/code, it does similar but code
        // is hard to track.
        //
        // All in all,
        // Medium generates an iterator, (for homogeneous it is the full ray)
        // for spatially varying media it is dense grid and does DDA march over it.
        //
        // Iterator calls a callback function that does the actual work,
        // It can prematurely terminate the iteration due to scattering/absrobtion etc.
        // March logic should not be here it will be the renderer's responsibility
        // Phase function should be here so we need a scatter function
        // that creates a ray.
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

        // TODO:
        // We need to expose the iterator in a different way here, because we may
        // dedicate a warp to handle a single ray, so code should abstract it away
    };
}

class MediumGroupVacuum : public GenericGroupMedium<MediumGroupVacuum>
{
    public:
    using DataSoA   = EmptyType;

    template<class STContext = SpectrumConverterContextIdentity>
    using Medium  = MediumDetail::MediumVacuum;

    public:
    static std::string_view TypeName();

                    MediumGroupVacuum(uint32_t groupId,
                                      const GPUSystem&,
                                      const TextureViewMap&);

    //
    void            CommitReservations() override;

    AttribInfoList  AttributeInfo() const override;
    void            PushAttribute(MediumKey id,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(MediumKey id,
                                  uint32_t attributeIndex,
                                  const Vector2ui& subRange,
                                  TransientData data,
                                  const GPUQueue& queue) override;
    void            PushAttribute(MediumKey idStart, MediumKey idEnd,
                                  uint32_t attributeIndex,
                                  TransientData data,
                                  const GPUQueue& queue) override;

    // Extra
    void            PushTexAttribute(MediumKey idStart, MediumKey idEnd,
                                     uint32_t attributeIndex,
                                     TransientData,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(MediumKey idStart, MediumKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) override;
    void            PushTexAttribute(MediumKey idStart, MediumKey idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<TextureId>,
                                     const GPUQueue& queue) override;

    DataSoA         SoA() const;
};

#include "MediumsDefault.hpp"

static_assert(MediumC<MediumDetail::MediumVacuum>);
static_assert(MediumGroupC<MediumGroupVacuum>);