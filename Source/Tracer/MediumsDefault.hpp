#pragma once

namespace MediumDetail
{

MRAY_HYBRID MRAY_CGPU_INLINE
ScatterSample MediumVacuum::SampleScattering(const Vector3&, RNGDispenser&) const
{
    return ScatterSample
    {
        .sampledResult =
        {
            .wI = Vector3::Zero(),
            .phaseVal = Float(0.0)
        },
        .pdf = Float(0.0)
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float MediumVacuum::PdfScattering(const Vector3&, const Vector3&) const
{
    return Float(0.0);
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t MediumVacuum::SampleScatteringRNCount() const
{
    return 0;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MediumVacuum::IoR(const Vector3&) const
{
    return Spectrum(1.0);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MediumVacuum::SigmaA(const Vector3&) const
{
    return Spectrum::Zero();
}

MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MediumVacuum::SigmaS(const Vector3&) const
{
    return Spectrum::Zero();
}

MRAY_HYBRID MRAY_CGPU_INLINE
Spectrum MediumVacuum::Emission(const Vector3&) const
{
    return Spectrum::Zero();
}

}