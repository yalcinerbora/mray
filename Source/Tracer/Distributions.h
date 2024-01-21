#pragma once

#include <cstdint>
#include "Core/Types.h"
#include "TracerTypes.h"
#include "Device/GPUAlgorithms.h"

// TODO: Maybe move data to textures?
template <uint32_t DIMS>
class DistributionPwC;

template<>
class DistributionPwC<1>
{
    private:
    Span<const Float> dCDF;
    Float             normRecip;
    Float             sizeRecip;

    public:
    MRAY_HYBRID     DistributionPwC(const Span<const Float>& data, Float normRecip);

    MRAY_HYBRID
    SampleT<Float>  SampleIndex(Float xi) const;
    MRAY_HYBRID
    SampleT<Float>  SampleUV(Float xi) const;
    MRAY_HYBRID
    Float           PdfIndex(Float index) const;
    MRAY_HYBRID
    Float           PdfUV(Float uv) const;
    MRAY_HYBRID
    uint32_t        Size() const;
    MRAY_HYBRID
    uint32_t        SizeRecip() const;
};

// Lets see if I can do this fully templated
template <uint32_t DIMS>
class DistributionPwC
{
    public:
    using VectorT   = Vector<DIMS, Float>;
    using SizeT     = Vector<DIMS, uint32_t>;

    private:
    Span<const DistributionPwC<DIMS - 1>>   dNextDistributions;
    DistributionPwC<1>                      dCurrentDistribution;

    public:
    MRAY_HYBRID         DistributionPwC(const Span<const DistributionPwC<DIMS - 1>>& nextDists,
                                        const Span<const DistributionPwC<1>>& myDist);
    MRAY_HYBRID
    SampleT<VectorT>    SampleIndex(const Vector<DIMS, Float>& xi) const;
    MRAY_HYBRID
    SampleT<VectorT>    SampleUV(const Vector<DIMS, Float>& xi) const;
    MRAY_HYBRID
    Float               PdfIndex(const VectorT& index) const;
    MRAY_HYBRID
    Float               PdfUV(const VectorT& index) const;
    MRAY_HYBRID
    SizeT               Size() const;
    MRAY_HYBRID
    VectorT             SizeRecip() const;
};

MRAY_HYBRID MRAY_CGPU_INLINE
DistributionPwC<1>::DistributionPwC(const Span<const Float>& data, Float normRecip)
    : dCDF(data)
    , normRecip(normRecip)
    , sizeRecip(1 / Float{dCDF.size() - 1})
{}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Float> DistributionPwC<1>::SampleIndex(Float xi) const
{
    uint32_t index = DeviceAlgorithms::LowerBound(dCDF, xi);
    Float t = (xi - dCDF[i - 1]) / (dCDF[i] - dCDF[i - 1]);

    return SampleT<Float>
    {
        .pdf = PdfIndex(index),
        .sampledResult = Float{index} + t
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Float> DistributionPwC<1>::SampleUV(Float xi) const
{
    SampleT<Float> s = SampleIndex(xi);
    s.sampledResult *= sizeRecip;
    return s;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float DistributionPwC<1>::PdfIndex(Float index) const
{
    uint32_t indexI = uint32_t{index};
    return (dCDF[indexI + 1] - dCDF[indexI]);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float DistributionPwC<1>::PdfUV(const Float uv) const
{
    return PdfIndex[uv * Float{dCDF.size()}]);
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t DistributionPwC<1>::Size() const
{
    return static_cast<uint32_t>(dCDF.size() - 1);
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t DistributionPwC<1>::SizeRecip() const
{
    return sizeRecip;
}

template <uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
DistributionPwC<D>::DistributionPwC(const Span<const DistributionPwC<DIMS - 1>>& nextDists,
                                    const Span<const DistributionPwC<1>>& myDist)
    : nextDists(nextDists)
    , dCurrentDistribution(myDist)
{}

template <uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<VectorT> DistributionPwC<D>::SampleIndex(const VectorT& xi) const
{
    using NextVecT = Vector<D-1, Float>;

    SampleT<Float> r = dCurrentDistribution.SampleIndex(xi[D - 1]);
    uint32_t index = uint32_t{r.sampledResult};
    SampleT<NextVecT> rN = dNextDistribution[index];

    return SampleT<VectorT>
    {
        .sampledResult = VectorT(rN.sampledResult, rSampledResult),
        .pdf = r.pdf * rN.pdf
    };
}

template <uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<VectorT> DistributionPwC<D>::SampleUV(const VectorT& xi) const
{
    SampleT<Float> r = dCurrentDistribution.SampleIndex(xi[D - 1]);
    uint32_t index = uint32_t{r.sampledResult};
    SampleT<NextVecT> rN = dNextDistribution[index];

    VectorT result(rN.sampledResult, rSampledResult);
    result *= sizeRecip;

    return SampleT<VectorT>
    {
        .sampledResult = result,
        .pdf = r.pdf * rN.pdf
    };
}

template <uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
Float DistributionPwC<D>::PdfIndex(const VectorT& index) const
{
    uint32_t indexInt = static_cast<uint32_t>(index[D - 1]);
    Float pdfA = dNextDistributions[indexInt].PdfIndex(index);
    Float pdfB = dCurrentDistribution.PdfIndex(index[D - 1]);
    return pdfA * pdfB;
}

template <uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
Float DistributionPwC<D>::PdfUV(const VectorT& uv) const
{
    Float indexF = uv[D - 1] * dCurrentDistribution.size();
    uint32_t indexI = static_cast<uint32_t>(indexF);
    Float pdfA = dNextDistributions[indexI].PdfIndex(indexF);
    Float pdfB = dCurrentDistribution.PdfUV(uv[D - 1]);
    return pdfA * pdfB;
}

template <uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
SizeT DistributionPwC<D>::Size() const
{
    return size;
}

template <uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t DistributionPwC<D>::SizeRecip() const
{
    return VectorT(dNextDistributions[0].SizeRecip(),
                   dCurrentDistribution.SizeRecip());
}