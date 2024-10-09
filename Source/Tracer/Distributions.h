#pragma once

#include <cstdint>
#include <type_traits>

#include "Core/Types.h"

#include "TracerTypes.h"

#include "Device/GPUAlgBinarySearch.h"
#include "Device/GPUSystem.h"
#include "Device/GPUMemory.h"

namespace Distribution
{
// TODO: Maybe move data to textures?
template <uint32_t DIMS>
class DistributionPwC;

template<>
class DistributionPwC<1>
{
    private:
    Span<const Float> dCDF;
    Float             sizeRecip;

    public:
    MRAY_HYBRID     DistributionPwC(const Span<const Float>& data);

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
    Float           SizeRecip() const;
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
                                        const DistributionPwC<1>& myDist);
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

static_assert(std::is_move_constructible_v<DistributionPwC<1>>);
static_assert(std::is_move_assignable_v<DistributionPwC<1>>);
static_assert(std::is_move_constructible_v<DistributionPwC<2>>);
static_assert(std::is_move_assignable_v<DistributionPwC<2>>);

class DistributionGroupPwC2D
{
    public:
    using Distribution2D    = Distribution::DistributionPwC<2>;
    using Distribution1D    = Distribution::DistributionPwC<1>;

    struct DistData
    {
        Span<Float>             dCDFsX;
        Span<Float>             dCDFsY;
        Span<Distribution1D>    dDistsX;
        Span<Distribution1D, 1> dDistY;
    };

    struct DistDataConst
    {
        Span<const Float>             dCDFsX;
        Span<const Float>             dCDFsY;
        Span<const Distribution1D>    dDistsX;
        Span<const Distribution1D, 1> dDistY;
    };

    private:
    const GPUSystem&            system;
    DeviceMemory                memory;
    Span<Distribution2D>        dDistributions;
    std::vector<DistData>       distData;
    std::vector<Vector2ui>      sizes;

    protected:
    public:
                                DistributionGroupPwC2D(const GPUSystem&);

    uint32_t                    Reserve(Vector2ui size);
    void                        Commit();
    void                        Construct(uint32_t index,
                                          const Span<const Float>& function);

    Span<const Distribution2D>  DeviceDistributions() const;

    size_t                      GPUMemoryUsage() const;
    // For testing
    DistDataConst               DistMemory(uint32_t index) const;
};

}

namespace Distribution
{

MRAY_HYBRID MRAY_CGPU_INLINE
DistributionPwC<1>::DistributionPwC(const Span<const Float>& data)
    : dCDF(data)
    , sizeRecip(Float{1} / static_cast<Float>(dCDF.size()))
{}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Float> DistributionPwC<1>::SampleIndex(Float xi) const
{
    using namespace DeviceAlgorithms;
    uint32_t index = static_cast<uint32_t>(LowerBound(dCDF, xi));
    Float prevCDF = (index == 0) ? Float{0.0} : dCDF[index - 1];
    Float myCDF = dCDF[index];
    Float t = (xi - prevCDF) / (myCDF - prevCDF);

    Float indexF = static_cast<Float>(index) + t;
    indexF = (indexF < 1.0) ? indexF : Math::PrevFloat(indexF);
    assert(indexF < static_cast<Float>(dCDF.size()));

    Float pdf = (myCDF - prevCDF) * static_cast<Float>(dCDF.size());
    return SampleT<Float>
    {
        .value = indexF,
        .pdf = pdf
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Float> DistributionPwC<1>::SampleUV(Float xi) const
{
    SampleT<Float> s = SampleIndex(xi);
    s.value *= sizeRecip;
    return s;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float DistributionPwC<1>::PdfIndex(Float index) const
{
    assert(index < 1);
    uint32_t indexI = static_cast<uint32_t>(index);
    Float prevCDF = (indexI == 0) ? Float{0.0} : dCDF[indexI - 1];
    Float myCDF = dCDF[indexI];

    Float pdf = (myCDF - prevCDF) * static_cast<Float>(dCDF.size());
    return pdf;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float DistributionPwC<1>::PdfUV(const Float uv) const
{
    return PdfIndex(uv * static_cast<Float>(dCDF.size()));
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t DistributionPwC<1>::Size() const
{
    return static_cast<uint32_t>(dCDF.size());
}

MRAY_HYBRID MRAY_CGPU_INLINE
Float DistributionPwC<1>::SizeRecip() const
{
    return sizeRecip;
}

template <uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
DistributionPwC<D>::DistributionPwC(const Span<const DistributionPwC<D - 1>>& nextDists,
                                    const DistributionPwC<1>& myDist)
    : dNextDistributions(nextDists)
    , dCurrentDistribution(myDist)
{}

template <uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector<D,Float>> DistributionPwC<D>::SampleIndex(const VectorT& xi) const
{
    using NextIndexT = std::conditional_t<(D - 1) == 1, Float, Vector<D - 1, Float>>;
    SampleT<Float> r = dCurrentDistribution.SampleIndex(xi[D - 1]);
    uint32_t index = static_cast<uint32_t>(r.value);

    // Dist<1> asks for Float since there is no Vector<1, T> type
    // Compile time change the statement if "D == 2" (2D Distribution)
    SampleT<NextIndexT> rN;
    if constexpr((D - 1) == 1)
        rN = dNextDistributions[index].SampleIndex(xi[0]);
    else
        rN = dNextDistributions[index].SampleIndex(xi);

    return SampleT<VectorT>
    {
        .value = VectorT(rN.value, r.value),
        .pdf = r.pdf * rN.pdf
    };
}

template <uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector<D, Float>> DistributionPwC<D>::SampleUV(const VectorT& xi) const
{
    SampleT<VectorT> s = SampleIndex(xi);
    s.value *= SizeRecip();
    return s;
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
    Float indexF = uv[D - 1] * dCurrentDistribution.Size();
    uint32_t indexI = static_cast<uint32_t>(indexF);
    Float pdfA = dNextDistributions[indexI].PdfIndex(indexF);
    Float pdfB = dCurrentDistribution.PdfUV(uv[D - 1]);
    return pdfA * pdfB;
}

template <uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
Vector<D, uint32_t> DistributionPwC<D>::Size() const
{
    return SizeT(dNextDistributions[0].Size(),
                 dCurrentDistribution.Size());
}

template <uint32_t D>
MRAY_HYBRID MRAY_CGPU_INLINE
Vector<D, Float> DistributionPwC<D>::SizeRecip() const
{
    return VectorT(dNextDistributions[0].SizeRecip(),
                   dCurrentDistribution.SizeRecip());
}

}

using DistributionGroupPwC2D = Distribution::DistributionGroupPwC2D;