#pragma once

#include "SpectrumContext.h"

namespace Jacob2019Detail
{

MR_GF_DEF
Vector3 Converter::FetchCoeffs(uint32_t i, Vector3 uv) const
{
    // TODO: This header leaks to full .cpp file and on CUDA, it cannot
    // find tex3D function. We need to properly adjust our headers
    // for a long time, it requires changing the class strcuture and
    // seperating CPU and GPU code to different headers.
    // For now we just skip the instantiation
    #if defined MRAY_DEVICE_CODE_PATH || defined MRAY_GPU_BACKEND_CPU
        const auto& t = data.lut[i];
        Float c0 = t.c0(uv);
        Float c1 = t.c1(uv);
        Float c2 = t.c2(uv);
        return Vector3(c0, c1, c2);
    #else
        // This is to stop compiler to warn about unused variables
        return uv + Vector3(i);
    #endif
}

MR_HF_DEF
Converter::Converter(SpectrumWaves wavelengths,
                     const Data& data)
    : data(data)
    , wavelengths(wavelengths)
{}

MR_GF_DEF
Spectrum Converter::ConvertAlbedo(const Vector3& rgb) const noexcept
{
    auto EvalPolynomial = [](Vector3 coeffs, Float lambda)
    {
        Float t = Math::FMA(coeffs[0], lambda, coeffs[1]);
        Float x = Math::FMA(t, lambda, coeffs[2]);
        assert(Math::IsFinite(x));

        Float denomRecip = Math::RSqrt(Math::FMA(x, x, Float(1)));
        Float result = Math::FMA(Float(0.5) * x, denomRecip, Float(0.5));
        return result;
    };

    // PBRT has this shortcut for grayscale values,
    // I assume LUT fails? (But why exact comparison then?)
    //
    // Gray data assumed to have a flat response over wavelengths
    if(rgb[0] == rgb[1] && rgb[1] == rgb[2])
        return Spectrum(rgb[0]);


    uint32_t maxI = rgb.Maximum();
    Float maxChannel = rgb[maxI];
    Vector3 xyz = Vector3::Zero();
    if(maxChannel > MathConstants::SmallEpsilon<Float>())
    {
        Float maxChannelFactor = Float(1) / maxChannel;
        xyz = Vector3(rgb[(maxI + 1) % 3] * maxChannelFactor,
                      rgb[(maxI + 2) % 3] * maxChannelFactor,
                      Float(0));
    }

    // Ok, we can do a binary search just like the paper
    // but binary search over 64 variables to fetch a texture feels costly.
    // (We have slight performance problems with the PWC distribution sample over
    // HDR environment map already)
    //
    // It seems Smoothstep has analytic solution for its inverse.
    // https://iquilezles.org/articles/ismoothstep/
    //
    // But now solution has asin and sin, also do we really need
    // perfect inverse? (Already we use LUT and interpolation, such
    // perfection may not be a necessity)
    //
    // I've find this approximation, the link below explains the process
    // (kudos to the author of the blog).
    // https://iradicator.com/fast-inverse-smoothstep/
    //
    // TODO:
    // Now, it is not free (1-2 division and multiple FMAs probably)
    // Approx sine in CUDA is 4x slower than basic instructions throughput-wise.
    // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions
    // Don't know inverse sin so I can't make an informed decision
    // if we should use approximate or accurate version.
    //
    // However; I have assembled a desmos to validate
    // https://www.desmos.com/calculator/nkszevum7u and
    // approx version is not accurate around edges (0.01 < x || x > 0.99)
    // With default parameters, (64x64x64 table) we lose around 12 tables
    // (6 on each edge, out of 64) which will not be used at all.
    // All in all, we use accurate version that has transcendentals.
    // TODO: Measure and check the other one.
    xyz[2] = Math::InvSmoothstep(Math::InvSmoothstep(maxChannel));

    // Another thing is that we want to use GPU's texture unit,
    // so we feed normalized texture coordinates to the system (this is
    // not necessary, we can feed texel coordinates as well but we already
    // have normalized values).
    //
    // However; texture system assumes (0,0) is the edge of the texture
    // but not the first element. So we need to adjust for that.
    //
    // That is why the N is compile time constant.
    // so we constexpr some factors and minimize register/runtime cost.
    constexpr Float N = Float(Data::N);
    constexpr Float A = Float(N - 1) / Float(N);
    constexpr Float B = Float(0.5)   / Float(N);
    Vector3 uv = xyz * A + B;

    // And finally, texture fetch for coefficients
    // and evaluation of the polynomial
    Vector3 coeffs = FetchCoeffs(maxI, uv);
    Spectrum result;
    static constexpr auto WAVE_COUNT = SpectrumWaves::Dims;
    MRAY_UNROLL_LOOP_N(WAVE_COUNT)
    for(uint32_t i = 0; i < WAVE_COUNT; i++)
        result[i] = EvalPolynomial(coeffs, wavelengths[i]);

    return result;
}

MR_GF_DEF
Spectrum Converter::ConvertRadiance(const Vector3& radiance) const noexcept
{
    // TODO: I do not remember what is this for?
    // Should we return identitiy?
    // Fail here for now;
    assert(false);
    return Spectrum(radiance, Float(0));
}

MR_PF_DEF
SpectrumWaves Converter::Wavelengths() const noexcept
{
    return wavelengths;
}

}