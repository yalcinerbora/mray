#pragma once

// Stochastic Texture Filtering Routines
// https://dl.acm.org/doi/10.1145/3651293
// Supplementary material and the paper
//
// All coordinates are in texel space
// returned values are floats since you may pass it to the GPU as UV
// coordinates.
//
// One thing that I do not understand (I've skimmed the paper and suppl material)
// these routines do not return the weight, so it is pre-normalized I guess?
//
#include "Core/Vector.h"
#include "Core/Math.h"

#include "DistributionFunctions.h"

namespace StochasticTF::Detail
{
    MR_PF_DECL std::array<Float, 4> BSplineWeights(Float t);
}

namespace StochasticTF
{
    // 2D
    MR_PF_DECL Vector2 Bilinear(Vector2 st, Float xi);
    MR_PF_DECL Vector2 Bicubic(Vector2 st, Float xi);
    MR_PF_DECL Vector2 Anisotrophic2D(Vector2 st, Vector2 dpdx, Vector2 dpdy, Float xi);
    // 3D
    MR_PF_DECL Vector3 Trilinear(Vector3 st, Float xi);
    MR_PF_DECL Vector3 Tricubic(Vector3 st, Float xi);
}

MR_PF_DEF
std::array<Float, 4>
StochasticTF::Detail::BSplineWeights(Float t)
{
    Float t2 = t * t;
    Float t3 = t2 * t;
    constexpr Float FACTOR = Float(1) / Float(6);
    std::array<Float, 4> weights;
    weights[0] = FACTOR * (Float(-1) * t3 + Float(3) * t2 - Float(3) * t + Float(1));
    weights[1] = FACTOR * (Float( 3) * t3 + Float(6) * t2 + Float(4));
    weights[2] = FACTOR * (Float(-3) * t3 + Float(3) * t2 - Float(3) * t + Float(1));
    weights[3] = FACTOR * t3;
    //
    return weights;
};

MR_PF_DEF
Vector2 StochasticTF::Bilinear(Vector2 st, Float xi0)
{
    using Distribution::Common::BisectSample1;

    Vector2 bl = Math::Floor(st);
    Vector2 t = st - bl;
    auto [i0, xi1] = BisectSample1(xi0, t[0]);
    auto [i1, _  ] = BisectSample1(xi1, t[1]);
    return bl + Vector2(i0, i1);
}

MR_PF_DEF
Vector2 StochasticTF::Bicubic(Vector2 st, Float xi)
{
    using Distribution::Common::BisectSample;

    std::array<Float, 4> wU = Detail::BSplineWeights(st[0] - Math::Floor(st[0]));
    std::array<Float, 4> wV = Detail::BSplineWeights(st[1] - Math::Floor(st[1]));
    //
    Vector2 bl = Math::Floor(st) - Float(1);
    auto [i0, xi0] = BisectSample<4>(xi , wU, true);
    auto [i1, _  ] = BisectSample<4>(xi0, wV, true);
    return bl + Vector2(i0, i1);
}

MR_PF_DECL
Vector2 StochasticTF::Anisotrophic2D(Vector2, Vector2, Vector2, Float)
{
    //// Listing 4
    //Float A = dpdx[1] * dpdx[1] + dpdy[1] * dpdy[1] + Float(1);
    //Float B = Float(-2) - (dpdx[0] * dpdx[1] + dpdy[0] * dpdy[1]);
    //Float C = dpdx[0] * dpdx[0] + dpdy[0] * dpdy[0] + Float(1);

    ////
    // TODO:
    return Vector2::Zero();
}

MR_PF_DECL
Vector3 StochasticTF::Trilinear(Vector3 st, Float xi)
{
    using Distribution::Common::BisectSample1;

    Float m0 = Math::Floor(st[3]);
    auto [i0, xi0] = BisectSample1(xi, st[3] - m0);
    Vector2 st2D = Bilinear(Vector2(st), xi0);
    return Vector3(st2D, m0 + Float(i0));
}

MR_PF_DECL
Vector3 StochasticTF::Tricubic(Vector3 st, Float xi)
{
    using Distribution::Common::BisectSample1;

    Vector3 bl = Math::Floor(st);
    Vector3 t = st - bl;
    Vector3 ijk = Vector3::Zero();
    //
    for(uint32_t i = 0; i < 3; i++)
    {
        auto w = Detail::BSplineWeights(t[i]);
        Float wSum = w[0];
        // Weighted Reservoir Sampling
        MRAY_UNROLL_LOOP
        for(uint32_t j = 1; j < 4; j++)
        {
            wSum += w[j];
            Float ratio = w[j] / wSum;
            auto [p, xi0] = BisectSample1(xi, ratio);
            if(p == 0) ijk[i] = Float(p);
            //
            xi = xi0;
        }
    }
    return bl - Vector3(1) + ijk;
}