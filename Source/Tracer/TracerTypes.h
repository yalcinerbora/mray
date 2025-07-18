#pragma once

#include "Core/Definitions.h"
#include "Core/Vector.h"
#include "Core/Ray.h"
#include "Core/AABB.h"
#include "Core/Quaternion.h"
#include "Core/MRayDescriptions.h"

#include "Hit.h"
#include "Key.h"

template <std::unsigned_integral T>
struct RNGDispenserT;

using RNGDispenser = RNGDispenserT<uint32_t>;

using MetaHit = MetaHitT<2>;

// Ray differentials
// https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf
// This paper also explains ray cones which dramatically reduces memory consumption
// (~120 MiB). But it had not have aniso-filtering support.
//
// This paper introduces the aniso (which is the continuation of the paper above)
// https://d1qx31qr3h6wln.cloudfront.net/publications/Akenine-Moller2021LOD.pdf
//
//
// Disadvantages are come coupling between surface and ray
// So we will implement ray cones!
//
// For example, traditional ray differentials use 12 floats!
// For average 2M paths in circulation, this corresponds to 96MiB of data!
// (Maybe more when we do shadow rays)
//struct alignas(16) RayDiff
//{
//    Vector3 dPdx;
//    Vector3 dPdy;
//    Vector3 dDdx;
//    Vector3 dDdy;
//};
//
// TODO: Check and test for half precision. Angle is revolve around (-Pi/2, Pi/2)
// Width is world space related it can be large (but should fit to half)
// aperture can be 16-bit UNORM as well since it has a fixed range. (We can round down
// to increase aliasing, since we do monte carlo simulation)
struct alignas(8) RayCone
{
    Float aperture  = Float(0);
    Float width     = Float(0);

    MRAY_HYBRID
    RayCone Advance(Float t) const;

    MRAY_HYBRID
    std::array<Vector3, 2>
    Project(Vector3 surfaceNormal, Vector3 dirTowards) const;
};

struct RayConeSurface
{
    RayCone rayConeFront;   // Incoming ray cone of the surface
    RayCone rayConeBack;    // Potential refracted ray cone
    Float betaN = Float(0); // Curvature estimation of surface

    MRAY_HYBRID
    RayCone ConeAfterScatter(const Vector3& wO,
                             const Vector3& surfNormal) const;
};

// Image coordinate is little bit special.
// Instead of it being a Vector2 or something
// The integer part and sub pixel part is separated.
//
// This is due to future stochastic filtering (filtering after shading)
// implementation. Some simple renderers may also find this usefull.
// Path tracer; for example, will have a separate RNG state for each pixel
// So a kernel can directly access the state via the integer part.
struct alignas(8) ImageCoordinate
{
    Vector2us   pixelIndex;
    SNorm2x16   offset;

    MRAY_HYBRID
    Vector2 GetPixelIndex() const;
};

// Spectral Samples or RGB color etc.
// For spectrum we need wavelengths as well,
// but for RGB we dont need it.
//
// So frequencies are separate.
// The intra-render calculations will hold SpectrumT types
// throughout.
//
// On spectral mode, boundary calculations will require accessing
// the wavelengths for sampling (i.e. from RGB texture for input boundary case
// or converting to the output image for output boundary case).
//
// Pairing these two as a single type is not good because of these.
// We hold a separate entity that holds waves
//
// TODO: The design leaks here,
// change this to prevent Vector + Spectrum addition etc.
//
// CRTP is the best bet for minimal code duplication (like in PBRT)
template <unsigned int SPS, std::floating_point T>
using SpectrumT = Vector<SPS, T>;

template <unsigned int SPS, std::floating_point T>
using SpectrumWavesT = Vector<SPS, T>;

// Actual spectrum for this compilation
// For RGB values this should at least be 3
using Spectrum      = SpectrumT<SpectraPerSpectrum, Float>;
using SpectrumWaves = SpectrumWavesT<SpectraPerSpectrum, Float>;

// Visible spectrum definitions
inline constexpr Vector2   VisibleSpectrumRange = Vector2(380, 700);
inline constexpr Float     VisibleSpectrumMiddle = VisibleSpectrumRange.Sum() * Float(0.5);

// Invalid spectrum, this will be set when some form of numerical
// error occurs (i.e. a NaN is found). It is specifically over-saturated
// to "pop out" on the image (tonemapper probably darken everything else etc.)
MRAY_HYBRID MRAY_GPU_INLINE
constexpr Vector3 BIG_MAGENTA() { return Vector3(1e7, 0.0, 1e7); }

// Invalid texture fetch, this will be set when streaming texture
// system unable to tap the required texture.
// It is specifically over-saturated to "pop out" on the
// image (tonemapper probably darken everything else etc.)
MRAY_HYBRID MRAY_GPU_INLINE
constexpr Vector3 BIG_CYAN() { return Vector3(0.0, 1e7, 1e7); }

// Some key types
// these are defined separately for fine-tuning
// for different use-cases.

// Common Key/Index Types
// (TODO: make these compile time def (via cmake input)
// and all the key combination bit sizes)
// User may want to mix and match these
using CommonKey = MRay::CommonKey;
using CommonIndex = MRay::CommonIndex;
static constexpr CommonKey CommonKeyBits = sizeof(CommonKey) * CHAR_BIT;

// Work key when a ray hit an object
// this key will be used to partition
// rays with respect to materials
//using SurfaceWorkKey = TriKeyT<CommonKey, 1, 14, 17>;
using AccelWorkKey = KeyT<CommonKey, 8, 24>;

// Accelerator key
using AcceleratorKey    = KeyT<CommonKey, 12, CommonKeyBits - 12>;
using PrimitiveKey      = KeyT<CommonKey,  4, CommonKeyBits -  4>;
using PrimBatchKey      = KeyT<CommonKey,  4, CommonKeyBits -  4>;
using TransformKey      = KeyT<CommonKey,  8, CommonKeyBits -  8>;
using MediumKey         = KeyT<CommonKey,  8, CommonKeyBits -  8>;
using CameraKey         = KeyT<CommonKey,  8, CommonKeyBits -  8>;
using MaterialKey       = KeyT<CommonKey, 11, CommonKeyBits - 11>;
using LightKey          = MaterialKey;
using LightOrMatKey     = TriKeyT<CommonKey, 1,
                                  LightKey::BatchBits - 1,
                                  LightKey::IdBits>;

static constexpr CommonKey IS_MAT_KEY_FLAG = 0u;
static constexpr CommonKey IS_LIGHT_KEY_FLAG = 1u;

static_assert(std::is_same_v<LightKey, MaterialKey>,
              "Material and Light keys must match due to variant like usage");

static_assert(PrimBatchKey::BatchBits == PrimitiveKey::BatchBits,
              "\"PrimBatch\" batch bits (groupId) must be "
              "the same of \"Primitive\" batch bits");

using RayIndex = CommonIndex;

// Quadruplet of Ids
static constexpr size_t HitKeyPackAlignment = (sizeof(PrimitiveKey) +
                                               sizeof(MaterialKey) +
                                               sizeof(TransformKey) +
                                               sizeof(AcceleratorKey));
struct alignas(HitKeyPackAlignment) HitKeyPack
{
    PrimitiveKey    primKey;
    LightOrMatKey   lightOrMatKey;
    TransformKey    transKey;
    AcceleratorKey  accelKey;
};

template <class HitType>
struct IntersectionT
{
    HitType         hit;
    Float           t;
};

struct BxDFResult
{
    Ray         wI;
    Spectrum    reflectance;
    MediumKey   mediumKey;
    bool        isPassedThrough = false;
};

struct VoxelizationParameters
{
    AABB3       sceneExtents;
    Vector3i    resolution;
};

// Most bare bone leaf
struct DefaultLeaf
{
    PrimitiveKey primKey;
};

// It may seem useless but it can be used by debug
// materials where directly material related info
// is queried
using EmptySurface = EmptyType;

struct BasicSurface
{
    Vector3 position;
    Vector3 normal;
};

struct DefaultSurface
{
    Vector3     position;
    Vector3     geoNormal;
    Quaternion  shadingTBN;
    Vector2     uv;

    Vector2     dpdx;
    Vector2     dpdy;
    //
    bool        backSide;
};

struct alignas(32) RayGMem
{
    Vector3     pos;
    Float       tMin;
    Vector3     dir;
    Float       tMax;

};

MRAY_HYBRID MRAY_CGPU_INLINE
Pair<Ray, Vector2> RayFromGMem(Span<const RayGMem> gRays, RayIndex index)
{
    RayGMem rayGMem = gRays[index];
    return std::make_pair(Ray(rayGMem.dir, rayGMem.pos),
                          Vector2(rayGMem.tMin, rayGMem.tMax));
}

MRAY_HYBRID MRAY_CGPU_INLINE
void RayToGMem(Span<RayGMem> gRays, RayIndex index,
               const Ray& r, const Vector2& tMinMax)
{
    RayGMem rayGMem =
    {
        .pos = r.Pos(),
        .tMin = tMinMax[0],
        .dir = r.Dir(),
        .tMax = tMinMax[1]
    };
    gRays[index] = rayGMem;
}

MRAY_HYBRID MRAY_CGPU_INLINE
void UpdateTMax(Span<RayGMem> gRays, RayIndex index, Float tMax)
{
    gRays[index].tMax = tMax;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector2 ImageCoordinate::GetPixelIndex() const
{
    Vector2 result(pixelIndex);
    result += Vector2(offset);
    return result;
}

MRAY_HYBRID MRAY_CGPU_INLINE
RayCone RayCone::Advance(Float t) const
{
    return RayCone
    {
        .aperture = aperture,
        // Exact version
        //.width = width + tanf(aperture * Float(0.5)) * t * Float(2)
        // Approx version in the paper (RT Gems chapter 20, Eq. under Figure 20-5)
        .width = width + aperture * t
    };
}


MRAY_HYBRID MRAY_CGPU_INLINE
RayCone RayConeSurface::ConeAfterScatter(const Vector3& wO, const Vector3& n) const
{
    auto rcFront = RayCone
    {
        .aperture = rayConeFront.aperture + Float(2) * betaN,
        .width = rayConeFront.width
    };
    auto rcBack = RayCone
    {
        .aperture = rayConeBack.aperture - betaN,
        .width = rayConeBack.width
    };

    return (wO.Dot(n) > Float(0)) ? rcFront : rcBack;
}

MRAY_HYBRID MRAY_CGPU_INLINE
std::array<Vector3, 2> RayCone::Project(Vector3 f, Vector3 d) const
{
    // Equation 8, 9;
    // Preprocess the elliptic axes
    Vector3 h1 = d - f.Dot(d) * f;
    Vector3 h2 = Vector3::Cross(f, h1);
    Float r = width * Float(0.5);
    auto EllipseAxes = [&](Vector3 h) -> Vector3
    {
        Float denom = (h - d.Dot(h) * d).Length();
        denom = std::max(MathConstants::Epsilon<Float>(), denom);
        return (r / denom) * h;
    };
    Vector3 a1 = EllipseAxes(h1);
    Vector3 a2 = EllipseAxes(h2);
    return {a1, a2};
}