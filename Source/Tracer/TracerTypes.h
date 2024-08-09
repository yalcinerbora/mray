#pragma once

#include "Core/Definitions.h"
#include "Core/Vector.h"
#include "Core/Ray.h"
#include "Core/AABB.h"
#include "Core/Quaternion.h"

#include "Hit.h"
#include "Key.h"

template <std::unsigned_integral T>
struct RNGDispenserT;

using RNGDispenser = RNGDispenserT<uint32_t>;

using MetaHit = MetaHitT<2>;

// Differential portion of a ray
class RayDiff{};

// Spectral Samples or RGB color etc.
// For spectrum we need wavelengths as well,
// but for RGB we dont need it.
//
// So frequencies are seperate.
// The intra-render calculations will hold SpectrumT types
// throughout.
//
// On spectral mode, boundary calculations will require accessing
// the wavelengths for sampling (i.e. from RGB texture for input boundary case
// or converting to the output image for output boundary case).
//
// Pairing these two as a single type is not good because of these.
// We hold a seperate entity that holds waves
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

// Some key types
// these are defined seperately for fine-tuning
// for different use-cases.

// Common Key/Index Types
// (TODO: make these compile time def (via cmake input)
// and all the key combination bit sizes)
// User may want to mix and match these
using CommonKey = uint32_t;
using CommonIndex = uint32_t;

// Work key when a ray hit an object
// this key will be used to partition
// rays with respect to materials
using SurfaceWorkKey = TriKeyT<CommonKey, 1, 14, 17>;
using AccelWorkKey = KeyT<CommonKey, 8, 24>;

// Accelerator key
using AcceleratorKey    = KeyT<CommonKey, 12, 20>;
using PrimitiveKey      = KeyT<CommonKey, 4, 28>;
using PrimBatchKey      = KeyT<CommonKey, 4, 28>;
using TransformKey      = KeyT<CommonKey, 8, 24>;
using MediumKey         = KeyT<CommonKey, 8, 24>;
using CameraKey         = KeyT<CommonKey, 8, 24>;
using MaterialKey       = KeyT<CommonKey, 11, 21>;
using LightKey          = MaterialKey;
using LightOrMatKey     = TriKeyT<CommonKey, 1,
                                  LightKey::BatchBits - 1,
                                  LightKey::IdBits>;

static constexpr CommonKey IS_MAT_KEY_FLAG = 0;
static constexpr CommonKey IS_LIGHT_KEY_FLAG = 1;

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

//static constexpr size_t AccelIdPackAlignment = (sizeof(TransformKey) +
//                                                sizeof(AcceleratorKey));
//struct alignas (AccelIdPackAlignment) AcceleratorIdPack
//{
//    TransformKey    transId;
//    AcceleratorKey  accelId;
//};

template <class HitType>
struct IntersectionT
{
    HitType         hit;
    Float           t;
};

struct BxDFResult
{
    Ray         wO;
    Spectrum    reflectance;
    MediumKey   mediumId;
};

struct VoxelizationParameters
{
    AABB3       sceneExtents;
    Vector3i    resolution;
};

// Most barebone leaf
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

struct BarycentricSurface
{
    Vector3 position;
    Vector3 baryCoords;
};

struct DefaultSurface
{
    Vector3     position;
    Vector3     geoNormal;
    Quaternion  shadingTBN;
    Vector2     uv;

    Vector2     dpdu;
    Vector2     dpdv;

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