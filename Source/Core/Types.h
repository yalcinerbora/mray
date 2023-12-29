#pragma once

#include <span>
#include <optional>
#include <concepts>

#include "Vector.h"
#include "Ray.h"
#include "AABB.h"
#include "Quaternion.h"

#include "Hit.h"
#include "Key.h"

using MetaHitPtr = MetaHitPtrT<Vector2, Vector3>;

// Differential portion of a ray
class DiffRay{};

class RNGDispenser
{
    public:
    Vector2 NextUniform2D()
    {
        return Vector2::Zero();
    }
};

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
template <int SPS, std::floating_point T>
using SpectrumT = Vector<SPS, T>;

template <int SPS, std::floating_point T>
using SpectrumWavesT = Vector<SPS, T>;

// Actual spectrum for this compilation
// For RGB values this should at least be 3
using Spectrum      = SpectrumT<SpectraPerSpectrum, Float>;
using SpectrumWaves = SpectrumWavesT<SpectraPerSpectrum, Float>;

// Similar to the other types this
class SpectrumGroup
{

};

//using

// Some key types
// these are defined seperately for fine-tuning
// for different use-cases.

// Work key when a ray hit an object
// this key will be used to partition
// rays with respect to materials
using WorkId = KeyT<uint32_t, 16, 16>;

// Medium key
using MediumId = KeyT<uint32_t, 8, 24>;
// Primitive key
using PrimitiveId = KeyT<uint32_t, 4, 28>;
// Transform key
using TransformId = KeyT<uint32_t, 8, 24>;
// Material key
using MaterialId = KeyT<uint32_t, 10, 22>;

using RayIdPack = std::tuple<PrimitiveId, MaterialId, TransformId, MediumId>;

using RayIndex = uint32_t;

template <class T>
struct Sample
{
    T           sampledResult;
    Float       pdf;
};

template <class T>
struct Intersection
{
    Float   tMin;
    T       hit;
};

struct BxDFResult
{
    Ray         wO;
    Spectrum    reflectance;
    MediumId    mediumId;
};

struct VoxelizationParameters
{
    AABB3 sceneExtents;
    Vector3i resolution;
};

// Rename the std::optional, gpu may not like it
// most(all after c++20) of optional is constexpr
// so the "relaxed-constexpr" flag of nvcc will be able to compile it
template <class T>
using Optional = std::optional<T>;

// Same as optional
template <class T, std::size_t Extent = std::dynamic_extent>
using Span = std::span<T, Extent>;

// Most barebone leaf
struct DefaultLeaf
{
    PrimitiveId primId;
};

// It may seem useless but it can be used by debug
// materials where directly material related info
// is queried
using EmptySurface = EmptyType;

struct BasicSurface
{
    Vector3 position;
    Vector3 geoNormal;
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

struct RayReg
{
    Ray     r;
    Vector2 t;

    MRAY_HYBRID         RayReg(const RayGMem* gRays, RayIndex index);
    MRAY_HYBRID void    Update(RayGMem* gRays, RayIndex index) const;
    MRAY_HYBRID void    UpdateTMax(RayGMem* gRays, RayIndex index) const;
};

MRAY_HYBRID MRAY_CGPU_INLINE
RayReg::RayReg(const RayGMem* gRays, RayIndex index)
{
    RayGMem rayGMem = gRays[index];
    r = Ray(rayGMem.dir, rayGMem.pos);
    t = Vector2(rayGMem.tMin, rayGMem.tMax);
}

MRAY_HYBRID MRAY_CGPU_INLINE
void RayReg::Update(RayGMem* gRays, RayIndex index) const
{
    RayGMem rayGMem =
    {
        .pos = r.Pos(),
        .tMin = t[0],
        .dir = r.Dir(),
        .tMax = t[1]
    };
    gRays[index] = rayGMem;
}

MRAY_HYBRID MRAY_CGPU_INLINE
void RayReg::UpdateTMax(RayGMem* gRays, RayIndex index) const
{
    gRays[index].tMax = t[1];
}