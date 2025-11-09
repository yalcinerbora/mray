#pragma once

#include "Vector.h"
#include "Types.h"
#include "BitFunctions.h"

namespace Graphics
{
    // Convention of wi (v) and normal is as follows
    //        normal
    // wi(v)         wo(out)
    //     ^    ^    ^
    //      \   |   /
    //       \  |  /
    //        \ | /
    //   --------------- Boundary
    //
    // So wi should be aligned with the normal, it is callers responsibility
    // to provide the appropriate normal. It returns wo, again outwards from the
    // surface.
    MR_HF_DECL constexpr
    Vector3 Reflect(const Vector3& normal, const Vector3& v);

    // Convention of wi (v) and normal is as follows
    //          wo (out)
    //          ^
    //         /
    //        /    etaTo (IOR)
    //--------------- Boundary
    //     /  |    etaFrom (IOR)
    //    /   |
    //   /    |
    //  v     v
    //  wi  normal
    //
    // So wi should be aligned with the normal, it is caller's responsibility
    // to provide the appropriate normal. It returns wo again "outwards" from the
    // surface. Both "normal" and "wi" is assumed to be normalized.
    MR_HF_DECL constexpr
    Optional<Vector3> Refract(const Vector3& normal, const Vector3& v,
                              Float etaFrom, Float etaTo);

    // Changes the direction of vector "v" towards n
    MR_HF_DECL constexpr
    Vector3 Orient(const Vector3& v, const Vector3& n);

    // Return an orthogonal vector
    MR_HF_DECL constexpr
    Vector3 OrthogonalVector(const Vector3&);

    // Coordinate Conversions
    // Spherical <-> Cartesian
    // Naming are in **physical** notation/convention,
    // Theta : azimuth (-pi <= theta <= pi)
    //         "Greenwich" lies on XZ plane towards +X
    //
    // Phi   : inclination (0 <= phi <= pi)
    //         0 is the "north pole"
    //
    // Results lie on the scaled unit sphere where north pole is +Z
    MR_HF_DECL constexpr
    Vector3 SphericalToCartesian(const Vector3&);
    MR_HF_DECL constexpr
    Vector3 CartesianToSpherical(const Vector3&);
    MR_HF_DECL constexpr
    Vector3 UnitSphericalToCartesian(const Vector2&);
    MR_HF_DECL constexpr
    Vector3 UnitSphericalToCartesian(const Vector2& sinCosTheta,
                                     const Vector2& sinCosPhi);
    MR_HF_DECL constexpr
    Vector2 CartesianToUnitSpherical(const Vector3&);

    // 2D Variant of spherical coordinates
    MR_HF_DECL constexpr
    Vector2 PolarToCartesian(const Vector2&);
    MR_HF_DECL constexpr
    Vector2 CartesianToPolar(const Vector2&);
    MR_HF_DECL constexpr
    Vector2 UnitPolarToCartesian(Float);
    MR_HF_DECL constexpr
    Float   CartesianToUnitPolar(const Vector2&);

    // Concentric Octahedral Mapping
    // https://fileadmin.cs.lth.se/graphics/research/papers/2008/simdmapping/clarberg_simdmapping08_preprint.pdf
    MR_HF_DECL constexpr
    Vector2 DirectionToConcentricOctahedral(const Vector3&);
    MR_HF_DECL constexpr
    Vector3 ConcentricOctahedralToDirection(const Vector2&);
    MR_HF_DECL constexpr
    Vector2 ConcentricOctahedralWrap(const Vector2&);

    template<std::integral T>
    MR_HF_DECL constexpr
    Vector<2, T> ConcentricOctahedralWrapInt(const Vector<2, T>& st,
                                             const Vector<2, T>& dimensions);

    // Orthonormalization Gram-Schmidt process
    // Returns first (n-1) modified vectors.
    MR_HF_DECL constexpr
    Vector3                 GSOrthonormalize(const Vector3& x,
                                             const Vector3& y);
    MR_HF_DECL constexpr
    std::array<Vector3, 2>  GSOrthonormalize(const Vector3& x,
                                             const Vector3& y,
                                             const Vector3& z);

    MR_HF_DECL constexpr
    Vector2 UVToSphericalAngles(const Vector2& uv);
    MR_HF_DECL constexpr
    Vector2 SphericalAnglesToUV(const Vector2& thetaPhi);

    template<class DimType>
    MR_HF_DECL constexpr
    DimType TextureMipSize(const DimType& resolution,
                                           uint32_t mipLevel);
    template<class DimType>
    MR_HF_DECL constexpr
    uint32_t TextureMipCount(const DimType& resolution);
    template<class DimType>
    MR_HF_DECL constexpr
    uint32_t TextureMipPixelStart(const DimType& baseResolution,
                                  uint32_t mipLevel);

    template<uint32_t C>
    MR_HF_DECL constexpr
    Vector<C, Float> ConvertPixelIndices(const Vector<C, Float>& inputIndex,
                                         const Vector<C, Float>& toResolution,
                                         const Vector<C, Float>& fromResolution);

    MR_PF_DECL
    Vector3 ZUpToNSpace(const Vector3& v, const Vector3& N) noexcept;
    MR_PF_DECL
    Vector3 NSpaceToZUp(const Vector3& v, const Vector3& N) noexcept;

    namespace MortonCode
    {
        template <class T>
        MR_PF_DECL T         MaxBits3D() noexcept;
        template <class T>
        MR_PF_DECL T         Compose3D(const Vector3ui&) noexcept;
        template <class T>
        MR_PF_DECL Vector3ui Decompose3D(T code) noexcept;
        //
        template <class T>
        MR_PF_DECL T         MaxBits2D() noexcept;
        template <class T>
        MR_PF_DECL T         Compose2D(const Vector2ui&) noexcept;
        template <class T>
        MR_PF_DECL Vector2ui Decompose2D(T code) noexcept;
    }
}

namespace Graphics
{

MR_HF_DEF constexpr
Vector3 Reflect(const Vector3& normal, const Vector3& v)
{
    // Convention of wi (v) and normal is as follows
    //        normal
    // wi(v)         wo(out)
    //     ^    ^    ^
    //      \   |   /
    //       \  |  /
    //        \ | /
    //   --------------- Boundary
    //
    Vector3 result = Float{2} * Math::Dot(v, normal) * normal - v;
    return result;
}

MR_HF_DEF constexpr
Optional<Vector3> Refract(const Vector3& normal,
                          const Vector3& v, Float etaFrom, Float etaTo)
{
    using namespace Math;
    // Convention of wi (v) and normal is as follows
    //          wo (out)
    //          ^
    //         /
    //        /    etaTo (IOR)
    //--------------- Boundary
    //     /  |    etaFrom (IOR)
    //    /   |
    //   /    |
    //  v     v
    //  wi  normal
    //
    // So wi should be aligned with the normal, it is caller's responsibility
    // to provide the appropriate normal. It returns wo again "outwards" from the
    // surface. Both "normal" and "wi" is assumed to be normalized.
    Float etaRatio = etaFrom / etaTo;
    Float cosIn = Dot(normal, v);
    Float sinInSqr = Max(Float{0}, Float{1} - cosIn * cosIn);
    // Snell's Law
    Float sinOutSqr = etaRatio * etaRatio * sinInSqr;
    Float cosOut = SqrtMax(Float{1} - sinOutSqr);

    // Check total internal reflection
    if(sinOutSqr >= Float{1}) return std::nullopt;

    return (etaRatio * (-v) + (etaRatio * cosIn - cosOut) * normal);
}

MR_HF_DEF constexpr
Vector3 Orient(const Vector3& v, const Vector3& n)
{
    return Math::Dot(v, n) >= Float{0} ? v : (-v);
}

MR_HF_DEF constexpr
Vector3 OrthogonalVector(const Vector3& v)
{
    // PBRT Book
    // https://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Vectors#CoordinateSystem
    using Math::Abs;
    using Math::Sqrt;
    if(Abs(v[0]) > Abs(v[1]))
        return Vector3(-v[2], 0, v[0]) / Sqrt(v[0] * v[0] + v[2] * v[2]);
    else
        return Vector3(0, v[2], -v[1]) / Sqrt(v[1] * v[1] + v[2] * v[2]);
}

MR_HF_DEF constexpr
Vector3 SphericalToCartesian(const Vector3& sphrRTP)
{
    using Math::SinCos;
    const auto& [r, theta, phi] = sphrRTP.AsArray();
    const auto& [sT, cT] = Math::SinCos(theta);
    const auto& [sP, cP] = Math::SinCos(phi);
    Float x = r * cT * sP;
    Float y = r * sT * sP;
    Float z = r * cP;
    return Vector3(x, y, z);
}

MR_HF_DEF constexpr
Vector3 CartesianToSpherical(const Vector3& xyz)
{
    // Convert to Spherical Coordinates
    Vector3 norm = Math::Normalize(xyz);
    Float r = Math::Length(xyz);
    // range [-pi, pi]
    Float azimuth = Math::ArcTan2(norm[1], norm[0]);
    // range [0, pi]
    // Dot product between ZAxis and normalized vector
    Float incl = Math::ArcCos(norm[2]);
    return Vector3(r, azimuth, incl);
}

// Unit version
MR_HF_DEF constexpr
Vector3 UnitSphericalToCartesian(const Vector2& sphrTP)
{
    const auto& [theta, phi] = sphrTP.AsArray();
    const auto& [sT, cT] = Math::SinCos(theta);
    const auto& [sP, cP] = Math::SinCos(phi);
    Float x = cT * sP;
    Float y = sT * sP;
    Float z = cP;
    return Vector3(x, y, z);
}

MR_HF_DEF constexpr
Vector3 UnitSphericalToCartesian(const Vector2& sinCosTheta,
                                 const Vector2& sinCosPhi)
{
    Float x = sinCosTheta[1] * sinCosPhi[0];
    Float y = sinCosTheta[0] * sinCosPhi[0];
    Float z = sinCosPhi[1];
    return Vector3(x, y, z);
}

MR_HF_DEF constexpr
Vector2 CartesianToUnitSpherical(const Vector3& xyz)
{
    // Convert to Spherical Coordinates
    // range [-pi, pi]
    Float azimuth = Math::ArcTan2(xyz[1], xyz[0]);
    // range [0, pi]
    // Sometimes normalized cartesian coords may invoke NaN here
    // clamp it to the range
    Float incl = Math::ArcCos(Math::Clamp<Float>(xyz[2], -1, 1));
    return Vector2(azimuth, incl);
}

MR_HF_DEF constexpr
Vector2 PolarToCartesian(const Vector2& polarRT)
{
    const auto& [r, theta] = polarRT.AsArray();
    const auto& [s, c] = Math::SinCos(theta);
    Float x = r * s;
    Float y = r * c;
    return Vector2(x, y);
}

MR_HF_DEF constexpr
Vector2 CartesianToPolar(const Vector2& xy)
{
    Float r = Math::Length(xy);
    Float theta = Math::ArcTan2(xy[1], xy[0]);
    return Vector2(r, theta);
}

MR_HF_DEF constexpr
Vector2 UnitPolarToCartesian(Float theta)
{
    const auto& [x, y] = Math::SinCos(theta);
    return Vector2(x, y);
}

MR_HF_DEF constexpr
Float CartesianToUnitPolar(const Vector2& xy)
{
    return Math::ArcTan2(xy[1], xy[0]);
}

MR_HF_DEF constexpr
Vector2 DirectionToConcentricOctahedral(const Vector3& dir)
{
    using namespace MathConstants;
    constexpr Float TwoOvrPi = InvPi<Float>() * Float{2};

    // Edge case
    if(dir[0] == 0 && dir[1] == 0) return Vector2(0);

    Float xAbs = Math::Abs(dir[0]);
    Float yAbs = Math::Abs(dir[1]);
    Float atanIn = yAbs / xAbs;
    Float phiPrime = Math::ArcTan(atanIn);

    Float radius = Math::SqrtMax(Float{1} - Math::Abs(dir[2]));

    Float v = radius * TwoOvrPi * phiPrime;
    Float u = radius - v;
    // Now convert to the quadrant
    if(dir[2] < Float(0))
    {
        Float uPrime = Float{1} - v;
        Float vPrime = Float{1} - u;
        u = uPrime;
        v = vPrime;
    }
    // Sign extend the uv
    u *= Math::SignPM1(dir[0]);
    v *= Math::SignPM1(dir[1]);

    // Finally
    // [-1,1] to [0,1]
    Vector2 st = Vector2(u, v);
    st = (st + Vector2(1)) * Float{0.5};
    return st;
}

MR_HF_DEF constexpr
Vector3 ConcentricOctahedralToDirection(const Vector2& st)
{
    using namespace MathConstants;
    constexpr Float PiOvr4 = Pi<Float>() * Float{0.25};
    // [0,1] to [-1,1]
    Vector2 uv = st * Float(2) - Float(1);
    Vector2 uvAbs = Math::Abs(uv);

    // Radius
    Float d = Float(1) - uvAbs.Sum();
    Float radius = Float(1) - Math::Abs(d);
    Float phiPrime = Float(0);
    // Avoid division by zero
    if(radius != Float(0)) phiPrime = ((uvAbs[1] - uvAbs[0]) / radius + 1) * PiOvr4;
    // Coords
    const auto& [sinPhiP, cosPhiP] = Math::SinCos(phiPrime);
    Float cosPhi = Math::SignPM1(uv[0]) * cosPhiP;
    Float sinPhi = Math::SignPM1(uv[1]) * sinPhiP;
    Float z = Math::SignPM1(d) * (Float(1) - radius * radius);

    // Now all is OK do the concentric disk stuff
    Float xyFactor = radius * Math::Sqrt(Float(2) - radius * radius);
    Float x = cosPhi * xyFactor;
    Float y = sinPhi * xyFactor;

    return Vector3(x, y, z);
}

MR_HF_DEF constexpr
Vector2 ConcentricOctahedralWrap(const Vector2& st)
{
    // Given st => (-inf, inf) convert to [0, 1]
    // Octahedral Concentric mapping has straightforward properties
    // if either s or t is odd (integral part) we mirror the st on both sides
    // If both is odd or even do not mirror
    // Convert the negative numbers
    Vector2f stConv = st;
    if(st[0] < 0) stConv[0] = -2 - st[0];
    if(st[1] < 0) stConv[1] = -2 - st[1];
    auto [iS, fS] = Math::ModFInt(stConv[0]);
    auto [iT, fT] = Math::ModFInt(stConv[1]);
    fS = Math::Abs(fS); fT = Math::Abs(fT);
    bool doMirror = static_cast<bool>((iS & 0x1) ^ (iT & 0x1));
    if(doMirror)
    {
        fS = 1 - fS;
        fT = 1 - fT;
    }
    return Vector2(fS, fT);
}

template<std::integral T>
MR_HF_DEF constexpr
Vector<2, T>  ConcentricOctahedralWrapInt(const Vector<2, T>& st,
                                          const Vector<2, T>& dimensions)
{
    Vector<2, T> stConv = st;
    if constexpr(std::is_signed_v<T>)
    {
        if(st[0] < 0) stConv[0] = -2 * dimensions[0] - st[0];
        if(st[1] < 0) stConv[1] = -2 * dimensions[1] - st[1];
    }

    Vector<2, T> dimClamp = dimensions - 1;
    Vector<2, T> intPart = stConv / dimensions;
    Vector<2, T> fracPart = Math::Abs(stConv % dimensions);

    T xOdd = (intPart[0] & 0x1);
    T yOdd = (intPart[1] & 0x1);
    bool doMirror = static_cast<bool>(xOdd ^ yOdd);
    if(doMirror) fracPart = dimClamp - fracPart;
    return fracPart;
}

MR_HF_DEF constexpr
Vector3 GSOrthonormalize(const Vector3& x, const Vector3& y)
{
    return Math::Normalize(x - y * Math::Dot(x, y));
}

MR_HF_DEF constexpr
std::array<Vector3, 2> GSOrthonormalize(const Vector3& x,
                                        const Vector3& y,
                                        const Vector3& z)
{
    Vector3 rY = GSOrthonormalize(y, z);
    Vector3 rX = GSOrthonormalize(x, rY);
    return {rX, rY};
}

MR_HF_DEF constexpr
Vector2 UVToSphericalAngles(const Vector2& uv)
{
    using namespace MathConstants;
    return Vector2(// [-pi, pi]
                   (uv[0] * Pi<Float>() * 2) - Pi<Float>(),
                   // [0, pi]
                   (1 - uv[1]) * Pi<Float>());
}

MR_HF_DEF constexpr
Vector2 SphericalAnglesToUV(const Vector2& thetaPhi)
{
    using namespace MathConstants;
    // Theta range [-pi, pi)
    assert(thetaPhi[0] >= -Pi<Float>() &&
           thetaPhi[0] <= Pi<Float>());
    // phi range [0, pi]
    assert(thetaPhi[1] >= 0 &&
           thetaPhi[1] <= Pi<Float>());
    // Normalize to generate UV [0, 1]
    Float u = (thetaPhi[0] + Pi<Float>()) * Float(0.5) / Pi<Float>();
    Float v = Float(1) - (thetaPhi[1] * InvPi<Float>());
    return Vector2(u, v);
}

template<>
MR_HF_DEF constexpr
uint32_t TextureMipSize<uint32_t>(const uint32_t& resolution, uint32_t mipLevel)
{
    return Math::Max(resolution >> mipLevel, 1u);
}

template<class DimType>
requires(std::is_same_v<DimType, Vector2ui> || std::is_same_v<DimType, Vector3ui>)
MR_HF_DEF constexpr
DimType TextureMipSize(const DimType& resolution, uint32_t mipLevel)
{
    DimType mipRes;
    MRAY_UNROLL_LOOP_N(DimType::Dims)
    for(uint32_t i = 0; i < DimType::Dims; i++)
        mipRes[i] = resolution[i] >> mipLevel;

    return Math::Max(mipRes, DimType(1));
}

template<>
MR_HF_DEF constexpr
uint32_t TextureMipCount<uint32_t>(const uint32_t& resolution)
{
    return Bit::RequiredBitsToRepresent(resolution);
}

template<class DimType>
requires(std::is_same_v<DimType, Vector2ui> || std::is_same_v<DimType, Vector3ui>)
MR_HF_DEF constexpr
uint32_t TextureMipCount(const DimType& resolution)
{
    uint32_t maxDim = resolution[resolution.Maximum()];
    return Bit::RequiredBitsToRepresent(maxDim);
}

template<class DimType>
requires(std::is_same_v<DimType, Vector2ui> ||
         std::is_same_v<DimType, Vector3ui> ||
         std::is_same_v<DimType, uint32_t>)
MR_HF_DEF constexpr
uint32_t TextureMipPixelStart(const DimType& baseResolution, uint32_t mipLevel)
{
    assert(TextureMipCount(baseResolution) > mipLevel);
    uint32_t mipPixelStart = 0;
    for(uint32_t i = 0; i < mipLevel; i++)
    {
        if constexpr(std::is_same_v<DimType, uint32_t>)
            mipPixelStart += TextureMipSize(baseResolution, i);
        else
            mipPixelStart += TextureMipSize(baseResolution, i).Multiply();
    }
    return mipPixelStart;
}

template<uint32_t C>
MR_HF_DEF constexpr
Vector<C, Float> ConvertPixelIndices(const Vector<C, Float>& inputIndex,
                                     const Vector<C, Float>& toResolution,
                                     const Vector<C, Float>& fromResolution)
{
    Vector<C, Float> result;
    Vector<C, Float> uvRatio = toResolution / fromResolution;
    result = (inputIndex + Float(0.5)) * uvRatio - Float(0.5);
    result = Math::Clamp(result, Vector2::Zero(), toResolution - Vector2(1));
    return result;
}

MR_PF_DECL
Vector3 ZUpToNSpace(const Vector3& v, const Vector3& N) noexcept
{
    // https://jcgt.org/published/0006/01/01/
    Float s = Math::SignPM1(N[2]);
    Float c = Float(-1) / (s + N[2]);
    Float b = N[0] * N[1] * c;
    Vector3 r0 = Vector3(Float(1) + s * N[0] * N[0] * c, b, N[0]);
    Vector3 r1 = Vector3(s * b, s + N[1] * N[1] * c, N[1]);
    Vector3 r2 = Vector3(-s * N[0], -N[1], N[2]);
    Vector3 result = Vector3(Math::Dot(r0, v),
                             Math::Dot(r1, v),
                             Math::Dot(r2, v));
    return result;
};

MR_PF_DECL
Vector3 NSpaceToZUp(const Vector3& v, const Vector3& N) noexcept
{
    // https://jcgt.org/published/0006/01/01/
    Float s = Math::SignPM1(N[2]);
    Float c = Float(-1) / (s + N[2]);
    Float b = N[0] * N[1] * c;
    Float r0X = Float(1) + s * N[0] * N[0] * c;
    Vector3 r0 = Vector3(r0X, s * b, -s * N[0]);
    Vector3 r1 = Vector3(b, s + N[1] * N[1] * c, -N[1]);
    Vector3 r2 = N;
    Vector3 result = Vector3(Math::Dot(r0, v),
                             Math::Dot(r1, v),
                             Math::Dot(r2, v));
    return result;
};

template <>
MR_PF_DEF uint32_t MortonCode::MaxBits3D() noexcept
{
    return 10u;
}

template <>
MR_PF_DEF uint64_t MortonCode::MaxBits3D() noexcept
{
    return 21u;
}

template <>
MR_PF_DEF uint32_t MortonCode::Compose3D(const Vector3ui& val) noexcept
{
    auto Expand3D = [](uint32_t x) -> uint32_t
    {
        // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
        x &= 0x000003ff;
        x = (x ^ (x << 16)) & 0xff0000ff;
        x = (x ^ (x << 8) ) & 0x0300f00f;
        x = (x ^ (x << 4) ) & 0x030c30c3;
        x = (x ^ (x << 2) ) & 0x09249249;
        return x;
    };

    uint32_t x = Expand3D(val[0]);
    uint32_t y = Expand3D(val[1]);
    uint32_t z = Expand3D(val[2]);
    return ((x << 0) | (y << 1) | (z << 2));
}

template <>
MR_PF_DEF uint64_t MortonCode::Compose3D(const Vector3ui& val) noexcept
{
    auto Expand3D = [](uint32_t v) -> uint64_t
    {
        // https://stackoverflow.com/questions/18529057/produce-interleaving-bit-patterns-morton-keys-for-32-bit-64-bit-and-128bit
        uint64_t x = v;
        x &= 0x1fffff;
        x = (x | x << 32) & 0x001f00000000ffff;
        x = (x | x << 16) & 0x001f0000ff0000ff;
        x = (x | x << 8 ) & 0x100f00f00f00f00f;
        x = (x | x << 4 ) & 0x10c30c30c30c30c3;
        x = (x | x << 2 ) & 0x1249249249249249;
        return x;
    };

    uint64_t x = Expand3D(val[0]);
    uint64_t y = Expand3D(val[1]);
    uint64_t z = Expand3D(val[2]);
    return ((x << 0) | (y << 1) | (z << 2));
}

template <>
MR_PF_DEF Vector3ui MortonCode::Decompose3D(uint32_t code) noexcept
{
    auto Shrink3D = [](uint32_t x) -> uint32_t
    {
        x &= 0x09249249;
        x = (x ^ (x >> 2) ) & 0x030c30c3;
        x = (x ^ (x >> 4) ) & 0x0300f00f;
        x = (x ^ (x >> 8) ) & 0xff0000ff;
        x = (x ^ (x >> 16)) & 0x000003ff;
        return x;
    };

    uint32_t x = Shrink3D(code >> 0);
    uint32_t y = Shrink3D(code >> 1);
    uint32_t z = Shrink3D(code >> 2);
    return Vector3ui(x, y, z);
}

template <>
MR_PF_DEF Vector3ui MortonCode::Decompose3D(uint64_t code) noexcept
{
    auto Shrink3D = [](uint64_t x) -> uint32_t
    {
        x &= 0x1249249249249249;
        x = (x ^ (x >> 2) ) & 0x30c30c30c30c30c3;
        x = (x ^ (x >> 4) ) & 0xf00f00f00f00f00f;
        x = (x ^ (x >> 8) ) & 0x00ff0000ff0000ff;
        x = (x ^ (x >> 16)) & 0x00ff00000000ffff;
        x = (x ^ (x >> 32)) & 0x00000000001fffff;
        return static_cast<uint32_t>(x);
    };

    uint32_t x = Shrink3D(code >> 0);
    uint32_t y = Shrink3D(code >> 1);
    uint32_t z = Shrink3D(code >> 2);
    return Vector3ui(x, y, z);
}

template <>
MR_PF_DEF uint32_t MortonCode::MaxBits2D() noexcept
{
    return 16u;
}

template <>
MR_PF_DEF uint64_t MortonCode::MaxBits2D() noexcept
{
    return 32u;
}

template <>
MR_PF_DEF uint32_t MortonCode::Compose2D(const Vector2ui& val) noexcept
{
    auto Expand2D = [](uint32_t val) -> uint32_t
    {
        // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
        uint32_t x = val;
        x &= 0x0000ffff;
        x = (x ^ (x << 8)) & 0x00ff00ff;
        x = (x ^ (x << 4)) & 0x0f0f0f0f;
        x = (x ^ (x << 2)) & 0x33333333;
        x = (x ^ (x << 1)) & 0x55555555;
        return x;
    };
    uint32_t x = Expand2D(val[0]);
    uint32_t y = Expand2D(val[1]);
    return ((x << 0) | (y << 1));
}

template <>
MR_PF_DEF uint64_t MortonCode::Compose2D(const Vector2ui& val) noexcept
{
    auto Expand2D = [](uint32_t val) -> uint64_t
    {
        // https://stackoverflow.com/questions/30539347/2d-morton-code-encode-decode-64bits
        uint64_t x = val;
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
        x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
        x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
        x = (x | (x << 2)) & 0x3333333333333333;
        x = (x | (x << 1)) & 0x5555555555555555;
        return x;
    };
    uint64_t x = Expand2D(val[0]);
    uint64_t y = Expand2D(val[1]);
    return ((x << 0) | (y << 1));

}

template <>
MR_PF_DEF Vector2ui MortonCode::Decompose2D(uint32_t code) noexcept
{
    auto Shrink2D = [](uint32_t x)
    {
        x &= 0x55555555;
        x = (x ^ (x >> 1)) & 0x33333333;
        x = (x ^ (x >> 2)) & 0x0f0f0f0f;
        x = (x ^ (x >> 4)) & 0x00ff00ff;
        x = (x ^ (x >> 8)) & 0x0000ffff;
        return x;
    };
    uint32_t x = Shrink2D(code >> 0);
    uint32_t y = Shrink2D(code >> 1);
    return Vector2ui(x, y);
}

template <>
MR_PF_DEF Vector2ui MortonCode::Decompose2D(uint64_t code) noexcept
{
    auto Shrink2D = [](uint64_t x)
    {
        // https://stackoverflow.com/questions/30539347/2d-morton-code-encode-decode-64bits
        x = x & 0x5555555555555555;
        x = (x | (x >> 1)) & 0x3333333333333333;
        x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0F;
        x = (x | (x >> 4)) & 0x00FF00FF00FF00FF;
        x = (x | (x >> 8)) & 0x0000FFFF0000FFFF;
        x = (x | (x >> 16)) & 0x00000000FFFFFFFF;
        return static_cast<uint32_t>(x);
    };
    uint32_t x = Shrink2D(code >> 0);
    uint32_t y = Shrink2D(code >> 1);
    return Vector2ui(x, y);
}

}