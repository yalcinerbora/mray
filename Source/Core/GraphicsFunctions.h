#pragma once

#include "Vector.h"
#include "Types.h"
#include "Quaternion.h"

namespace GraphicsFunctions
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
    MRAY_HYBRID
    constexpr Vector3   Reflect(const Vector3& normal, const Vector3& v);

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
    MRAY_HYBRID
    Optional<Vector3>   Refract(const Vector3& normal, const Vector3& v,
                                Float etaFrom, Float etaTo);

    // Changes the direction of vector "v" towards n
    MRAY_HYBRID
    constexpr Vector3   Orient(const Vector3& v, const Vector3& n);

    // Simple Sampling Functions
    // Sample cosine weighted direction from unit hemisphere
    // Unit hemisphere's normal is implicitly +Z
    MRAY_HYBRID SampleT<Vector3>    SampleCosDirection(const Vector2& xi);
    MRAY_HYBRID constexpr Float     PDFCosDirection(const Vector3& v,
                                                    const Vector3& n = Vector3::ZAxis());
    // Sample uniform direction from unit hemisphere
    // Unit hemisphere's normal is implicitly +Z
    MRAY_HYBRID SampleT<Vector3>    SampleUniformDirection(const Vector2& xi);
    MRAY_HYBRID constexpr Float     PDFUniformDirection();

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
    MRAY_HYBRID Vector3     SphericalToCartesian(const Vector3&);
    MRAY_HYBRID Vector3     CartesianToSpherical(const Vector3&);
    MRAY_HYBRID Vector3     UnitSphericalToCartesian(const Vector2&);
    MRAY_HYBRID Vector3     UnitSphericalToCartesian(const Vector2& sinCosTheta,
                                                 const Vector2& sinCosPhi);
    MRAY_HYBRID Vector2     CartesianToUnitSpherical(const Vector3&);

    //
    // Cocentric Octohedral Mapping
    // https://fileadmin.cs.lth.se/graphics/research/papers/2008/simdmapping/clarberg_simdmapping08_preprint.pdf
    MRAY_HYBRID Vector2     DirectionToCocentricOctohedral(const Vector3&);
    MRAY_HYBRID Vector3     CocentricOctohedralToDirection(const Vector2&);
    MRAY_HYBRID Vector2     CocentricOctohedralWrap(const Vector2&);

    template<std::integral T>
    MRAY_HYBRID
    constexpr Vector<2, T>  CocentricOctohedralWrapInt(const Vector<2, T>& st,
                                                       const Vector<2, T>& dimensions);

    // Orthonormalization Gram-Schmidt process
    // Returns first (n-1) modified vectors.
    MRAY_HYBRID
    constexpr Vector3                   GSOrthonormalize(const Vector3& x,
                                                         const Vector3& y);
    MRAY_HYBRID
    constexpr Pair<Vector3, Vector3>    GSOrthonormalize(const Vector3& x,
                                                        const Vector3& y,
                                                        const Vector3& z);

    MRAY_HYBRID constexpr Vector2       UVToSphericalAngles(const Vector2& uv);
    MRAY_HYBRID constexpr Vector2       SphericalAnglesToUV(const Vector2& thetaPhi);

    namespace MortonCode
    {
        template <class T>
        MRAY_HYBRID
        constexpr T         Compose3D(const Vector3ui&);
        template <class T>
        MRAY_HYBRID
        constexpr Vector3ui Decompose3D(T code);
        template <class T>
        MRAY_HYBRID
        constexpr T         Compose2D(const Vector2ui&);
        template <class T>
        MRAY_HYBRID
        constexpr Vector2ui Decompose2D(T code);
    }
}

namespace GraphicsFunctions
{

MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector3 Reflect(const Vector3& normal, const Vector3& v)
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
    Vector3 result = Float{2} * v.Dot(normal) * normal - v;
    return result;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Optional<Vector3> Refract(const Vector3& normal,
                          const Vector3& v, Float etaFrom, Float etaTo)
{
    using MathFunctions::SqrtMax;
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
    Float cosIn = normal.Dot(v);
    Float sinInSqr = std::max(Float{0}, Float{1} - cosIn * cosIn);
    // Snell's Law
    Float sinOutSqr = etaRatio * etaRatio * sinInSqr;
    Float cosOut = SqrtMax(Float{1} - sinOutSqr);

    // Check total internal reflection
    if(sinOutSqr >= Float{1}) return std::nullopt;

    return (etaRatio * (-v) + (etaRatio * cosIn - cosOut) * normal);
}

MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector3 Orient(const Vector3& v, const Vector3& n)
{
    return v.Dot(n) >= Float{0} ? v : (-v);
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> SampleCosDirection(const Vector2& xi)
{
    using namespace MathConstants;
    using MathFunctions::SqrtMax;

    // Generated direction is on unit space (+Z oriented hemisphere)
    Float xi1Angle = Float{2} * Pi<Float>() * xi[1];
    Float xi0Sqrt = std::sqrt(xi[0]);

    Vector3 dir;
    dir[0] = xi0Sqrt * std::cos(xi1Angle);
    dir[1] = xi0Sqrt * std::sin(xi1Angle);
    dir[2] = SqrtMax(Float{1} - Vector2(dir).Dot(Vector2(dir)));

    // Fast tangent space dot product and domain constant
    Float pdf = dir[2] * InvPi<Float>();

    // Finally the result!
    return SampleT<Vector3>
    {
        .sampledResult = dir,
        .pdf = pdf
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Float PDFCosDirection(const Vector3& v, const Vector3& n)
{
    Float pdf = n.Dot(v) * MathConstants::InvPi<Float>();
    return pdf;
}

MRAY_HYBRID MRAY_CGPU_INLINE
SampleT<Vector3> SampleUniformDirection(const Vector2& xi)
{
    using namespace MathConstants;
    using MathFunctions::SqrtMax;

    Float xi0Sqrt = SqrtMax(Float{1} - xi[0] * xi[0]);
    Float xi1Angle = 2 * Pi<Float>() * xi[1];

    Vector3 dir;
    dir[0] = xi0Sqrt * std::cos(xi1Angle);
    dir[1] = xi0Sqrt * std::sin(xi1Angle);
    dir[2] = xi[0];

    // Uniform pdf is invariant
    constexpr Float pdf = InvPi<Float>() * Float{0.5};
    return SampleT<Vector3>
    {
        .sampledResult = dir,
        .pdf = pdf
    };
}

MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Float PDFUniformDirection()
{
    return MathConstants::InvPi<Float>() * Float{0.5};
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 SphericalToCartesian(const Vector3& sphrRTP)
{
    const auto& [r, theta, phi] = sphrRTP.AsArray();
    Float x = r * std::cos(theta) * std::sin(phi);
    Float y = r * std::sin(theta) * std::sin(phi);
    Float z = r * std::cos(phi);
    return Vector3(x, y, z);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 CartesianToSpherical(const Vector3& xyz)
{
    // Convert to Spherical Coordinates
    Vector3 norm = xyz.Normalize();
    Float r = xyz.Length();
    // range [-pi, pi]
    Float azimuth = std::atan2(norm[1], norm[0]);
    // range [0, pi]
    // Dot product between ZAxis and normalized vector
    Float incl = std::acos(norm[2]);
    return Vector3(r, azimuth, incl);
}

// Unit version
MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 UnitSphericalToCartesian(const Vector2& sphrTP)
{
    const auto& [theta, phi] = sphrTP.AsArray();
    Float x = std::cos(theta) * std::sin(phi);
    Float y = std::sin(theta) * std::sin(phi);
    Float z = std::cos(phi);
    return Vector3(x, y, z);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 UnitSphericalToCartesian(const Vector2& sinCosTheta,
                                 const Vector2& sinCosPhi)
{
    Float x = sinCosTheta[1] * sinCosPhi[0];
    Float y = sinCosTheta[0] * sinCosPhi[0];
    Float z = sinCosPhi[1];
    return Vector3(x, y, z);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector2 CartesianToUnitSpherical(const Vector3& xyz)
{
    // Convert to Spherical Coordinates
    // range [-pi, pi]
    Float azimuth = std::atan2(xyz[1], xyz[0]);
    // range [0, pi]
    // Sometimes normalized cartesian coords may invoke NaN here
    // clamp it to the range
    Float incl = std::acos(MathFunctions::Clamp<Float>(xyz[2], -1, 1));
    return Vector2(azimuth, incl);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector2 DirectionToCocentricOctohedral(const Vector3& dir)
{
    using namespace MathConstants;
    constexpr Float TwoOvrPi = InvPi<Float>() * Float{2};

    // Edge case
    if(dir[0] == 0 && dir[1] == 0) return Vector2(0);

    Float xAbs = std::abs(dir[0]);
    Float yAbs = std::abs(dir[1]);
    Float atanIn = yAbs / xAbs;
    Float phiPrime = std::atan(atanIn);

    Float radius = std::sqrt(Float{1} - std::abs(dir[2]));

    Float v = radius * TwoOvrPi * phiPrime;
    Float u = radius - v;
    // Now convert to the quadrant
    if(dir[2] < 0)
    {
        Float uPrime = Float{1} - v;
        Float vPrime = Float{1} - u;
        u = uPrime;
        v = vPrime;
    }
    // Sign extend the uv
    u *= (std::signbit(dir[0]) ? Float{-1} : Float{1});
    v *= (std::signbit(dir[1]) ? Float{-1} : Float{1});

    // Finally
    // [-1,1] to [0,1]
    Vector2 st = Vector2(u, v);
    st = (st + Vector2(1)) * Float{0.5};
    return st;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 CocentricOctohedralToDirection(const Vector2& st)
{
    using namespace MathConstants;
    constexpr Float PiOvr4 = Pi<Float>() * Float{0.25};

    // Clang signbit definition is only on std namespace
    // this is a crappy workaround
    #ifndef __CUDA_ARCH__
        using namespace std;
    #endif

    // [0,1] to [-1,1]
    Vector2 uv = st * 2 - 1;
    Vector2 uvAbs = uv.Abs();

    // Radius
    Float d = 1 - uvAbs.Sum();
    Float radius = 1 - abs(d);
    Float phiPrime = 0;
    // Avoid division by zero
    if(radius != 0) phiPrime = ((uvAbs[1] - uvAbs[0]) / radius + 1) * PiOvr4;
    // Coords
    Float cosPhi = (std::signbit(uv[0]) ? -1 : 1) * std::cos(phiPrime);
    Float sinPhi = (std::signbit(uv[1]) ? -1 : 1) * std::sin(phiPrime);
    Float z = (signbit(d) ? -1 : 1) * (1 - radius * radius);

    // Now all is OK do the cocentric disk stuff
    Float xyFactor = radius * std::sqrt(2 - radius * radius);
    Float x = cosPhi * xyFactor;
    Float y = sinPhi * xyFactor;

    return Vector3(x, y, z);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector2 CocentricOctohedralWrap(const Vector2& st)
{
    // Get a good signed int candidate according to the "Float" type
    using IntType = typename std::conditional_t<std::is_same_v<Float, double>, int64_t, int32_t>;

    // Given st => (-inf, inf) convert to [0, 1]
    // Octohedral Cocentric mapping has straightforward properties
    // if either s or t is odd (integral part) we mirror the st on both sides
    // If both is odd or even do not mirror
    // Convert the negative numbers
    Vector2f stConv = st;
    if(st[0] < 0) stConv[0] = -2 - st[0];
    if(st[1] < 0) stConv[1] = -2 - st[1];

    Float iS; Float fS = std::abs(std::modf(stConv[0], &iS));
    Float iT; Float fT = std::abs(std::modf(stConv[1], &iT));
    IntType iSInt = static_cast<IntType>(iS);
    IntType iTInt = static_cast<IntType>(iT);
    bool doMirror = static_cast<bool>((iSInt & 0x1) ^ (iTInt & 0x1));
    if(doMirror)
    {
        fS = 1 - fS;
        fT = 1 - fT;
    }
    return Vector2(fS, fT);
}

template<std::integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<2, T>  CocentricOctohedralWrapInt(const Vector<2, T>& st,
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
    Vector<2, T> fracPart = (stConv % dimensions).Abs();

    T xOdd = (intPart[0] & 0x1);
    T yOdd = (intPart[1] & 0x1);
    bool doMirror = static_cast<bool>(xOdd ^ yOdd);
    if(doMirror) fracPart = dimClamp - fracPart;
    return fracPart;
}

MRAY_HYBRID
constexpr Vector3 GSOrthonormalize(const Vector3& x, const Vector3& y)
{
    return (x - y * x.Dot(y)).Normalize();
}

MRAY_HYBRID
constexpr Pair<Vector3, Vector3> GSOrthonormalize(const Vector3& x,
                                                  const Vector3& y,
                                                  const Vector3& z)
{
    Vector3 rY = GSOrthonormalize(y, z);
    Vector3 rX = GSOrthonormalize(x, rY);
    return std::make_pair(rX, rY);
}

MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector2 UVToSphericalAngles(const Vector2& uv)
{
    using namespace MathConstants;
    return Vector2(// [-pi, pi]
                   (uv[0] * Pi<Float>() * 2) - Pi<Float>(),
                   // [0, pi]
                   (1 - uv[1]) * Pi<Float>());
}

MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector2 SphericalAnglesToUV(const Vector2& thetaPhi)
{
    using namespace MathConstants;
    // Theta range [-pi, pi)
    assert(thetaPhi[0] >= -Pi<Float>() &&
           thetaPhi[0] < Pi<Float>());
    // phi range [0, pi]
    assert(thetaPhi[1] >= 0 &&
           thetaPhi[1] <= Pi<Float>());

    // Normalize to generate UV [0, 1]
    Float u = (thetaPhi[0] + Pi<Float>()) * 0.5f / Pi<Float>();
    Float v = 1 - (thetaPhi[1] * InvPi<Float>());
    return Vector2(u, v);
}


namespace MortonCode
{
    template <>
    MRAY_HYBRID MRAY_CGPU_INLINE
    constexpr uint32_t Compose3D(const Vector3ui& val)
    {
        auto Expand3D = [](uint32_t x) -> uint32_t
        {
            // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
            x &= 0x000003ff;
            x = (x ^ (x << 16)) & 0xff0000ff;
            x = (x ^ (x << 8)) & 0x0300f00f;
            x = (x ^ (x << 4)) & 0x030c30c3;
            x = (x ^ (x << 2)) & 0x09249249;
            return x;
        };

        uint32_t x = Expand3D(val[0]);
        uint32_t y = Expand3D(val[1]);
        uint32_t z = Expand3D(val[2]);
        return ((x << 0) | (y << 1) | (z << 2));
    }

    template <>
    MRAY_HYBRID MRAY_CGPU_INLINE
    constexpr uint64_t Compose3D(const Vector3ui& val)
    {
        auto Expand3D = [](uint32_t v) -> uint64_t
        {
            // https://stackoverflow.com/questions/18529057/produce-interleaving-bit-patterns-morton-keys-for-32-bit-64-bit-and-128bit
            uint64_t x = v;
            x &= 0x1fffff;
            x = (x | x << 32) & 0x001f00000000ffff;
            x = (x | x << 16) & 0x001f0000ff0000ff;
            x = (x | x << 8) & 0x100f00f00f00f00f;
            x = (x | x << 4) & 0x10c30c30c30c30c3;
            x = (x | x << 2) & 0x1249249249249249;
            return x;
        };

        uint64_t x = Expand3D(val[0]);
        uint64_t y = Expand3D(val[1]);
        uint64_t z = Expand3D(val[2]);
        return ((x << 0) | (y << 1) | (z << 2));
    }

    template <>
    MRAY_HYBRID MRAY_CGPU_INLINE
    constexpr Vector3ui Decompose3D(uint32_t code)
    {
        auto Shrink3D = [](uint32_t x) -> uint32_t
        {
            x &= 0x09249249;
            x = (x ^ (x >> 2)) & 0x030c30c3;
            x = (x ^ (x >> 4)) & 0x0300f00f;
            x = (x ^ (x >> 8)) & 0xff0000ff;
            x = (x ^ (x >> 16)) & 0x000003ff;
            return x;
        };

        uint32_t x = Shrink3D(code >> 0);
        uint32_t y = Shrink3D(code >> 1);
        uint32_t z = Shrink3D(code >> 2);
        return Vector3ui(x, y, z);
    }

    template <>
    MRAY_HYBRID MRAY_CGPU_INLINE
    constexpr Vector3ui Decompose3D(uint64_t code)
    {
        auto Shrink3D = [](uint64_t x) -> uint32_t
        {
            x &= 0x1249249249249249;
            x = (x ^ (x >> 2)) & 0x30c30c30c30c30c3;
            x = (x ^ (x >> 4)) & 0xf00f00f00f00f00f;
            x = (x ^ (x >> 8)) & 0x00ff0000ff0000ff;
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
    MRAY_HYBRID MRAY_CGPU_INLINE
    constexpr uint32_t Compose2D(const Vector2ui& val)
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
    MRAY_HYBRID MRAY_CGPU_INLINE
    constexpr uint64_t Compose2D(const Vector2ui& val)
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
    MRAY_HYBRID MRAY_CGPU_INLINE
    constexpr Vector2ui Decompose2D(uint32_t code)
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
    MRAY_HYBRID MRAY_CGPU_INLINE
    constexpr Vector2ui Decompose2D(uint64_t code)
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

}