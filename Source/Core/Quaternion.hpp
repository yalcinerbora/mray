#pragma once

#include "Quaternion.h"

template<FloatC T>
MR_PF_DEF Quat<T>::Quat(T w, T x, T y, T z) noexcept
    : v(w, x, y, z)
{}

template<FloatC T>
MR_PF_DEF Quat<T>::Quat(const T* vv) noexcept
    : v(vv)
{}

template<FloatC T>
MR_PF_DEF Quat<T>::Quat(T angle, const Vector<3, T>& axis) noexcept
{
    angle *= T(0.5);
    const auto&[sinAngle, cosAngle] = Math::SinCos(angle);
    v[1] = axis[0] * sinAngle;
    v[2] = axis[1] * sinAngle;
    v[3] = axis[2] * sinAngle;
    v[0] = cosAngle;
}

template<FloatC T>
MR_PF_DEF Quat<T>::Quat(const Vector<4, T>& v) noexcept
    : v(v)
{}

template<FloatC T>
MR_PF_DEF Quat<T>::operator Vector<4, T>&() noexcept
{
    return v;
}

template<FloatC T>
MR_PF_DEF Quat<T>::operator const Vector<4, T>&() const noexcept
{
    return v;
}

template<FloatC T>
MR_PF_DEF Quat<T>::operator T*() noexcept
{
    return static_cast<T*>(v);
}

template<FloatC T>
MR_PF_DEF Quat<T>::operator const T*() const noexcept
{
    return static_cast<const T*>(v);
}

template<FloatC T>
MR_PF_DEF T& Quat<T>::operator[](unsigned int i) noexcept
{
    return v[i];
}

template<FloatC T>
MR_PF_DEF const T& Quat<T>::operator[](unsigned int i) const noexcept
{
    return v[i];
}

template<FloatC T>
MR_PF_DEF Quat<T> Quat<T>::operator*(const Quat& right) const noexcept
{
    return Quat(v[0] * right[0] - v[1] * right[1] - v[2] * right[2] - v[3] * right[3],  // W
                v[0] * right[1] + v[1] * right[0] + v[2] * right[3] - v[3] * right[2],  // X
                v[0] * right[2] - v[1] * right[3] + v[2] * right[0] + v[3] * right[1],  // Y
                v[0] * right[3] + v[1] * right[2] - v[2] * right[1] + v[3] * right[0]); // Z
}

template<FloatC T>
MR_PF_DEF Quat<T> Quat<T>::operator*(T right) const noexcept
{
    return Quat<T>(v * right);
}

template<FloatC T>
MR_PF_DEF Quat<T> Quat<T>::operator+(const Quat& right) const noexcept
{
    return Quat(v + right.v);
}

template<FloatC T>
MR_PF_DEF Quat<T> Quat<T>::operator-(const Quat& right) const noexcept
{
    return Quat(v - right.v);
}

template<FloatC T>
MR_PF_DEF Quat<T> Quat<T>::operator-() const noexcept
{
    return Quat(-v);
}

template<FloatC T>
MR_PF_DEF Quat<T> Quat<T>::operator/(T right) const noexcept
{
    return Quat<T>(v / right);
}

template<FloatC T>
MR_PF_DEF void Quat<T>::operator*=(const Quat& right) noexcept
{
    Quat copy(*this);
    (*this) = copy * right;
}

template<FloatC T>
MR_PF_DEF void Quat<T>::operator*=(T right) noexcept
{
    v *= right;
}

template<FloatC T>
MR_PF_DEF void Quat<T>::operator+=(const Quat& right) noexcept
{
    v += right.v;
}

template<FloatC T>
MR_PF_DEF void Quat<T>::operator-=(const Quat& right) noexcept
{
    v -= right.v;
}

template<FloatC T>
MR_PF_DEF void Quat<T>::operator/=(T right) noexcept
{
    v /= right;
}

template<FloatC T>
MR_PF_DEF bool Quat<T>::operator==(const Quat& right) const noexcept
{
    return v == right.v;
}

template<FloatC T>
MR_PF_DEF bool Quat<T>::operator!=(const Quat& right) const noexcept
{
    return v != right.v;
}

template<FloatC T>
MR_PF_DEF Quat<T> Quat<T>::Normalize() const noexcept
{
    return Quat(Math::Normalize(v));
}

template<FloatC T>
MR_PF_DEF T Quat<T>::Length() const noexcept
{
    return Math::Length(v);
}

template<FloatC T>
MR_PF_DEF T Quat<T>::LengthSqr() const noexcept
{
    return Math::Length(v);
}

template<FloatC T>
MR_PF_DEF Quat<T> Quat<T>::Conjugate() const noexcept
{
    return Quat(v[0], -v[1], -v[2], -v[3]);
}

template<FloatC T>
MR_PF_DEF T Quat<T>::Dot(const Quat& right) const noexcept
{
    return Math::Dot(v, right.v);
}

template<FloatC T>
MR_PF_DEF Vector<3, T> Quat<T>::ApplyRotation(const Vector<3, T>& vtor) const noexcept
{
    // q * v * qInv
    Quat qInv = Conjugate();
    Quat vtorQ(0.0f, vtor[0], vtor[1], vtor[2]);

    Quat result((*this) * (vtorQ * qInv));
    return Vector<3, T>(result[1], result[2], result[3]);
}

template<FloatC T>
MR_PF_DEF Vector<3, T> Quat<T>::ApplyInvRotation(const Vector<3, T>& vtor) const noexcept
{
    Quat qInv = Conjugate();
    return qInv.ApplyRotation(vtor);
}

template<FloatC T>
MR_PF_DEF bool Quat<T>::HasNaN() const noexcept
{
    return v.HasNaN();
}

template<FloatC T>
MR_PF_DEF Vector<3, T> Quat<T>::OrthoBasisX() const noexcept
{
    Float v00 = v[0] * v[0];
    Float v02 = v[0] * v[2];
    Float v03 = v[0] * v[3];
    Float v11 = v[1] * v[1];
    Float v12 = v[1] * v[2];
    Float v13 = v[1] * v[3];
    Float v22 = v[2] * v[2];
    Float v33 = v[3] * v[3];
    Float X = (v00 + v11 - v22 - v33);
    Float Y = (v12 - v03 + v12 - v03);
    Float Z = (v13 + v02 + v13 + v02);
    return Vector<3, T>(X, Y, Z);
}

template<FloatC T>
MR_PF_DEF Vector<3, T> Quat<T>::OrthoBasisY() const noexcept
{
    Float v00 = v[0] * v[0];
    Float v01 = v[0] * v[1];
    Float v03 = v[0] * v[3];
    Float v11 = v[1] * v[1];
    Float v12 = v[1] * v[2];
    Float v22 = v[2] * v[2];
    Float v23 = v[2] * v[3];
    Float v33 = v[3] * v[3];
    Float X = (v12 + v03 + v12 + v03);
    Float Y = (v00 - v11 + v22 - v33);
    Float Z = (v23 - v01 + v23 - v01);
    return Vector<3, T>(X, Y, Z);

}

template<FloatC T>
MR_PF_DEF Vector<3, T> Quat<T>::OrthoBasisZ() const noexcept
{
    Float v00 = v[0] * v[0];
    Float v01 = v[0] * v[1];
    Float v02 = v[0] * v[2];
    Float v11 = v[1] * v[1];
    Float v13 = v[1] * v[3];
    Float v22 = v[2] * v[2];
    Float v23 = v[2] * v[3];
    Float v33 = v[3] * v[3];
    Float X = v13 - v02 + v13 - v02;
    Float Y = v23 + v01 + v23 + v01;
    Float Z = v00 - v11 - v22 + v33;
    return Vector<3, T>(X, Y, Z);
}

template<FloatC T>
MR_PF_DEF Quat<T> Quat<T>::NLerp(const Quat<T>& start,
                                 const Quat<T>& end, T t) noexcept
{
    T cosTheta = Math::Dot(start, end);
    // Select closest approach
    T cosFlipped = (cosTheta >= 0) ? cosTheta : (-cosTheta);

    T s0 = (1 - t);
    T s1 = t;
    // Flip scale if cos is flipped
    s1 = (cosTheta >= 0) ? s1 : (-s1);
    Quat<T> result = (start * s0) + (end * s1);
    return result;
}

template<FloatC T>
MR_PF_DEF Quat<T> Quat<T>::SLerp(const Quat<T>& start,
                                 const Quat<T>& end, T t) noexcept
{
    using namespace MathConstants;

    T cosTheta = Math::Dot(start.v, end.v);
    // Select closest approach
    T cosFlipped = (cosTheta >= 0) ? cosTheta : (-cosTheta);

    T s0, s1;
    if(cosFlipped < (1 - Epsilon<T>()))
    {
        T angle = Math::ArcCos(cosFlipped);
        T sinAngleRecip = T(1) / Math::Sin(angle);
        s0 = Math::Sin(angle * (1 - t)) * sinAngleRecip;
        s1 = Math::Sin(angle * t) * sinAngleRecip;
    }
    else
    {
        // Fallback to Lerp
        s0 = (1 - t);
        s1 = t;
    }
    // Flip scale if cos is flipped
    s1 = (cosTheta >= 0) ? s1 : (-s1);
    Quat<T> result = (start * s0) + (end * s1);
    return result;
}

template<FloatC T>
MR_PF_DEF Quat<T> Quat<T>::BarySLerp(const Quat<T>& q0,
                                     const Quat<T>& q1,
                                     const Quat<T>& q2,
                                     T a, T b) noexcept
{
    #ifndef MRAY_DEVICE_CODE_PATH
        using namespace std;
    #endif
    // Proper way to do this is
    // http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf
    //
    // But it is computationally complex.
    //
    // However vertex quaternions of the triangle will be closer or same.
    // instead we can directly slerp them.
    // (for smooth edges neighboring tri's face normal will be averaged)
    //
    // One thing to note is to check quaternions are close
    // and use conjugate in order to have proper average
    using namespace MathConstants;

    // Align towards q0
    const Quat<T>& qA = q0;
    const Quat<T>& qB = q1;
    const Quat<T>& qC = q2;

    T c = (1 - a - b);
    Quat<T> result;
    if(abs(a + b) < Epsilon<T>())
        result = qC;
    else
    {
        T ab = a / (a + b);
        Quat<T> qAB = Quat::SLerp(qB, qA, ab);
        result = Quat::SLerp(qAB, qC, c);
    }
    return result;
}

template<FloatC T>
MR_PF_DEF Quat<T> Quat<T>::RotationBetween(const Vector<3, T>& a, const Vector<3, T>& b) noexcept
{
    Vector<3, T> aCrossB = Math::Cross(a, b);
    T aDotB = Math::Dot(a, b);
    if(aCrossB != Vector<3, T>::Zero())
        aCrossB = Math::Normalize(aCrossB);
    return Quat<T>(Math::ArcCos(aDotB), aCrossB);
}

template<FloatC T>
MR_PF_DEF Quat<T> Quat<T>::RotationBetweenZAxis(const Vector<3, T>& b) noexcept
{
    using namespace MathConstants;
    Vector<3, T> zCrossD(-b[1], b[0], 0);
    T zDotD = b[2];

    // Half angle theorem
    T sin = Math::SqrtMax((T(1) - zDotD) * T(0.5));
    T cos = Math::SqrtMax((zDotD + T(1)) * T(0.5));

    zCrossD = Math::Normalize(zCrossD);
    T x = zCrossD[0] * sin;
    T y = zCrossD[1] * sin;
    T z = zCrossD[2] * sin;
    T w = cos;
    // Handle singularities
    if(Math::Abs(zDotD + T(1)) < LargeEpsilon<T>())
    {
        // Spaces are 180 degree apart
        // Define pi turn
        return Quat<T>(0, 0, 1, 0);
    }
    else if(Math::Abs(zDotD - T(1)) < LargeEpsilon<T>())
    {
        // Spaces are nearly equivalent
        // Just turn identity
        return Quat<T>(1, 0, 0, 0);
    }
    else return Quat<T>(w, x, y, z);
}

template<FloatC T>
MR_PF_DEF Quat<T> Quat<T>::Identity() noexcept
{
    return Quat<T>(1, 0, 0, 0);
}

template<FloatC T>
MR_PF_DEF Quat<T> operator*(T t, const Quat<T>& q) noexcept
{
    return q * t;
}

template<std::floating_point T>
MR_PF_DEF
Quat<T> TransformGen::ToSpaceQuat(const Vector<3, T>& xIn,
                                  const Vector<3, T>& y,
                                  const Vector<3, T>& z) noexcept
{
    Quat<T> q;
    // Flip the coordinate system if inverted
    Vector<3, T> x = xIn;
    if(Math::Abs(Math::Cross(x, y) - z) > Vector3(0.1))
    {
        x = -x;
    }

    // Converting a Rotation Matrix to a Quat
    // Mike Day, Insomniac Games (2015)
    // https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    T t;
    if(z[2] < 0)
    {
        if(x[0] > y[1])
        {
            t = 1 + x[0] - y[1] - z[2];
            q = Quat<T>(y[2] - z[1],
                        t,
                        x[1] + y[0],
                        z[0] + x[2]);
        }
        else
        {
            t = 1 - x[0] + y[1] - z[2];
            q = Quat<T>(z[0] - x[2],
                        x[1] + y[0],
                        t,
                        y[2] + z[1]);
        }
    }
    else
    {
        if(x[0] < -y[1])
        {
            t = 1 - x[0] - y[1] + z[2];
            q = Quat<T>(x[1] - y[0],
                        z[0] + x[2],
                        y[2] + z[1],
                        t);
        }
        else
        {
            t = 1 + x[0] + y[1] + z[2];
            q = Quat<T>(t,
                        y[2] - z[1],
                        z[0] - x[2],
                        x[1] - y[0]);
        }
    }
    q *= T{0.5} / Math::Sqrt(t);
    q = q.Normalize();
    q = q.Conjugate();
    return q;

    //// Another implementation that i found in stack overflow
    //// https://stackoverflow.com/questions/63734840/how-to-convert-rotation-matrix-to-quaternion
    //// Clang min definition is only on std namespace
    //// this is a crappy workaround
    //#ifndef MRAY_DEVICE_CODE_PATH
    //    using namespace std;
    //#endif
    //// Our sign is one (according to the above link)
    //static constexpr T sign = 1;
    //T t = x[0] + y[1] + z[2];
    //T m = max(max(x[0], y[1]), max(z[2], t));
    //T qmax = static_cast<T>(0.5) * Math::Sqrt(1 - t + 2 * m);
    //T denom = static_cast<T>(0.25) * (1 / qmax);
    //if(m == x[0])
    //{
    //    q[1] = qmax;
    //    q[2] = (x[1] + y[0]) * denom;
    //    q[3] = (x[2] + z[0]) * denom;
    //    q[0] = sign * (z[1] - y[2]) * denom;
    //}
    //else if(m == y[1])
    //{
    //    q[1] = (x[1] + y[0]) * denom;
    //    q[2] = qmax;
    //    q[3] = (y[2] + z[1]) * denom;
    //    q[0] = sign * (x[2] - z[0]) * denom;
    //}
    //else if(m == z[2])
    //{
    //    q[1] = (x[2] + z[0]) * denom;
    //    q[2] = (y[2] + z[1]) * denom;
    //    q[3] = qmax;
    //    q[0] = sign * (x[2] - z[0]) * denom;
    //}
    //else
    //{
    //    q[1] = sign * (z[1] - y[2]) * denom;
    //    q[2] = sign * (x[2] - z[0]) * denom;
    //    q[3] = sign * (y[0] - x[1]) * denom;
    //    q[0] = qmax;
    //}
    //q.NormalizeSelf();

}

template <std::floating_point T>
MR_PF_DEF
Quat<T> TransformGen::ToInvSpaceQuat(const Vector<3, T>& x,
                                     const Vector<3, T>& y,
                                     const Vector<3, T>& z) noexcept
{
    return Space(x, y, z).Conjugate();
}