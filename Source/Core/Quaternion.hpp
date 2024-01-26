#pragma once

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T>::Quat(T w, T x, T y, T z)
    : v(w, x, y, z)
{}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T>::Quat(const T* v)
    : v(v)
{}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T>::Quat(T angle, const Vector<3, T>& axis)
{
    angle *= 0.5;
    T sinAngle = sin(angle);

    v[1] = axis[0] * sinAngle;
    v[2] = axis[1] * sinAngle;
    v[3] = axis[2] * sinAngle;
    v[0] = cos(angle);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T>::Quat(const Vector<4, T>& v)
    : v(v)
{}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
Quat<T>::operator Vector<4, T>& ()
{
    return v;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
Quat<T>::operator const Vector<4, T>& () const
{
    return v;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
Quat<T>::operator T* ()
{
    return static_cast<T*>(v);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
Quat<T>::operator const T* () const
{
    return static_cast<const T*>(v);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T& Quat<T>::operator[](int i)
{
    return v[i];
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr const T& Quat<T>::operator[](int i) const
{
    return v[i];
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T> Quat<T>::operator*(const Quat& right) const
{
    return Quat(v[0] * right[0] - v[1] * right[1] - v[2] * right[2] - v[3] * right[3],  // W
                v[0] * right[1] + v[1] * right[0] + v[2] * right[3] - v[3] * right[2],  // X
                v[0] * right[2] - v[1] * right[3] + v[2] * right[0] + v[3] * right[1],  // Y
                v[0] * right[3] + v[1] * right[2] - v[2] * right[1] + v[3] * right[0]); // Z
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T> Quat<T>::operator*(T right) const
{
    return Quat<T>(v * right);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T> Quat<T>::operator+(const Quat& right) const
{
    return Quat(v + right.v);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T> Quat<T>::operator-(const Quat& right) const
{
    return Quat(v - right.v);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T> Quat<T>::operator-() const
{
    return Quat(-v);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T> Quat<T>::operator/(T right) const
{
    return Quat<T>(v / right);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Quat<T>::operator*=(const Quat& right)
{
    Quat copy(*this);
    (*this) = copy * right;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Quat<T>::operator*=(T right)
{
    v *= right;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Quat<T>::operator+=(const Quat& right)
{
    v += right.v;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Quat<T>::operator-=(const Quat& right)
{
    v -= right.v;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr void Quat<T>::operator/=(T right)
{
    v /= right;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Quat<T>::operator==(const Quat& right) const
{
    return v == right.v;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr bool Quat<T>::operator!=(const Quat& right) const
{
    return v != right.v;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T> Quat<T>::Normalize() const
{
    return Quat(v.Normalize());
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T>& Quat<T>::NormalizeSelf()
{
    v.NormalizeSelf();
    return *this;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Quat<T>::Length() const
{
    return v.Length();
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Quat<T>::LengthSqr() const
{
    return v.LengthSqr();
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T> Quat<T>::Conjugate() const
{
    return Quat(v[0], -v[1], -v[2], -v[3]);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T>& Quat<T>::ConjugateSelf()
{
    v[1] = -v[1];
    v[2] = -v[2];
    v[3] = -v[3];
    return *this;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr T Quat<T>::Dot(const Quat& right) const
{
    return v.Dot(right.v);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<3, T> Quat<T>::ApplyRotation(const Vector<3, T>& vtor) const
{
    // q * v * qInv
    // .Normalize();
    Quat qInv = Conjugate();
    Quat vtorQ(0.0f, vtor[0], vtor[1], vtor[2]);

    Quat result((*this) * (vtorQ * qInv));
    return Vector<3, T>(result[1], result[2], result[3]);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Vector<3, T> Quat<T>::ApplyInvRotation(const Vector<3, T>& vtor) const
{
    Quat qInv = Conjugate();
    return qInv.ApplyRotation(vtor);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T> Quat<T>::NLerp(const Quat<T>& start,
                                 const Quat<T>& end, T t)
{
    T cosTetha = start.Dot(end);
    // Select closest approach
    T cosFlipped = (cosTetha >= 0) ? cosTetha : (-cosTetha);

    T s0 = (1 - t);
    T s1 = t;
    // Flip scale if cos is flipped
    s1 = (cosTetha >= 0) ? s1 : (-s1);
    Quat<T> result = (start * s0) + (end * s1);
    return result;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr  Quat<T> Quat<T>::SLerp(const Quat<T>& start,
                                  const Quat<T>& end, T t)
{
    using namespace MathConstants;

    T cosTetha = start.Dot(end);
    // Select closest approach
    T cosFlipped = (cosTetha >= 0) ? cosTetha : (-cosTetha);

    T s0, s1;
    if(cosFlipped < (1 - Epsilon<T>()))
    {
        T angle = acos(cosFlipped);
        s0 = sin(angle * (1 - t)) / sin(angle);
        s1 = sin(angle * t) / sin(angle);
    }
    else
    {
        // Fallback to Lerp
        s0 = (1 - t);
        s1 = t;
    }
    // Flip scale if cos is flipped
    s1 = (cosTetha >= 0) ? s1 : (-s1);
    Quat<T> result = (start * s0) + (end * s1);
    return result;
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T> Quat<T>::BarySLerp(const Quat<T>& q0,
                                     const Quat<T>& q1,
                                     const Quat<T>& q2,
                                     T a, T b)
{
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

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T> Quat<T>::RotationBetween(const Vector<3, T>& a, const Vector<3, T>& b)
{
    Vector<3, T> aCrossB = Cross(a, b);
    T aDotB = a.Dot(b);
    if(aCrossB != Vector<3, T>::Zero())
        aCrossB.NormalizeSelf();
    return Quat<T>(acos(aDotB), aCrossB);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T> Quat<T>::RotationBetweenZAxis(const Vector<3, T>& b)
{
    using namespace MathConstants;

    Vector<3, T> zCrossD(-b[1], b[0], 0);
    T zDotD = b[2];

    // Half angle theorem
    T sin = sqrt((1 - zDotD) * static_cast<T>(0.5));
    T cos = sqrt((zDotD + 1) * static_cast<T>(0.5));

    zCrossD.NormalizeSelf();
    T x = zCrossD[0] * sin;
    T y = zCrossD[1] * sin;
    T z = zCrossD[2] * sin;
    T w = cos;
    // Handle singularities
    if(abs(zDotD + 1) < LargeEpsilon<T>())
    {
        // Spaces are 180 degree apart
        // Define pi turn
        return Quat<T>(0, 0, 1, 0);
    }
    else if(abs(zDotD - 1) < LargeEpsilon<T>())
    {
        // Spaces are nearly equivalent
        // Just turn identity
        return Quat<T>(1, 0, 0, 0);
    }
    else return Quat<T>(w, x, y, z);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T> Quat<T>::Identity()
{
    return Quat<T>(1, 0, 0, 0);
}

template<FloatingPointC T>
MRAY_HYBRID MRAY_CGPU_INLINE
constexpr Quat<T> operator*(T t, const Quat<T>& q)
{
    return q * t;
}

template <class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Quat<T> TransformGen::Space(const Vector<3, T>& xIn,
                            const Vector<3, T>& y,
                            const Vector<3, T>& z)
{
    Quat<T> q;
    // Flip the coordinate system if inverted
    Vector<3, T> x = xIn;
    if((Cross(x, y) - z).Abs() > Vector3(0.1))
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
    q *= T{0.5} / sqrt(t);
    q.NormalizeSelf();
    q.ConjugateSelf();
    return q;

    //// Another implementation that i found in stack overflow
    //// https://stackoverflow.com/questions/63734840/how-to-convert-rotation-matrix-to-quaternion
    //// Clang min definition is only on std namespace
    //// this is a crappy workaround
    //#ifndef __CUDA_ARCH__
    //    using namespace std;
    //#endif
    //// Our sign is one (according to the above link)
    //static constexpr T sign = 1;
    //T t = x[0] + y[1] + z[2];
    //T m = max(max(x[0], y[1]), max(z[2], t));
    //T qmax = static_cast<T>(0.5) * sqrt(1 - t + 2 * m);
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

template <class T>
MRAY_HYBRID MRAY_CGPU_INLINE
Quat<T> TransformGen::InvSpace(const Vector<3, T>& x,
                            const Vector<3, T>& y,
                            const Vector<3, T>& z)
{
    return Space(x, y, z).Conjugate();
}