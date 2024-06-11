#pragma once

#include <concepts>

#define MRAY_FPSC template <std::floating_point T> static constexpr

namespace MathConstants
{
    MRAY_FPSC T Pi()            { return T(3.1415926535897932384626433); }
    MRAY_FPSC T PiSqr()         { return Pi<T>() * Pi<T>(); }
    MRAY_FPSC T InvPi()         { return T(1) / Pi<T>(); }
    MRAY_FPSC T Inv2Pi()        { return T(0.5) / Pi<T>(); }
    MRAY_FPSC T Inv4Pi()        { return T(0.25) / Pi<T>(); }
    MRAY_FPSC T InvPiSqr()      { return T{1} / (Pi<T>() * Pi<T>()); }
    MRAY_FPSC T SqrtPi()        { return T(1.772453850905516027298167); }
    MRAY_FPSC T Sqrt2()         { return T(1.4142135623730950488016887); }
    MRAY_FPSC T Sqrt3()         { return T(1.7320508075688772935274463); }
    MRAY_FPSC T Euler()         { return T(2.7182818284590452353602874); }
    MRAY_FPSC T InvEuler()      { return T(1) / Euler<T>(); }
    MRAY_FPSC T DegToRadCoef()  { return Pi<T>() / T(180); }
    MRAY_FPSC T RadToDegCoef()  { return T(180) / Pi<T>(); }

    // Epsilons (Totally arbitrary)
    // TODO: Reason about these values
    MRAY_FPSC T VerySmallEpsilon()  { return T(1.0e-8); }
    MRAY_FPSC T SmallEpsilon()      { return T(1.0e-7); }
    MRAY_FPSC T Epsilon()           { return T(1.0e-5); }
    MRAY_FPSC T LargeEpsilon()      { return T(1.0e-4); }
    MRAY_FPSC T VeryLargeEpsilon()  { return T(1.0e-3); }
    MRAY_FPSC T HugeEpsilon()       { return T(1.0e-2); }

}

#undef MRAY_PFC

// TODO: Hopefully, these will be half and constexpr in future.
#define MRAY_HALF_MAX 65504.0f
#define MRAY_HALF_MIN 0.000000059604645f

