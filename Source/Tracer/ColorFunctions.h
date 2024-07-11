#pragma once

#include "Core/Vector.h"

namespace Color
{
    MRAY_HYBRID Vector3   HSVToRGB(const Vector3& hsv);

    // Stateless random color,
    // Neighbouring pixels should have distinct colors
    MRAY_HYBRID Vector3   RandomColorRGB(uint32_t index);
}


MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 Color::HSVToRGB(const Vector3& hsv)
{
    // H, S, V both normalized
    // H: [0-1) (meaning 0 is 0, 1 is 360)
    // S: [0-1] (meaning 0 is 0, 1 is 100)
    // V: [0-1] (meaning 0 is 0, 1 is 100)
    Float h = hsv[0] * Float(360);
    constexpr Float o60 = Float(1) / Float(60);

    Float c = hsv[2] * hsv[1];
    Float m = hsv[2] - c;
    Float x;
    if constexpr(std::is_same_v<Float, float>)
    {
        x = fmodf(h * o60, Float(2));
        x = fabsf(x - Float(1));
    }
    else
    {
        x = fmod(h * o60, Float(2));
        x = fabs(x - Float(1));
    }
    x = c * (Float(1) - x);

    Vector3 result;
    int sextant = static_cast<int>(h) / 60 % 6;
    switch(sextant)
    {
        case 0: result = Vector3(c, x, 0); break;
        case 1: result = Vector3(x, c, 0); break;
        case 2: result = Vector3(0, c, x); break;
        case 3: result = Vector3(0, x, c); break;
        case 4: result = Vector3(x, 0, c); break;
        case 5: result = Vector3(c, 0, x); break;
    }
    result = result + m;
    return result;
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector3 Color::RandomColorRGB(uint32_t index)
{
    // https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
    constexpr Float SATURATION = Float(0.75);
    constexpr Float VALUE = Float(0.95);
    constexpr Float GOLDEN_RATIO_CONJ = Float(0.618033988749895);
    // For large numbers use double arithmetic here
    float hue = 0.1f + static_cast<float>(index) * GOLDEN_RATIO_CONJ;
    hue = fmod(hue, 1.0f);

    return HSVToRGB(Vector3f(hue, SATURATION, VALUE));
}