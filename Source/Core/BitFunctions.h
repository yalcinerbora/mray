#pragma once

#include <concepts>
#include <cassert>

#include "Vector.h"

namespace BitFunctions
{
    template<std::integral T>
    constexpr T FetchSubPortion(T value, Vector<2, T> bitRange);
}

template<std::integral T>
constexpr T BitFunctions::FetchSubPortion(T value, Vector<2, T> bitRange)
{
    assert(bitRange[0] < bitRange[1]);
    T bitCount = bitRange[1] - bitRange[0];
    T mask = (1 << bitCount) - 1;
    return (value >> bitRange[0]) & mask;
}