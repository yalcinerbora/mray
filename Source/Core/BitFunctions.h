#pragma once

#include <concepts>
#include <cassert>
#include <bit>

#include "Vector.h"

namespace BitFunctions
{
    template<std::integral T>
    constexpr T FetchSubPortion(T value, Vector<2, T> bitRange);

    template<std::unsigned_integral T>
    constexpr T RotateLeft(T value, T shiftAmount);

    template<std::unsigned_integral T>
    constexpr T RotateRight(T value, T shiftAmount);

    template<std::unsigned_integral T>
    constexpr T RequiredBitsToRepresent(T value);
}

template<std::integral T>
constexpr T BitFunctions::FetchSubPortion(T value, Vector<2, T> bitRange)
{
    assert(bitRange[0] < bitRange[1]);
    T bitCount = bitRange[1] - bitRange[0];
    T mask = (1 << bitCount) - 1;
    return (value >> bitRange[0]) & mask;
}

template<std::unsigned_integral T>
constexpr T BitFunctions::RotateLeft(T value, T shiftAmount)
{
    constexpr T Bits = sizeof(T) * CHAR_BIT;

    T result = (value << shiftAmount);
    result |= (value >> (Bits - shiftAmount));
    return result;
}

template<std::unsigned_integral T>
constexpr T BitFunctions::RotateRight(T value, T shiftAmount)
{
    constexpr T Bits = sizeof(T) * CHAR_BIT;

    T result = (value >> shiftAmount);
    result |= (value << (Bits - shiftAmount));
    return result;
}

template<std::unsigned_integral T>
constexpr T BitFunctions::RequiredBitsToRepresent(T value)
{
    constexpr T Bits = sizeof(T) * CHAR_BIT;
    return (Bits - std::countl_zero(value));
}