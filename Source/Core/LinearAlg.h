#pragma once

// Time has come... We need to define some
// matrix solving algorithms.
#include <concepts>

#include "Matrix.h"

namespace LinearAlg
{
    template<unsigned int N, std::floating_point T>
    struct LUPResult
    {
        Matrix<N, T> LU;
        Vector<N, uint8_t> P;
        bool success = false;
    };

    // https://en.wikipedia.org/wiki/LU_decomposition
    template<unsigned int N, std::floating_point T>
    constexpr LUPResult<N, T>
    LUDecompose(const Matrix<N, T>&);

    template<unsigned int N, std::floating_point T>
    constexpr Vector<N, T>
    SolveWithLU(const LUPResult<N, T>&,
                const Vector<N, T>&);
}

template<unsigned int N, std::floating_point T>
constexpr LinearAlg::LUPResult<N, T>
LinearAlg::LUDecompose(const Matrix<N, T>& M)
{
    static_assert(N <= std::numeric_limits<uint8_t>::max(),
                  "LUDecomposition matrix size exceeded the uint8_t size");
    LUPResult<N, T> r;
    r.LU = M;
    auto& LU = r.LU;
    auto& P = r.P;

    // Unit permutation (iota)
    MRAY_UNROLL_LOOP_N(N)
    for(uint8_t i = 0; i < uint8_t(N); i++)
        P[i] = i;

    // Actual computation
    for(uint32_t i = 0; i < N; i++)
    {
        T maxVal = T(0.0);
        uint32_t maxI = i;
        //
        for(uint32_t k = i; k < N; k++)
        {
            T absVal = Math::Abs(LU(k, i));
            if(absVal > maxVal)
            {
                maxVal = absVal;
                maxI = k;
            }
        }
        // Fail is near degenerate
        if constexpr(std::is_same_v<Float, double>)
        {
            if(maxVal < MathConstants::VerySmallEpsilon<T>())
                return r;
        }
        else
        {
            if(maxVal < 1e-16) return r;
        }
        // Pivot check
        if (maxI != i)
        {
            // Store the pivot info
            std::swap(P[i], P[maxI]);

            // Pivot the rows
            MRAY_UNROLL_LOOP_N(N)
            for(uint32_t x = 0; x < N; x++)
                std::swap(LU(i, x), LU(maxI, x));
        }

        T diagFactor = Float(1) / LU(i, i);
        for(uint32_t j = i + 1; j < N; j++)
        {
            LU(j, i) *= diagFactor;

            MRAY_UNROLL_LOOP_N(N)
            for(uint32_t k = i + 1; k < N; k++)
                LU(j, k) -= LU(j, i) * LU(i, k);
        }
    }
    r.success = true;
    return r;
}

template<unsigned int N, std::floating_point T>
constexpr Vector<N, T>
LinearAlg::SolveWithLU(const LinearAlg::LUPResult<N, T>& r,
                       const Vector<N, T>& y)
{
    auto& LU = r.LU;
    auto& P = r.P;

    Vector<N, T> x;
    // A * x = y; solve for x
    // A is already decomposed
    for(uint32_t i = 0; i < N; i++)
    {
        x[i] = y[P[i]];

        for(uint32_t k = 0; k < i; k++)
            x[i] -= LU(i,k) * x[k];
    }

    for(int32_t i = int32_t(N - 1); i >= 0; i--)
    {
        for(int32_t k = i + 1; k < N; k++)
            x[i] -= LU(i,k) * x[k];

        x[i] /= LU(i, i);
    }
    return x;
}