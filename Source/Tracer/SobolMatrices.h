#pragma once

#include <cstdint>
#include <array>

namespace SobolDetail
{
    static constexpr uint32_t SOBOL_MATRIX_WIDTH = 52;
    static constexpr uint32_t SOBOL_DIM_COUNT = 256;
    static constexpr uint32_t SOBOL_DATA_SIZE = SOBOL_DIM_COUNT * SOBOL_MATRIX_WIDTH;
    
    using SobolData = std::array<uint32_t, SobolDetail::SOBOL_DATA_SIZE>;
    extern const SobolData SobolMatrices;
}