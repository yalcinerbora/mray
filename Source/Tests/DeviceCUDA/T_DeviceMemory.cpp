#include <gtest/gtest.h>
#include <random>

#include "Device/GPUSystem.h"
#include "Core/MemAlloc.h"
#include "Core/Log.h"

TEST(GPUMemory, DeviceMemory)
{
    GPUSystem system;
    // Check invalid params
    {
        EXPECT_DEBUG_DEATH(DeviceMemory memory({&system.BestDevice()}, 0, 0), ".*");
    }
    // Simple Test
    {
        DeviceMemory memory({&system.BestDevice()}, 4_MiB, 16_MiB);
        memory.ResizeBuffer(8_MiB);
        memory.ResizeBuffer(4_MiB);
    }
    // Regression of two specific cases encountered from the fuzz test
    {
        DeviceMemory memory({&system.BestDevice()}, 263362, 25141063);
        memory.ResizeBuffer(148228496);
        memory.ResizeBuffer(196412823);
        memory.ResizeBuffer(156894527);
    }
    {
        DeviceMemory memory({&system.BestDevice()}, 1757938, 14452429);
        memory.ResizeBuffer(37685711);
        memory.ResizeBuffer(252095261);
        memory.ResizeBuffer(53178283);
    }

    // Fuzz test
    std::mt19937 rng(0);
    for(int i = 0; i < 4; i++)
    {
        std::uniform_int_distribution<size_t> distAlloc(1_KiB, 3_MiB);
        std::uniform_int_distribution<size_t> distReserve(1_KiB, 32_MiB);

        size_t dA = distAlloc(rng);
        size_t dR = distReserve(rng);
        DeviceMemory memory({&system.BestDevice()},
                            dA, dR);

        for(int j = 0; j < 8; j++)
        {
            std::uniform_int_distribution<size_t> distEnlarge(1_KiB, 256_MiB);
            size_t newSize = distEnlarge(rng);
            memory.ResizeBuffer(newSize);
        }
    }
}