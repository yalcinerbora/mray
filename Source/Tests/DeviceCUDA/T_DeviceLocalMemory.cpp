#include <gtest/gtest.h>
#include <random>

#include "Device/GPUSystem.h"
#include "Core/MemAlloc.h"
#include "Core/Log.h"

TEST(GPUMemory, DeviceLocalMemory)
{
    GPUSystem system;
    const GPUDevice& gpu = system.BestDevice();
    // Check invalid params
    {
        EXPECT_DEBUG_DEATH(DeviceLocalMemory memory(gpu, 0), ".*");
    }
    // Simple Test
    {
        DeviceLocalMemory memory(gpu, 16_MiB);
        memory.ResizeBuffer(8_MiB);
        memory.ResizeBuffer(4_MiB);
    }
    //// Regression of two specific cases encountered from the fuzz test
    //{
    //    DeviceLocalMemory memory(gpu, 25141063);
    //    memory.ResizeBuffer(148228496);
    //    memory.ResizeBuffer(196412823);
    //    memory.ResizeBuffer(156894527);
    //}
    //{
    //    DeviceLocalMemory memory(gpu, 14452429);
    //    memory.ResizeBuffer(37685711);
    //    memory.ResizeBuffer(252095261);
    //    memory.ResizeBuffer(53178283);
    //}

    // Fuzz test
    std::mt19937 rng(0);
    for(int i = 0; i < 4; i++)
    {
        std::uniform_int_distribution<size_t> distAlloc(1_KiB, 32_MiB);

        size_t dA = distAlloc(rng);
        DeviceLocalMemory memory(gpu, dA);

        for(int j = 0; j < 8; j++)
        {
            std::uniform_int_distribution<size_t> distEnlarge(1_KiB, 256_MiB);
            size_t newSize = distEnlarge(rng);
            memory.ResizeBuffer(newSize);
        }
    }
}