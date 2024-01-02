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
    //{
    //    EXPECT_DEBUG_DEATH(DeviceLocalMemory memory(gpu, 0), ".*");
    //}
    // Simple Test
    {
        DeviceLocalMemory memory(gpu, 16_MiB);
        memory.ResizeBuffer(8_MiB);
        memory.ResizeBuffer(4_MiB);
    }
    {
        DeviceLocalMemory memory(gpu, 32_MiB);
        memory.ResizeBuffer(40_MiB);
        //memory.ResizeBuffer(8_MiB);
    }
    {
        DeviceLocalMemory memory(gpu, 9_MiB);
        memory.ResizeBuffer(1_MiB);
        memory.ResizeBuffer(3_MiB);
    }

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