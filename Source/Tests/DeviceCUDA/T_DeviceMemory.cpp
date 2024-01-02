#include <gtest/gtest.h>
#include <random>

#include "Device/GPUSystem.h"
#include "Core/MemAlloc.h"
#include "Core/Log.h"

TEST(GPUMemory, DeviceMemory_Allocate)
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
    std::mt19937 rng(321);
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

    // Memcpy Test
    CUDA_CHECK(cudaSetDevice(system.BestDevice().DeviceId()));
    for(size_t i = 0; i < 2; i++)
    {
        DeviceMemory memoryFrom({&system.BestDevice()}, 3_MiB << i, 8_MiB << i);
        DeviceMemory memoryTo({&system.BestDevice()}, 1_MiB << i, 4_MiB << i);
        memoryFrom.ResizeBuffer(2_MiB << i);
        memoryTo.ResizeBuffer(1_MiB << i);

        size_t copySize = 1_MiB << i;

        CUDA_CHECK(cudaMemset(static_cast<void*>(memoryTo), 0x00,
                              memoryTo.Size()));
        CUDA_CHECK(cudaMemset(static_cast<void*>(memoryFrom), 0x11,
                              memoryFrom.Size()));

        CUDA_CHECK(cudaMemcpy(static_cast<void*>(memoryTo),
                              static_cast<void*>(memoryFrom),
                              copySize, cudaMemcpyDefault));

        std::vector<Byte> hostAlloc(copySize, 0x00);
        CUDA_CHECK(cudaMemcpy(hostAlloc.data(),
                              static_cast<void*>(memoryTo),
                              copySize, cudaMemcpyDefault));

        for(Byte b : hostAlloc)
        {
            EXPECT_EQ(b, 0x11);
        }
    }
}

TEST(GPUMemory, DeviceLocalMemory_Allocate)
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
    {
        DeviceLocalMemory memory(gpu, 32_MiB);
        memory.ResizeBuffer(40_MiB);
        memory.ResizeBuffer(8_MiB);
    }
    {
        DeviceLocalMemory memory(gpu, 9_MiB);
        memory.ResizeBuffer(1_MiB);
        memory.ResizeBuffer(3_MiB);
    }

    // Fuzz test
    std::mt19937 rng(123);
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

TEST(GPUMemory, DeviceLocalMemory_Migrate)
{
    GPUSystem system;
    if(system.SystemDevices().size() == 1)
    {
        GTEST_SKIP() << "Skipping DeviceLocalMemory.Migrate..(), since there is only one GPU";
    }

    // TODO: ....
}

TEST(GPUMemory, HostLocalMemory_Allocate)
{
    GPUSystem system;
    // Check invalid params
    {
        EXPECT_DEBUG_DEATH(HostLocalMemory memory(system, 0), ".*");
    }
    // Simple Test
    {
        HostLocalMemory memory(system, 16_MiB);
        memory.ResizeBuffer(8_MiB);
        memory.ResizeBuffer(4_MiB);
    }
    {
        HostLocalMemory memory(system, 32_MiB);
        memory.ResizeBuffer(40_MiB);
        memory.ResizeBuffer(8_MiB);
    }
    {
        HostLocalMemory memory(system, 9_MiB);
        memory.ResizeBuffer(1_MiB);
        memory.ResizeBuffer(3_MiB);
    }

    // Fuzz test
    std::mt19937 rng(123);
    for(int i = 0; i < 4; i++)
    {
        std::uniform_int_distribution<size_t> distAlloc(1_KiB, 32_MiB);

        size_t dA = distAlloc(rng);
        HostLocalMemory memory(system, dA);

        for(int j = 0; j < 8; j++)
        {
            std::uniform_int_distribution<size_t> distEnlarge(1_KiB, 256_MiB);
            size_t newSize = distEnlarge(rng);
            memory.ResizeBuffer(newSize);
        }
    }


    // Memcpy Test
    for(int i = 0; i < 3; i++)
    {
        DeviceMemory memoryFrom({&system.BestDevice()}, 3_MiB << i, 8_MiB << i);
        HostLocalMemory memoryTo(system, 1_MiB << i);
        memoryFrom.ResizeBuffer(2_MiB << i);
        size_t copySize = 1_MiB << i;

        CUDA_CHECK(cudaSetDevice(system.BestDevice().DeviceId()));

        CUDA_CHECK(cudaMemset(memoryTo.DevicePtr(), 0x00,
                              memoryTo.Size()));
        CUDA_CHECK(cudaMemset(static_cast<void*>(memoryFrom), 0x11,
                              memoryFrom.Size()));

        CUDA_CHECK(cudaMemcpy(memoryTo.DevicePtr(),
                              static_cast<void*>(memoryFrom),
                              copySize, cudaMemcpyDefault));

        Byte* hData = reinterpret_cast<Byte*>(memoryTo.HostPtr());
        for(size_t j = 0 ; j < memoryTo.Size(); j++)
        {
            EXPECT_EQ(hData[j], 0x11);
        }
    }
}