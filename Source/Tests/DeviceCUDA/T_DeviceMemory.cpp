#include <gtest/gtest.h>
#include <random>

#include "Device/GPUSystem.h"
#include "Core/MemAlloc.h"
#include "Core/Log.h"
#include "Core/Vector.h"
#include "Core/Matrix.h"

#include "Device/GPUTypes.h"

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

        CUDA_CHECK(cudaMemset(static_cast<Byte*>(memoryTo), 0x00,
                              memoryTo.Size()));
        CUDA_CHECK(cudaMemset(static_cast<Byte*>(memoryFrom), 0x12,
                              memoryFrom.Size()));

        CUDA_CHECK(cudaMemcpy(static_cast<Byte*>(memoryTo),
                              static_cast<Byte*>(memoryFrom),
                              copySize, cudaMemcpyDefault));

        std::vector<Byte> hostAlloc(copySize, Byte{0x00});
        CUDA_CHECK(cudaMemcpy(hostAlloc.data(),
                              static_cast<Byte*>(memoryTo),
                              copySize, cudaMemcpyDefault));

        for(Byte b : hostAlloc)
        {
            EXPECT_EQ(b, Byte{0x12});
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
        EXPECT_DEBUG_DEATH(HostLocalMemory memory(system, 0, true), ".*");
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
        CUDA_CHECK(cudaMemset(static_cast<Byte*>(memoryFrom), 0x11,
                              memoryFrom.Size()));

        CUDA_CHECK(cudaMemcpy(memoryTo.DevicePtr(),
                              static_cast<Byte*>(memoryFrom),
                              copySize, cudaMemcpyDefault));

        const Byte* hData = static_cast<const Byte*>(memoryTo);
        for(size_t j = 0 ; j < memoryTo.Size(); j++)
        {
            EXPECT_EQ(hData[j], Byte{0x11});
        }
    }
}

template <class T>
class GPUMemoryAlloc : public testing::Test
{
    using MemoryType = T;

    public:
    GPUSystem                   system;
    std::unique_ptr<MemoryType> memory;

    public:
    GPUMemoryAlloc()
    {
        if constexpr(std::is_same_v<MemoryType, DeviceMemory>)
        {
            std::vector<const GPUDevice*> devices = {&system.BestDevice()};
            memory = std::make_unique<MemoryType>(devices, 4_MiB, 16_MiB);
        }
        else if constexpr(std::is_same_v<MemoryType, DeviceLocalMemory>)
            memory = std::make_unique<MemoryType>(system.BestDevice());
        else
            memory = std::make_unique<MemoryType>(system);
    };

};

using Implementations = ::testing::Types<DeviceLocalMemory,
                                         HostLocalMemory,
                                         DeviceMemory>;

TYPED_TEST_SUITE(GPUMemoryAlloc, Implementations);

TYPED_TEST(GPUMemoryAlloc, MultiAlloc)
{
    Span<Float> dFloats;
    Span<Vector3> dVectors;
    Span<Matrix4x4> dMatrices;
    MemAlloc::AllocateMultiData(std::tie(dFloats,
                                         dVectors,
                                         dMatrices),
                                *(this->memory),
                                {100, 50, 40});

    this->system.Memset<Float>(dFloats, 0x12);
    this->system.Memset<Vector3>(dVectors, 0x34);
    this->system.Memset<Matrix4x4>(dMatrices, 0x56);

    std::vector<Float> hFloats(dFloats.size(), 0.0f);
    std::vector<Vector3> hVectors(dVectors.size(), Vector3::Zero());
    std::vector<Matrix4x4> hMatrices(dMatrices.size(), Matrix4x4::Zero());
    this->system.Memcpy(Span<Float>(hFloats.begin(), hFloats.end()),
                        ToConstSpan(dFloats));
    this->system.Memcpy(Span<Vector3>(hVectors.begin(), hVectors.end()),
                        ToConstSpan(dVectors));
    this->system.Memcpy(Span<Matrix4x4>(hMatrices.begin(), hMatrices.end()),
                        ToConstSpan(dMatrices));
    const Byte* data = nullptr;

    data = reinterpret_cast<const Byte*>(hFloats.data());
    for(size_t i = 0; i < dFloats.size_bytes(); i++)
    {
        EXPECT_EQ(data[i], Byte{0x12});
    }

    data = reinterpret_cast<const Byte*>(hVectors.data());
    for(size_t i = 0; i < dVectors.size_bytes(); i++)
    {
        EXPECT_EQ(data[i], Byte{0x34});
    }

    data = reinterpret_cast<const Byte*>(hMatrices.data());
    for(size_t i = 0; i < dMatrices.size_bytes(); i++)
    {
        EXPECT_EQ(data[i], Byte{0x56});
    }
}
