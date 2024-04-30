#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"
#include "Device/TextureCUDA.h"

#include "T_TextureTypes.h"

template <uint32_t D, class T>
MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void CheckCheckerboardPattern(Span<uint8_t> dResults,
                              TextureView<D, T> tex,
                              TextureExtent<D> extent,
                              uint32_t totalPixels)
{

    KernelCallParams params;
    for(uint32_t globalId = params.GlobalId();
        globalId < totalPixels;
        globalId += params.TotalSize())
    {
        UVType<D> uv = LinearToFloatIndex<D>(extent, globalId);
        T out = tex(uv).value();
        T compare = (globalId % 2 == 0) ? T(1) : T(0);
        uint8_t result = (out != compare) ? 1 : 0;
        dResults[globalId] = result;
    }
}

TYPED_TEST(GPUTextureTest, Construct)
{
    GPUSystem system;

    static constexpr auto D = GPUTextureTest<TypeParam>::D;
    using ChannelType = typename GPUTextureTest<TypeParam>::ChannelType;
    using ParamType = typename GPUTextureTest<TypeParam>::ParamType;
    using SizeType = typename GPUTextureTest<TypeParam>::SizeType;

    {
        ParamType tParams = {};
        tParams.size = SizeType(33);
        tParams.mipCount = 1;
        tParams.normIntegers = false;

        Texture<D, ChannelType> tex(system.BestDevice(), tParams);
    }
}

TYPED_TEST(GPUTextureTest, Allocate)
{
    static constexpr auto D = GPUTextureTest<TypeParam>::D;
    using ChannelType = typename GPUTextureTest<TypeParam>::ChannelType;
    using ParamType = typename GPUTextureTest<TypeParam>::ParamType;
    using SizeType = typename GPUTextureTest<TypeParam>::SizeType;
    using TexType = Texture<D, ChannelType>;
    static constexpr uint32_t TOTAL_TEX_COUNT = 128;

    GPUSystem system;

    std::vector<TexType> textures;
    std::vector<size_t> sizes;
    std::vector<size_t> alignments;
    textures.reserve(TOTAL_TEX_COUNT);
    sizes.reserve(TOTAL_TEX_COUNT);
    alignments.reserve(TOTAL_TEX_COUNT);
    for(uint32_t i = 0; i  < TOTAL_TEX_COUNT; i++)
    {
        // Do a default allocate
        ParamType tParams = {};
        tParams.size = SizeType(64);
        tParams.mipCount = 1;
        tParams.normIntegers = false;

        TexType tex(system.BestDevice(), tParams);

        sizes.emplace_back(tex.Size());
        alignments.emplace_back(tex.Alignment());
        textures.push_back(std::move(tex));
    }

    // Allocation
    TextureBackingMemory memory(system.BestDevice());
    std::vector<size_t> offsets;
    MemAlloc::AllocateTextureSpace(offsets, memory, sizes, alignments);

    // Committing memory
    std::vector<Span<Byte>> textureAllocations;
    for(uint32_t i = 0; i < TOTAL_TEX_COUNT; i++)
    {
        const GPUQueue& queue = system.BestDevice().GetQueue(0);
        textures[i].CommitMemory(queue, memory, offsets[i]);
    }
    // Commit is async so wait before destruction
    GPUFence fence = system.BestDevice().GetQueue(0).Barrier();

    fence.Wait();

}


TYPED_TEST(GPUTextureTest, Copy)
{
    static constexpr auto D = GPUTextureTest<TypeParam>::D;
    using ChannelType = typename GPUTextureTest<TypeParam>::ChannelType;
    using ParamType = typename GPUTextureTest<TypeParam>::ParamType;
    using SizeType = typename GPUTextureTest<TypeParam>::SizeType;
    using TexType = Texture<D, ChannelType>;
    using PaddedChannelType = typename TexType::PaddedChannelType;

    GPUSystem system;

    // Allocate Asymetric to check padding/access issues
    SizeType sz = SizeType(16);
    if constexpr(D == 2)
        sz = SizeType(16, 32);
    else if constexpr(D == 3)
        sz = SizeType(16, 32, 8);


    // Do a default allocate
    ParamType tParams = {};
    tParams.size = sz;
    tParams.mipCount = 1;
    tParams.normIntegers = false;
    tParams.normCoordinates = false;
    tParams.interp = InterpolationType::NEAREST;
    TexType tex(system.BestDevice(), tParams);

    // Allocation
    TextureBackingMemory memory(system.BestDevice());
    memory.ResizeBuffer(tex.Size());

    const GPUQueue& queue = system.BestDevice().GetQueue(0);
    tex.CommitMemory(queue, memory, 0);

    // Get a fence and calculate a checkerboard pattern
    GPUFence afterAllocFence = system.BestDevice().GetQueue(0).Barrier();

    uint32_t total;
    if constexpr (D == 1)
        total = tParams.size;
    else
        total = tParams.size.Multiply();

    std::vector<PaddedChannelType> hData(total, PaddedChannelType(0));
    for(uint32_t i = 0; i < total; i++)
    {
        if((i % 2) == 0)
            hData[i] = PaddedChannelType(1);
    };
    afterAllocFence.Wait();
    // Mem is ready now go memcpy
    tex.CopyFromAsync(queue, 0u, TextureExtent<D>(0), tex.Extents(),
                      Span<const PaddedChannelType>(hData.data(), hData.size()));

    DeviceLocalMemory mem(system.BestDevice());
    Span<uint8_t> dTrueFalseBuffer;
    MemAlloc::AllocateMultiData(std::tie(dTrueFalseBuffer), mem,
                                {total});
    dTrueFalseBuffer = dTrueFalseBuffer.subspan(0, total);

    // Get a texture view (same as the inner type)
    TextureView<D, ChannelType> view = tex. template View<ChannelType>();
    queue.IssueKernel<CheckCheckerboardPattern<D, ChannelType>>
    (
        "GTest Texture CheckCheckerboard",
        KernelIssueParams{.workCount = total, .sharedMemSize = 0},
        dTrueFalseBuffer,
        view,
        tex.Extents(),
        total
    );

    std::vector<uint8_t> hTrueFalseBuffer(total, 0);
    queue.MemcpyAsync(Span<uint8_t>(hTrueFalseBuffer),
                      ToConstSpan(dTrueFalseBuffer));
    // Wait the copy
    system.BestDevice().GetQueue(0).Barrier().Wait();

    uint8_t anyFalse = std::reduce
    (   hTrueFalseBuffer.cbegin(),
        hTrueFalseBuffer.cend(),
        uint8_t{0x00},
        [](uint8_t lhs, uint8_t rhs)
        {
            return lhs | rhs;
        }
    );
    EXPECT_EQ(anyFalse, 0x00);
}
