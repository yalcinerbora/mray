#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>

#include "Device/GPUSystem.h"
#include "Device/TextureCUDA.h"

#include "T_TextureTypes.h"


TYPED_TEST(GPUTextureTest, Construct)
{
    GPUSystem system;

    static constexpr auto D = GPUTextureTest<TypeParam>::D;
    using ChannelType = typename GPUTextureTest<TypeParam>::ChannelType;
    using ParamType = typename GPUTextureTest<TypeParam>::ParamType;
    using SizeType = typename ParamType::SizeType;

    {
        ParamType tParams = {};
        tParams.size = SizeType(33);
        tParams.mipCount = 1;

        Texture<D, ChannelType> tex(system.BestDevice(), tParams);
    }

}


TYPED_TEST(GPUTextureTest, Allocate)
{
    static constexpr auto D = 3;
    using ChannelType = Vector3uc;

    using ParamType = TextureInitParams<D, ChannelType>;
    using SizeType = typename ParamType::SizeType;
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


