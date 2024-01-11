#include <gtest/gtest.h>
#include <random>

#include "Tracer/RayPartitioner.h"
#include "Tracer/Key.h"

void SimulateBasicPathTracer()
{
    GPUSystem system;
    const GPUQueue& queue = system.BestDevice().GetQueue(0);

    static constexpr size_t RayCount = 1'000'000;
    // This is calculated via combining
    // medium type count & surface type count.
    // Technically this should be low. In a case with
    // uniform paritioning, "only" ~4k elements will be called.
    static constexpr uint32_t MaxPartitionCount = 256;

    DeviceMemory testMemory(system.AllGPUs(), 1_MiB, RayCount);
    Span<uint8_t> dIsRayAliveBuffer;
    Span<CommonKey> dHitKeyBuffer;
    MemAlloc::AllocateMultiData(std::tie(dIsRayAliveBuffer, dHitKeyBuffer),
                                testMemory,
                                {RayCount, RayCount});

    // Maximum medium counts / surface counts on scene.
    constexpr Vector2ui materialBits = Vector2ui(6, 8);
    constexpr Vector2ui materialCount = Vector2ui(1 << materialBits[0],
                                                  1 << materialBits[1]);
    // TODO: compile-time determine from "materialCount"
    constexpr Vector2ui materialBitStart = Vector2ui(16, 16);
    using TheKey = KeyT<CommonKey, materialBitStart[0], materialBitStart[1]>;
    constexpr Vector2ui materialBatchRange = Vector2ui(materialBitStart[0],
                                                       materialBitStart[0] + materialBits[0]);
    constexpr Vector2ui materialDataRange = Vector2ui(materialBitStart[1],
                                                      materialBitStart[1] + materialBits[1]);



    static_assert(MaxPartitionCount >= materialCount[0],
                  "Partitioner maximum partition count is not set correctly!");

    RayPartitioner partitioner(system, RayCount, MaxPartitionCount);
    queue.MemsetAsync(dIsRayAliveBuffer, 0x00);
    queue.MemsetAsync(dHitKeyBuffer, 0x00);

    // Simulate full path tracing pipeline
    // =================== //
    //        Start        //
    // =================== //
    auto [dInitialIndices, dInitialKeys] = partitioner.Start(RayCount, MaxPartitionCount);
    EXPECT_EQ(dInitialIndices.size(), RayCount);
    DeviceAlgorithms::Iota(dInitialIndices, CommonIndex{0}, queue);

    // Replace dead rays
    auto [hDeadAlivePartitions, dDeadAliveIndices] = partitioner.BinaryPartition
    (
        ToConstSpan(dInitialIndices), queue,
        [=]MRAY_HYBRID(const CommonIndex& i)
        {
            return !dIsRayAliveBuffer[i];
        }
    );
    // We need to wait for the host buffers to be filled
    queue.Barrier().Wait();

    // =================== //
    //   Issue New Rays    //
    // =================== //
    uint32_t deadRayCount = hDeadAlivePartitions[1] - hDeadAlivePartitions[0];
    queue.IssueLambda
    (
        KernelIssueParams{.workCount = deadRayCount},
        [=]MRAY_HYBRID(const KernelCallParams kp)
        {
            for(uint32_t i = kp.GlobalId(); i < deadRayCount;
                i += kp.TotalSize())
            {
                uint32_t index = dDeadAliveIndices[i];
                dIsRayAliveBuffer[index] = true;
            }
        }
    );


    // =================== //
    //   Check All Alive   //
    // =================== //
    std::vector<uint8_t> hRayStatus(RayCount, false);
    Span<uint8_t> hRayStatusSpan(hRayStatus.begin(), hRayStatus.end());
    queue.MemcpyAsync(hRayStatusSpan, ToConstSpan(dIsRayAliveBuffer));
    queue.Barrier().Wait();
    for(uint8_t b : hRayStatus)
    {
        EXPECT_EQ(b, false);
    }

    // =================== //
    //      Hit Rays       //
    // =================== //
    // Reset the indices
    DeviceAlgorithms::Iota(dInitialIndices, CommonIndex{0}, queue);

    std::mt19937 rng(123);
    std::vector<CommonKey> hHitKeys(RayCount, CommonKey{0});
    Span<CommonKey> hHitKeysSpan(hHitKeys.begin(), hHitKeys.end());
    for(CommonKey& k : hHitKeys)
    {
        std::uniform_int_distribution<CommonKey> distH(0, materialCount[0]);
        std::uniform_int_distribution<CommonKey> distL(0, materialCount[1]);

        k = TheKey::CombinedKey(distH(rng), distL(rng));
    }
    queue.MemcpyAsync(dInitialKeys, ToConstSpan(hHitKeysSpan));

    // =================== //
    // Partition wrt. Mat  //
    // =================== //
    auto
    [
        hMatPartitionOffsets,
        hMatPartitionCount,
        hMatPartitionKeys,
        dMatPartitionIndices,
        dMatPartitionKeys
    ] = partitioner.MultiPartition(dInitialKeys,
                                   dInitialIndices,
                                   materialDataRange,
                                   materialBatchRange,
                                   queue);
    // We need to wait for the host buffers to be filled
    queue.Barrier().Wait();
    for(uint32_t partitionsI = 0;
        partitionsI < hMatPartitionCount[0];
        partitionsI++)
    {
        Vector2ui partitionRange = Vector2ui(hMatPartitionOffsets[partitionsI],
                                             hMatPartitionOffsets[partitionsI + 1]); ;
        uint32_t partitionSize = partitionRange[1] - partitionRange[0];
        // Normally we launch a different kernel using
        // "mpo.hPartitionKeys". Here we launch the same kernel

        Span<CommonIndex> dIndicesLocal = dMatPartitionIndices.subspan(partitionRange[0],
                                                                       partitionRange[1]);
        Span<CommonIndex> dKeysLocal = dMatPartitionKeys.subspan(partitionRange[0],
                                                                 partitionRange[1]);

        queue.IssueLambda
        (
            KernelIssueParams{.workCount = partitionSize},
            [=]MRAY_HYBRID(const KernelCallParams kp)
            {
                for(uint32_t i = kp.GlobalId(); i < deadRayCount;
                    i += kp.TotalSize())
                {
                    uint32_t index = dIndicesLocal[i];
                    // Kill the rays!
                    dIsRayAliveBuffer[index] = false;
                }
            }
        );
    }

    // Copy back and check
    hRayStatus = std::vector<uint8_t>(RayCount, true);
    hRayStatusSpan = Span<uint8_t>(hRayStatus.begin(), hRayStatus.end());
    queue.MemcpyAsync(hRayStatusSpan, ToConstSpan(dIsRayAliveBuffer));
    queue.Barrier().Wait();
    for(uint8_t b : hRayStatus)
    {
        EXPECT_EQ(b, false);
    }
}

TEST(RayPartitionerTest, SimulateBasicPathTracer)
{
    SimulateBasicPathTracer();
}

TEST(RayPartitionerTest, SimulateAdvancedPathTracer)
{
    // =============== //
    //  State Killers  //
    // =============== //
    // Cull entire "partition wrt. medium" section
    // and binary partition of "medium-scattered medium-transmitted" section.
    // This means all rays reached to a surface
    static constexpr bool OnlyVacuumMedium = false;
    // Surfaces are only reflective/refractive/transmissive
    // (transmissive surfaces do not have BSSRDF heuristic)
    // BSSRDF surfaces sample a position (p_i) and this needs to be projected
    // to the surface itself. This flag skips the "bssrdf non-bssrdf" binary
    // partitioning.
    static constexpr bool NoBSSRDFSurface = true;
    // Surface partitioning can not be culled
    // Skips the shadow ray casting, on partitioning perspective it disables
    // entire shadow ray partitioning loop
    static constexpr bool SkipNEE = true;
    // Skips multiple medium punchthrough operation. If "OnlyVacuumMedium" is true,
    // this flag has no effect. If scene consists nested participating media this should be
    // set to true.
    static constexpr bool DoMultipleMediumPunchthroughs = true;


    // Check partition stability (is this a stable partition)
    //....

    // Reset back the buffer (we filled)
    // DeviceAlgorithms::Iota();


    // Hit-rays (mock host random rays)
    //std::vector<WorkKey>

    // First handle medium
    // Partition wrt. medium
    //partitioner.


    // Assume some of the rays are scattered
    // Binary partition scattered-ray / hit-ray

    // Continue with non-scattered rays
    // Partition hit-rays wrt. surface

    // Call surface scattering kernels...

    // Some of these were BSSRDF
    // Binary partition p_o !== p_i

    // BSSRDF rays goes to ray casting again (to determine p_i)


    // All rays are completed!

    // Repurpose partitioner for shadow rays

    // while(aliveShadowRays)
    //  Hit shadow rays
    //  Partition wrt. medium
    //  punchthrough medium
    //      If ray could not punchthrough (kill)
    //  Partition wrt. surface
    //      If medium transition surface continue
    //      If requested light (surface accumulate and kill ray)
    //      If any other surface (kill ray)

    // And all done!
}