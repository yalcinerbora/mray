#include <gtest/gtest.h>
#include "Tracer/Random.h"

#include "Device/GPUSystem.hpp"

using SoASpanTest = SoASpan<uint32_t, uint64_t, Float, Vector2i>;

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCHashStatic(MRAY_GRID_CONSTANT const Span<uint64_t> out,
                  MRAY_GRID_CONSTANT const SoASpanTest data)
{
    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < out.size();
        i += kp.TotalSize())
    {
        uint32_t    a = data.Get<0>()[i];
        uint64_t    b = data.Get<1>()[i];
        Float       c = data.Get<2>()[i];
        Vector2i    d = data.Get<3>()[i];

        using RNGFunctions::HashPCG64::Hash;
        uint64_t hash = 0;
        for(uint32_t _ = 0; _ < 512; _++)
        {
            hash += Hash(a, b, c, d, hash);
        }
        out[i] = hash;
    }
}

TEST(Hash, Constexpr)
{
    // First check if parameter pack expansion is correct
    constexpr int32_t A = -2;
    constexpr Vector2i B = Vector2i(3, 4);
    constexpr uint64_t C = 123;

    using RNGFunctions::HashPCG64::Hash;
    constexpr uint64_t HashPacked = Hash(A, B, C);
    constexpr uint64_t HashHand = Hash(Hash(Hash(std::bit_cast<uint32_t>(A)) +
                                         std::bit_cast<uint64_t>(B)) +
                                    std::bit_cast<uint64_t>(C));
    static_assert(HashPacked == HashHand,
                  "Incorrect parameter pack expansion for"
                  "\"RNGFunctions::HashPCG64\"");

    // Commenting a benchmark here,
    // used to check if the generated code is plausible
    //Span<uint64_t> dOutputs;
    //Span<uint32_t> dInput2A;
    //Span<uint64_t> dInput2B;
    //Span<Float> dInput2C;
    //Span<Vector2i> dInput2D;

    //static constexpr uint32_t TOTAL_THREADS = 32'768 * 4;
    //GPUSystem s;
    //DeviceLocalMemory memo(s.BestDevice());

    //MemAlloc::AllocateMultiData(std::tie(dOutputs,
    //                                     dInput2A, dInput2B,
    //                                     dInput2C, dInput2D),
    //                            memo,
    //                            {TOTAL_THREADS, TOTAL_THREADS,
    //                             TOTAL_THREADS, TOTAL_THREADS,
    //                             TOTAL_THREADS});
    //const GPUQueue& q = s.BestDevice().GetQueue(0);
    //using namespace std::string_view_literals;
    //q.IssueSaturatingKernel<KCHashStatic>
    //(
    //    "KCHashStatic"sv,
    //    KernelIssueParams{.workCount = TOTAL_THREADS},
    //    dOutputs,
    //    SoASpanTest(dInput2A, dInput2B, dInput2C, dInput2D)
    //);
}