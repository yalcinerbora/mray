#include <gtest/gtest.h>
#include <numeric>
#include <random>

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUMemory.h"
#include "Device/GPUAlgBinaryPartition.h"

#include "T_AlgTypes.h"

template <class T>
struct Partitioner
{
    MR_PF_DECL
    bool operator()(const T& t) const noexcept
    {
        return t != T(0);
    }
};

template<class Value>
void BinPartitionTest(const GPUSystem& system)
{
    const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);
    static constexpr size_t ElementCount = 1'111;
    size_t tempMemSize = DeviceAlgorithms::BinPartitionTMSize<Value>(ElementCount, queue);
    DeviceMemory mem({&system.BestDevice()}, 1_MiB, 8_MiB);

    Span<Value> dInputs;
    Span<Value> dOutputs;
    Span<uint32_t> dOffset;
    Span<Byte> dTempMemory;
    MemAlloc::AllocateMultiData(Tie(dInputs, dOutputs, dTempMemory, dOffset),
                                mem, {ElementCount, ElementCount, tempMemSize, 1});

    std::vector<Value> hInputs(ElementCount, Value(0));
    std::iota(hInputs.begin(), hInputs.end(), Value(0));
    std::mt19937 rng(123);
    for(Value& v : hInputs)
    {
        v = (rng() % 2 == 0) ? Value(0) : Value(1);
    }

    queue.MemcpyAsync(dInputs, Span<const Value>(hInputs.begin(), hInputs.end()));

    DeviceAlgorithms::BinaryPartition(dOutputs, Span<uint32_t, 1>(dOffset),
                                      dTempMemory, ToConstSpan(dInputs),
                                      queue, Partitioner<Value>());

    uint32_t hOffsetFunctor;
    std::vector<Value> hOutputsFunctor(ElementCount, Value(0));
    queue.MemcpyAsync(Span<uint32_t>(&hOffsetFunctor, 1), ToConstSpan(dOffset));
    queue.MemcpyAsync(Span<Value>(hOutputsFunctor.begin(), hOutputsFunctor.end()), ToConstSpan(dOutputs));
    queue.MemsetAsync(dOffset, 0x00);
    queue.MemsetAsync(dOutputs, 0x00);

    // Do the reduction again with a lambda
    DeviceAlgorithms::BinaryPartition(dOutputs, Span<uint32_t, 1>(dOffset),
                                      dTempMemory, ToConstSpan(dInputs),
                                      queue, []MRAY_HYBRID(const Value& t) -> bool
                                      {
                                          return t != Value(0);
                                      });
    uint32_t hOffsetLambda;
    std::vector<Value> hOutputsLambda(ElementCount, Value(0));
    queue.MemcpyAsync(Span<uint32_t>(&hOffsetLambda, 1), ToConstSpan(dOffset));
    queue.MemcpyAsync(Span<Value>(hOutputsLambda.begin(), hOutputsLambda.end()), ToConstSpan(dOutputs));
    queue.Barrier().Wait();

    //
    EXPECT_EQ(hOffsetLambda, hOffsetFunctor);
    for(uint32_t i = 0; i < ElementCount; i++)
    {
        Value expected = (i < hOffsetLambda) ? Value(1) : Value(0);
        ExpectEqualVecOrArithmetic(expected, hOutputsFunctor[i]);
        ExpectEqualVecOrArithmetic(expected, hOutputsLambda[i]);
    }
}

TYPED_TEST(DeviceAlgorithmsTest, BinaryPartition)
{
    using Value = typename DeviceAlgorithmsTest<TypeParam>::ValueType;
    GPUSystem system;
    BinPartitionTest<Value>(system);
}