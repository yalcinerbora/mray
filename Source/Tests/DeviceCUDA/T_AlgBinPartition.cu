#include <gtest/gtest.h>
#include <numeric>
#include <random>

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"

#include "T_AlgTypes.h"

template <class T>
struct Partitioner
{
    MRAY_HYBRID MRAY_CGPU_INLINE
    bool operator()(const T& t)
    {
        return t != T(0);
    }
};

// Temporarily define a increment for iota
template <int D, class T>
Vector<D, T> operator++(Vector<D, T>& a)
{
    for(int i = 0; i < D; i++)
        a[i] += 1;
    return a;
}

template<class Value>
void BinPartitionTest(const GPUSystem& system)
{
    static constexpr size_t ElementCount = 1'000;
    size_t tempMemSize = DeviceAlgorithms::BinPartitionTMSize<Value>(ElementCount);
    DeviceMemory mem({&system.BestDevice()}, 1_MiB, 8_MiB);

    Span<Value> dInputs;
    Span<Value> dOutputs;
    Span<uint32_t> dOffset;
    Span<Byte> dTempMemory;
    MemAlloc::AllocateMultiData(std::tie(dInputs, dOutputs, dTempMemory, dOffset),
                                mem, {ElementCount, ElementCount, tempMemSize, 1});

    Span<uint32_t> dOffsetExact = dOffset.subspan(0, 1);
    Span<Value> dOutputsExact = dOutputs.subspan(0, ElementCount);
    Span<Value> dInputsExact = dInputs.subspan(0, ElementCount);

    std::vector<Value> hInputs(ElementCount, Value(0));
    std::iota(hInputs.begin(), hInputs.end(), Value(0));
    std::mt19937 rng(123);
    for(Value& v : hInputs)
    {
        v = (rng() % 2 == 0) ? Value(0) : Value(1);
    }

    const GPUQueue& queue = system.BestDevice().GetQueue(0);
    queue.MemcpyAsync(dInputsExact, Span<const Value>(hInputs.begin(), hInputs.end()));

    DeviceAlgorithms::BinaryPartition(dOutputsExact, Span<uint32_t, 1>(dOffsetExact),
                                      dTempMemory, ToConstSpan(dInputsExact),
                                      queue, Partitioner<Value>());

    uint32_t hOffsetFunctor;
    std::vector<Value> hOutputsFunctor(ElementCount, Value(0));
    queue.MemcpyAsync(Span<uint32_t>(&hOffsetFunctor, 1), ToConstSpan(dOffsetExact));
    queue.MemcpyAsync(Span<Value>(hOutputsFunctor.begin(), hOutputsFunctor.end()), ToConstSpan(dOutputsExact));
    queue.MemsetAsync(dOffsetExact, 0x00);
    queue.MemsetAsync(dOutputsExact, 0x00);

    // Do the reduction again with a lambda
    DeviceAlgorithms::BinaryPartition(dOutputsExact, Span<uint32_t, 1>(dOffsetExact),
                                      dTempMemory, ToConstSpan(dInputsExact),
                                      queue, []MRAY_HYBRID(const Value& t) -> bool
                                      {
                                          return t != Value(0);
                                      });
    uint32_t hOffsetLambda;
    std::vector<Value> hOutputsLambda(ElementCount, Value(0));
    queue.MemcpyAsync(Span<uint32_t>(&hOffsetLambda, 1), ToConstSpan(dOffsetExact));
    queue.MemcpyAsync(Span<Value>(hOutputsLambda.begin(), hOutputsLambda.end()), ToConstSpan(dOutputsExact));
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


TYPED_TEST(DeviceAlorithmsTest, BinaryPartition)
{
    using Value = typename DeviceAlorithmsTest<TypeParam>::ValueType;
    GPUSystem system;
    BinPartitionTest<Value>(system);
}