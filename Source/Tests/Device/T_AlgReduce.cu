#include <gtest/gtest.h>
#include <numeric>
#include <random>

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgReduce.h"

#include "T_AlgTypes.h"

template <class T>
struct Adder
{
    MRAY_HYBRID MRAY_CGPU_INLINE
    T operator()(const T& l, const T& r)
    {
        return r + l;
    }
};

template<class Value>
void ReduceTest(const GPUSystem& system)
{
    static constexpr size_t ElementCount = 1'000;
    size_t tempMemSize = DeviceAlgorithms::ReduceTMSize<Value>(ElementCount);
    DeviceMemory mem({&system.BestDevice()}, 1_MiB, 8_MiB);

    Span<Value> dInputs;
    Span<Value> dOutput;
    Span<Byte> dTempMemory;
    MemAlloc::AllocateMultiData(std::tie(dInputs, dTempMemory, dOutput),
                                mem, {ElementCount, tempMemSize, 1});

    std::vector<Value> hInputs(ElementCount, Value(0));
    std::iota(hInputs.begin(), hInputs.end(), Value(0));

    const GPUQueue& queue = system.BestDevice().GetQueue(0);
    queue.MemcpyAsync(dInputs, Span<const Value>(hInputs.begin(), hInputs.end()));

    DeviceAlgorithms::Reduce(Span<Value, 1>(dOutput), dTempMemory,
                             ToConstSpan(dInputs.subspan(0, ElementCount)),
                             Value(0), queue, Adder<Value>());

    Value hResult;
    queue.MemcpyAsync(Span<Value>(&hResult, 1), ToConstSpan(dOutput));
    queue.MemsetAsync(dOutput, 0x00);

    // Do the reduction again with a lambda
    DeviceAlgorithms::Reduce(Span<Value, 1>(dOutput), dTempMemory,
                             ToConstSpan(dInputs.subspan(0, ElementCount)),
                             Value(0), queue,
                             []MRAY_HYBRID(const Value& l, const Value& r)
                             {
                                 return l + r;
                             });

    Value hResultLambda;
    queue.MemcpyAsync(Span<Value>(&hResultLambda, 1), ToConstSpan(dOutput));
    queue.Barrier().Wait();

    Value result = (Value(ElementCount) * (Value(ElementCount) - Value(1))) / Value(2);

    ExpectEqualVecOrArithmetic(result, hResult);
    ExpectEqualVecOrArithmetic(result, hResultLambda);
}

TYPED_TEST(DeviceAlorithmsTest, Reduce)
{
    using Value = typename DeviceAlorithmsTest<TypeParam>::ValueType;
    GPUSystem system;
    ReduceTest<Value>(system);
}