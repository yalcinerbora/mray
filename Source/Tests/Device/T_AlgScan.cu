#include <gtest/gtest.h>
#include <numeric>
#include <random>

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgScan.h"

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
void MultiScanTest(const GPUSystem& system)
{
    static constexpr size_t SegmentSize = 96;
    static constexpr size_t ElementCount = SegmentSize * 32;
    DeviceMemory mem({&system.BestDevice()}, 1_MiB, 8_MiB);

    Span<Value> dInputs;
    Span<Value> dOutputs;
    MemAlloc::AllocateMultiData(std::tie(dInputs, dOutputs),
                                mem, {ElementCount, ElementCount});

    std::vector<Value> hInputs(ElementCount, Value(0));
    std::iota(hInputs.begin(), hInputs.end(), Value(0));

    const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);
    queue.MemcpyAsync(dInputs, Span<const Value>(hInputs.begin(), hInputs.end()));

    DeviceAlgorithms::InclusiveMultiScan(dOutputs, ToConstSpan(dInputs),
                                         SegmentSize, Value(0), queue, Adder<Value>());

    std::vector<Value> hResults(ElementCount);
    queue.MemcpyAsync(Span<Value>(hResults.begin(), hResults.end()),
                      ToConstSpan(dOutputs));
    queue.MemsetAsync(dOutputs, 0x00);

    // Do the reduction again with a lambda
    DeviceAlgorithms::InclusiveMultiScan(dOutputs, ToConstSpan(dInputs),
                                         SegmentSize, Value(0), queue,
                                         []MRAY_HYBRID(const Value & l, const Value & r)
    {
        return l + r;
    });

    std::vector<Value> hResultsLambda(ElementCount);
    queue.MemcpyAsync(Span<Value>(hResultsLambda.begin(), hResultsLambda.end()),
                      ToConstSpan(dOutputs));
    queue.Barrier().Wait();

    for(size_t i = 0; i < ElementCount; i++)
    {
        size_t segment = i / SegmentSize;
        size_t prevSegmentStart = (segment == 0) ? 0 : (segment * SegmentSize - 1);

        Value val = hInputs[i];
        Value prevVal = hInputs[prevSegmentStart];

        Value result = val * (val + 1) / 2;
        result -= prevVal * (prevVal + 1) / 2;

        ExpectEqualVecOrArithmetic(result, hResults[i]);
        ExpectEqualVecOrArithmetic(result, hResultsLambda[i]);

    }
}

TYPED_TEST(DeviceAlorithmsTest, MultiScan)
{
    using Value = typename DeviceAlorithmsTest<TypeParam>::ValueType;
    GPUSystem system;
    MultiScanTest<Value>(system);
}