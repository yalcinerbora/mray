#include <gtest/gtest.h>
#include <numeric>
#include <random>

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgorithms.h"

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

// Temporarily define a increment for iota
template <int D, class T>
Vector<D, T> operator++(Vector<D, T>& a)
{
    for(int i = 0; i < D; i++)
        a[i] += 1;
    return a;
}

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

    Span<Value> dOutputsExact = dOutputs.subspan(0, ElementCount);
    Span<Value> dInputsExact = dInputs.subspan(0, ElementCount);

    std::vector<Value> hInputs(ElementCount, Value(0));
    std::iota(hInputs.begin(), hInputs.end(), Value(0));

    const GPUQueue& queue = system.BestDevice().GetQueue(0);
    queue.MemcpyAsync(dInputsExact, Span<const Value>(hInputs.begin(), hInputs.end()));

    DeviceAlgorithms::InclusiveMultiScan(dOutputsExact, ToConstSpan(dInputsExact),
                                         SegmentSize, Value(0), queue, Adder<Value>());

    std::vector<Value> hResults(ElementCount);
    queue.MemcpyAsync(Span<Value>(hResults.begin(), hResults.end()),
                      ToConstSpan(dOutputsExact));

    DeviceDebug::DumpGPUMemToFile("multiScanFreeF", ToConstSpan(dOutputsExact),
                                  queue);

    queue.MemsetAsync(dOutputsExact, 0x00);

    // Do the reduction again with a lambda
    DeviceAlgorithms::InclusiveMultiScan(dOutputsExact, ToConstSpan(dInputsExact),
                                         SegmentSize, Value(0), queue,
                                         []MRAY_HYBRID(const Value & l, const Value & r)
    {
        return l + r;
    });

    std::vector<Value> hResultsLambda(ElementCount);
    queue.MemcpyAsync(Span<Value>(hResultsLambda.begin(), hResultsLambda.end()),
                      ToConstSpan(dOutputsExact));
    queue.Barrier().Wait();

    DeviceDebug::DumpGPUMemToFile("multiScanLambda", ToConstSpan(dOutputsExact),
                                  queue);

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