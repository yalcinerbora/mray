#include <gtest/gtest.h>
#include <numeric>

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUMemory.h"
#include "Device/GPUAlgScan.h"

#include "T_AlgTypes.h"

template <class T>
struct Adder
{
    MRAY_HYBRID MRAY_CGPU_INLINE
    T operator()(const T& l, const T& r) const
    {
        return r + l;
    }
};

template<class Value>
void MultiScanTest(const GPUSystem& system)
{
    static constexpr size_t SegmentSize = 97;
    static constexpr size_t ElementCount = SegmentSize * 37;
    DeviceMemory mem({&system.BestDevice()}, 1_MiB, 8_MiB);

    Span<Value> dInputs;
    Span<Value> dOutputs;
    MemAlloc::AllocateMultiData(std::tie(dInputs, dOutputs),
                                mem, {ElementCount, ElementCount});

    std::vector<Value> hInputs(ElementCount, Value(0));
    std::iota(hInputs.begin(), hInputs.end(), Value(0));

    const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);
    queue.MemcpyAsync(dInputs, Span<const Value>(hInputs.begin(), hInputs.end()));
    queue.MemsetAsync(dOutputs, 0xCD);
    DeviceAlgorithms::InclusiveSegmentedScan(dOutputs, ToConstSpan(dInputs),
                                             SegmentSize, Value(0), queue, Adder<Value>());

    std::vector<Value> hResults(ElementCount);
    queue.MemcpyAsync(Span<Value>(hResults.begin(), hResults.end()),
                      ToConstSpan(dOutputs));
    queue.MemsetAsync(dOutputs, 0xCD);

    // Do the reduction again with a lambda
    DeviceAlgorithms::InclusiveSegmentedScan(dOutputs, ToConstSpan(dInputs),
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

template<class Value>
void ExclusiveScanTest(const GPUSystem& system)
{
    const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);

    static constexpr size_t ElementCount = 1'00;
    DeviceMemory mem({&system.BestDevice()}, 1_MiB, 8_MiB);

    Span<Value> dInputs;
    Span<Value> dOutputs;
    Span<Byte> dTempMem;
    size_t tempMemSize = DeviceAlgorithms::ExclusiveScanTMSize<Value>(ElementCount, queue);
    MemAlloc::AllocateMultiData(std::tie(dInputs, dOutputs, dTempMem),
                                mem, {ElementCount, ElementCount, tempMemSize});

    std::vector<Value> hInputs(ElementCount, Value(0));
    std::iota(hInputs.begin(), hInputs.end(), Value(1));
    queue.MemcpyAsync(dInputs, Span<const Value>(hInputs.begin(), hInputs.end()));
    queue.MemsetAsync(dOutputs, 0xCD);

    DeviceAlgorithms::ExclusiveScan(dOutputs, dTempMem,
                                    ToConstSpan(dInputs),
                                    Value(0), queue, Adder<Value>());

    std::vector<Value> hResults(ElementCount);
    queue.MemcpyAsync(Span<Value>(hResults.begin(), hResults.end()),
                      ToConstSpan(dOutputs));
    queue.MemsetAsync(dOutputs, 0xCD);

    // Do the reduction again with a lambda
    DeviceAlgorithms::ExclusiveScan
    (
        dOutputs, dTempMem,
        ToConstSpan(dInputs),
        Value(0), queue,
        []MRAY_HYBRID(const Value & l, const Value & r)
        {
            return l + r;
        }
    );
    std::vector<Value> hResultsLambda(ElementCount);
    queue.MemcpyAsync(Span<Value>(hResultsLambda.begin(), hResultsLambda.end()),
                      ToConstSpan(dOutputs));
    queue.Barrier().Wait();

    for(size_t i = 0; i < ElementCount; i++)
    {
        Value val = hInputs[i];
        Value result = val * (val - 1) / 2;

        ExpectEqualVecOrArithmetic(result, hResults[i]);
        ExpectEqualVecOrArithmetic(result, hResultsLambda[i]);
    }
}

TYPED_TEST(DeviceAlgorithmsTest, MultiScan)
{
    using Value = typename DeviceAlgorithmsTest<TypeParam>::ValueType;
    GPUSystem system;
    MultiScanTest<Value>(system);
}

TYPED_TEST(DeviceAlgorithmsTest, ExclusiveScan)
{
    using Value = typename DeviceAlgorithmsTest<TypeParam>::ValueType;
    GPUSystem system;
    ExclusiveScanTest<Value>(system);
}