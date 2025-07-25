#include <gtest/gtest.h>
#include <numeric>

#include "Device/GPUSystem.h"
#include "Device/GPUMemory.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgGeneric.h"

#include "T_AlgTypes.h"

template <class T>
struct Transformer
{
    MRAY_HYBRID MRAY_CGPU_INLINE
    T operator()(const T& t) const
    {
        return t * T(2);
    }
};

template<class Value>
void IotaTest(const GPUSystem& system)
{
    const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);
    static constexpr size_t ElementCount = 1'111;
    DeviceMemory mem({&system.BestDevice()}, 1_MiB, 8_MiB);

    Span<Value> dOutputs;
    MemAlloc::AllocateMultiData(std::tie(dOutputs), mem, {ElementCount});

    Value initialValue = Value(0);

    std::vector<Value> hReference(ElementCount, Value(0));
    std::iota(hReference.begin(), hReference.end(), initialValue);

    DeviceAlgorithms::Iota(dOutputs, initialValue, queue);

    std::vector<Value> hOutputs(ElementCount, Value(0));
    queue.MemcpyAsync(Span<Value>(hOutputs.begin(), hOutputs.end()),
                      ToConstSpan(dOutputs));
    queue.Barrier().Wait();

    for(uint32_t i = 0; i < ElementCount; i++)
    {
        ExpectEqualVecOrArithmetic(hReference[i], hOutputs[i]);
    }
}

template<class Value>
void TransformTest(const GPUSystem& system)
{
    const GPUQueue& queue = system.BestDevice().GetComputeQueue(0);
    static constexpr size_t ElementCount = 1'111;
    DeviceMemory mem({&system.BestDevice()}, 1_MiB, 8_MiB);

    Span<Value> dOutputs, dInputs;
    MemAlloc::AllocateMultiData(std::tie(dOutputs, dInputs), mem,
                                {ElementCount, ElementCount});

    Value initialValue = Value(0);

    std::vector<Value> hIota(ElementCount, Value(0));
    std::iota(hIota.begin(), hIota.end(), initialValue);

    queue.MemcpyAsync(dInputs, Span<const Value>(hIota.begin(), hIota.end()));


    DeviceAlgorithms::Transform(dOutputs,
                                ToConstSpan(dInputs),
                                queue,
                                Transformer<Value>());

    std::vector<Value> hOutputs(ElementCount, Value(0));
    queue.MemcpyAsync(Span<Value>(hOutputs.begin(), hOutputs.end()),
                      ToConstSpan(dOutputs));
    queue.Barrier().Wait();

    for(uint32_t i = 0; i < ElementCount; i++)
    {
        Value expected = Transformer<Value>()(hIota[i]);
        ExpectEqualVecOrArithmetic(expected, hOutputs[i]);
    }
}

TYPED_TEST(DeviceAlgorithmsTest, Iota)
{
    using Value = typename DeviceAlgorithmsTest<TypeParam>::ValueType;
    GPUSystem system;
    IotaTest<Value>(system);
}

TYPED_TEST(DeviceAlgorithmsTest, Transform)
{
    using Value = typename DeviceAlgorithmsTest<TypeParam>::ValueType;
    GPUSystem system;
    TransformTest<Value>(system);
}