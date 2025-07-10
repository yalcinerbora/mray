#pragma once
// IWYU pragma: private; include "AlgReduce.h"

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCPU.h"

namespace mray::host::algorithms
{

template <class T>
MRAY_HOST inline
size_t ReduceTMSize(size_t, const GPUQueueCPU&)
{
    return 0u;
}

template <class OutT, class InT>
MRAY_HOST inline
size_t TransformReduceTMSize(size_t, const GPUQueueCPU&)
{
    return 0u;
}

template <class OutT, class InT>
MRAY_HOST inline
size_t SegmentedTransformReduceTMSize(size_t, const GPUQueueCPU&)
{
    return 0u;
}

template <class T, class BinaryOp>
MRAY_HOST inline
void Reduce(Span<T, 1> dReducedValue,
            Span<Byte>,
            Span<const T> dValues,
            const T& initialValue,
            const GPUQueueCPU& queue,
            BinaryOp&& op)
{
    using namespace std::string_view_literals;

    uint32_t elemCount = static_cast<uint32_t>(dValues.size());
    queue.IssueBlockLambda
    (
        "KCReduce-SetInitValue"sv,
        DeviceBlockIssueParams{.gridSize = 1u, .blockSize = 1u},
        [dReducedValue, initialValue](KernelCallParams)
        {
            dReducedValue[0] = initialValue;
        }
    );
    queue.IssueWorkLambda
    (
        "KCReduce-Actual"sv,
        DeviceWorkIssueParams{.workCount = elemCount},
        [=](KernelCallParams kp)
        {
            MRAY_SHARED_MEMORY T local;

            if(kp.GlobalId() < elemCount)
            {
                if(kp.threadId == 0)
                    local = dValues[kp.GlobalId()];
                else
                    local = op(local, dValues[kp.GlobalId()]);
            }
            //
            if(kp.threadId == kp.blockSize - 1)
            {

                std::atomic_ref<T> finalRef(dReducedValue[0]);
                T myLocal = local;
                T expected = finalRef;
                T desired;
                do
                {
                    desired = op(myLocal, expected);
                } while(!finalRef.compare_exchange_weak(expected, desired));
            }
        }
    );
}

template <class OutT, class InT, class BinaryOp, class TransformOp>
MRAY_HOST inline
void TransformReduce(Span<OutT, 1> dReducedValue,
                     Span<Byte>,
                     Span<const InT> dValues,
                     const OutT& initialValue,
                     const GPUQueueCPU& queue,
                     BinaryOp&& binaryOp,
                     TransformOp&& transformOp)
{
    using namespace std::string_view_literals;

    uint32_t elemCount = static_cast<uint32_t>(dValues.size());
    queue.IssueBlockLambda
    (
        "KCTransformReduce-SetInitValue"sv,
        DeviceBlockIssueParams{.gridSize = 1u, .blockSize = 1u},
        [dReducedValue, initialValue](KernelCallParams)
        {
            dReducedValue[0] = initialValue;
        }
    );
    queue.IssueWorkLambda
    (
        "KCTransformReduce-Actual"sv,
        DeviceWorkIssueParams{.workCount = elemCount},
        [=](KernelCallParams kp)
        {
            MRAY_SHARED_MEMORY OutT local;
            if(kp.threadId == 0)
                local = transformOp(dValues[kp.GlobalId()]);
            else
                local = binaryOp(local, transformOp(dValues[kp.GlobalId()]));

            if(kp.threadId == kp.blockSize - 1)
            {
                // Do this at the end?????
                std::atomic_ref<OutT> finalRef(dReducedValue[0]);
                OutT expected = finalRef.load();
                OutT desired;
                do
                {
                    desired = op(local, expected);
                } while(!finalRef.compare_exchange_weak(expected, desired));
            }
        }
    );
}

template <class OutT, class InT, class BinaryOp, class TransformOp>
MRAY_HOST inline
void SegmentedTransformReduce(Span<OutT> dReducedValues,
                              Span<Byte>,
                              Span<const InT> dValues,
                              Span<const uint32_t> dSegmentRanges,
                              const OutT& initialValue,
                              const GPUQueueCPU& queue,
                              BinaryOp&& binaryOp,
                              TransformOp&& transformOp)
{
    assert(dSegmentRanges.size() >= 2);
    // Dedicate a block for each segment
    using namespace std::string_view_literals;
    uint32_t segmentCount = static_cast<uint32_t>(dSegmentRanges.size() - 1);
    queue.IssueBlockLambda
    (
        "KCSegmentedTransformReduce"sv,
        DeviceBlockIssueParams
        {
            .gridSize = segmentCount,
            // We are CPU so 1 TPB is fine
            .blockSize = 1u
        },
        [=](KernelCallParams kp)
        {
            assert(kp.threadId == 0);

            // Block-stride loop
            for(uint32_t bId = kp.blockId; bId < segmentCount;
                bId += kp.gridSize)
            {
                OutT local = initialValue;
                uint32_t start = dSegmentRanges[bId];
                uint32_t end = dSegmentRanges[bId + 1u];
                for(uint32_t i = start; i < end; i++)
                {
                    local = binaryOp(local, transformOp(dValues[i]));
                }
                if(end - start != 0) dReducedValues[bId] = local;
            }
        }
    );
}

}