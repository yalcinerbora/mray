#pragma once

#include <concepts>
#include "GPUSystem.h"

namespace mray::algorithms
{
    template <class T>
    requires (requires(T x) { {x + x} -> std::same_as<T>; }&& std::is_constructible_v<T, uint32_t>)
    void Iota(Span<T> dOut, const T& hInitialValue, const GPUQueue& queue);

    template <class T>
    requires (requires(T x) { { x + x } -> std::same_as<T>; }&& std::is_constructible_v<T, uint32_t>)
    void SegmentedIota(Span<T> dOut, Span<const uint32_t> dSegmentRanges,
                       const T& hInitialValue, const GPUQueue& queue);

    template <class OutT, class InT, class BinaryFunction>
    requires requires(BinaryFunction f, InT x) { { f(x, x) } -> std::convertible_to<OutT>; }
    void AdjacentDifference(Span<OutT> dOut, Span<const InT> dIn, const GPUQueue& queue,
                            BinaryFunction&&);

    template <class OutT, class InT, class UnaryFunction>
    requires requires(UnaryFunction f, InT x) { { f(x) } -> std::convertible_to<OutT>; }
    void Transform(Span<OutT> dOut, Span<const InT> dIn, const GPUQueue& queue, UnaryFunction&&);

    template <class T, class UnaryFunction>
    requires requires(UnaryFunction f, T x) { {f(x)} -> std::convertible_to<T>; }
    void InPlaceTransform(Span<T> dInOut, const GPUQueue& queue, UnaryFunction&&);

    template <class T, class UnaryFunction>
    requires requires(UnaryFunction f, T x) { { f(x) } -> std::convertible_to<T>; }
    void InPlaceTransformIndirect(Span<T> dInOut, Span<const uint32_t> dIndices,
                                  const GPUQueue& queue, UnaryFunction&&);


}

namespace mray::algorithms
{
    template <class T>
    requires (requires(T x) { {x + x} -> std::same_as<T>; } && std::is_constructible_v<T, uint32_t>)
    void Iota(Span<T> dOut, const T& hInitialValue, const GPUQueue& queue)
    {
        DeviceWorkIssueParams p
        {
            .workCount = static_cast<uint32_t>(dOut.size()),
            .sharedMemSize = 0
        };
        queue.IssueWorkLambda
        (
            "KCIota", p,
            [=] MRAY_GPU(KernelCallParams kp)
            {
                for(uint32_t i = kp.GlobalId();
                    i < static_cast<uint32_t>(dOut.size());
                    i += kp.TotalSize())
                {
                    dOut[i] = T(i) + hInitialValue;
                }
            }
        );
    }

    template <class T>
    requires (requires(T x) { { x + x } -> std::same_as<T>; }&& std::is_constructible_v<T, uint32_t>)
    void SegmentedIota(Span<T> dOut, Span<const uint32_t> dSegmentRanges,
                       const T& hInitialValue, const GPUQueue& queue)
    {
        // TODO: This is a basic streaming algorithm, this may not be optimal
        // for every segment size etc. We fully saturate the device
        // and dedicate "BlockPerSegment" amount of block each segment
        static constexpr uint32_t BlockPerSegment = 16;
        uint32_t maxBlocks = queue.SMCount() * queue.Device()->MaxActiveBlockPerSM();
        uint32_t totalSegments = static_cast<uint32_t>(dSegmentRanges.size() - 1);
        uint32_t totalBlocks = totalSegments * BlockPerSegment;

        DeviceBlockIssueParams p
        {
            .gridSize = maxBlocks,
            .blockSize = StaticThreadPerBlock1D()
        };
        queue.IssueBlockLambda
        (
            "KCSegmentedIota", p,
            [=] MRAY_HYBRID(KernelCallParams kp)
            {
                // Block-stride loop
                for(uint32_t block = kp.blockId; block < totalBlocks;
                    block += kp.gridSize)
                {
                    uint32_t sId = block / BlockPerSegment;
                    uint32_t localBlockId = block % BlockPerSegment;
                    if(sId >= totalSegments) continue;

                    uint32_t elementStart = dSegmentRanges[sId];
                    uint32_t totalElements = dSegmentRanges[sId + 1] - elementStart;
                    auto dOutSpan = dOut.subspan(elementStart, totalElements);

                    uint32_t iStart = kp.threadId + localBlockId * StaticThreadPerBlock1D();
                    uint32_t iJump = StaticThreadPerBlock1D() * BlockPerSegment;
                    // Actual loop
                    for(uint32_t i = iStart; i < totalElements; i += iJump)
                        dOutSpan[i] = T(i) + hInitialValue;
                }
            }
        );
    }

    template <class OutT, class InT, class BinaryFunction>
    requires requires(BinaryFunction f, InT x) { { f(x, x) } -> std::convertible_to<OutT>; }
    void AdjacentDifference(Span<OutT> dOut, Span<const InT> dIn, const GPUQueue& queue,
                            BinaryFunction&& DiffFunction)
    {
        assert(dOut.size() == dIn.size() - 1);
        assert(dIn.size() > 1);
        uint32_t workCount = static_cast<uint32_t>(dIn.size() - 1);
        DeviceWorkIssueParams p
        {
            .workCount = workCount,
            .sharedMemSize = 0
        };
        queue.IssueWorkLambda
        (
            "KCAdjacentDifference", p,
            [=] MRAY_GPU(KernelCallParams kp)
            {
                for(uint32_t i = kp.GlobalId();
                    i < workCount;
                    i += kp.TotalSize())
                {
                    dOut[i] = DiffFunction(dIn[i], dIn[i + 1]);
                }
            }
        );
    }

    template <class OutT, class InT, class UnaryFunction>
    requires requires(UnaryFunction f, InT x) {{f(x)} -> std::convertible_to<OutT>; }
    void Transform(Span<OutT> dOut, Span<const InT> dIn,
                   const GPUQueue& queue,
                   UnaryFunction&& TransFunction)
    {
        assert(dOut.size() == dIn.size());
        DeviceWorkIssueParams p
        {
            .workCount = static_cast<uint32_t>(dOut.size()),
            .sharedMemSize = 0
        };
        queue.IssueWorkLambda
        (
            "KCTransform", p,
            [=] MRAY_GPU(KernelCallParams kp)
            {
                for(uint32_t i = kp.GlobalId();
                    i < static_cast<uint32_t>(dOut.size());
                    i += kp.TotalSize())
                {
                    dOut[i] = TransFunction(dIn[i]);
                }
            }
        );
    }

    template <class T, class UnaryFunction>
    requires requires(UnaryFunction f, T& x) { {f(x)} -> std::same_as<void>; }
    void InPlaceTransform(Span<T> dInOut, const GPUQueue& queue,
                          UnaryFunction&& TransFunction)
    {
        DeviceWorkIssueParams p
        {
            .workCount = static_cast<uint32_t>(dInOut.size()),
            .sharedMemSize = 0
        };
        queue.IssueWorkLambda
        (
            "KCInPlaceTransform", p,
            [=] MRAY_GPU(KernelCallParams kp)
            {
                for(uint32_t i = kp.GlobalId();
                    i < static_cast<uint32_t>(dInOut.size());
                    i += kp.TotalSize())
                {
                    TransFunction(dInOut[i]);
                }
            }
        );
    }

    template <class T, class UnaryFunction>
    requires requires(UnaryFunction f, T& x) { { f(x) } -> std::same_as<void>; }
    void InPlaceTransformIndirect(Span<T> dInOut, Span<const uint32_t> dIndices,
                                  const GPUQueue& queue, UnaryFunction&& TransFunction)
    {
        DeviceWorkIssueParams p
        {
            .workCount = static_cast<uint32_t>(dIndices.size()),
            .sharedMemSize = 0
        };
        queue.IssueWorkLambda
        (
            "KCInPlaceTransformIndirect", p,
            [=] MRAY_GPU (KernelCallParams kp)
            {
                for(uint32_t i = kp.GlobalId();
                    i < static_cast<uint32_t>(dIndices.size());
                    i += kp.TotalSize())
                {
                    uint32_t index = dIndices[i];
                    TransFunction(dInOut[index]);
                }
            }
        );
    }
}

namespace DeviceAlgorithms
{
    inline namespace DeviceAgnostic{ using namespace ::mray::algorithms; }
}