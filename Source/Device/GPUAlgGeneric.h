#pragma once

#include <concepts>
#include "GPUSystem.h"
#include "GPUSystem.hpp"

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

    template <class T, class UnaryFunction>
    requires requires(UnaryFunction f, T x) { {f(x)} -> std::convertible_to<T>; }
    void Transform(Span<T> dOut, Span<const T> dIn, const GPUQueue& queue, UnaryFunction&&);

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
        KernelIssueParams p
        {
            .workCount = static_cast<uint32_t>(dOut.size()),
            .sharedMemSize = 0
        };
        queue.IssueLambda
        (
            "KCIota", p,
            [=] MRAY_HYBRID(KernelCallParams kp)
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

        KernelExactIssueParams p
        {
            .gridSize = maxBlocks,
            .blockSize = StaticThreadPerBlock1D()
        };
        queue.IssueExactLambda
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
        uint32_t workCount = static_cast<uint32_t>(dOut.size() - 1);
        KernelIssueParams p
        {
            .workCount = workCount,
            .sharedMemSize = 0
        };
        queue.IssueLambda
        (
            "KCAdjacentDifference", p,
            [=] MRAY_HYBRID(KernelCallParams kp)
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

    template <class T, class UnaryFunction>
    requires requires(UnaryFunction f, T x) {{f(x)} -> std::convertible_to<T>; }
    void Transform(Span<T> dOut, Span<const T> dIn,
                   const GPUQueue& queue,
                   UnaryFunction&& TransFunction)
    {
        assert(dOut.size() == dIn.size());
        KernelIssueParams p
        {
            .workCount = static_cast<uint32_t>(dOut.size()),
            .sharedMemSize = 0
        };
        queue.IssueLambda
        (
            "KCTransform", p,
            [=] MRAY_HYBRID(KernelCallParams kp)
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
    requires requires(UnaryFunction f, T x) { {f(x)} -> std::convertible_to<T>; }
    void InPlaceTransform(Span<T> dInOut, const GPUQueue& queue,
                          UnaryFunction&& TransFunction)
    {
        KernelIssueParams p
        {
            .workCount = static_cast<uint32_t>(dInOut.size()),
            .sharedMemSize = 0
        };
        queue.IssueLambda
        (
            "KCInPlaceTransform", p,
            [=] MRAY_HYBRID(KernelCallParams kp)
            {
                for(uint32_t i = kp.GlobalId();
                    i < static_cast<uint32_t>(dInOut.size());
                    i += kp.TotalSize())
                {
                    T tReg = dInOut[i];
                    dInOut[i] = TransFunction(tReg);
                }
            }
        );
    }

    template <class T, class UnaryFunction>
        requires requires(UnaryFunction f, T x) { { f(x) } -> std::convertible_to<T>; }
    void InPlaceTransformIndirect(Span<T> dInOut, Span<const uint32_t> dIndices,
                                  const GPUQueue& queue, UnaryFunction&& TransFunction)
    {
        KernelIssueParams p
        {
            .workCount = static_cast<uint32_t>(dIndices.size()),
            .sharedMemSize = 0
        };
        queue.IssueLambda
        (
            "KCInPlaceTransformIndirect", p,
            [=] MRAY_HYBRID(KernelCallParams kp)
            {
                for(uint32_t i = kp.GlobalId();
                    i < static_cast<uint32_t>(dIndices.size());
                    i += kp.TotalSize())
                {
                    uint32_t index = dIndices[i];
                    T tReg = dInOut[index];
                    dInOut[index] = TransFunction(tReg);
                }
            }
        );
    }
}

namespace DeviceAlgorithms
{
    inline namespace DeviceAgnostic{ using namespace ::mray::algorithms; }
}