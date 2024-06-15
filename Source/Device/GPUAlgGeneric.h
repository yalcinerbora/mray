#pragma once

#include <concepts>
#include "GPUSystem.h"
#include "GPUSystem.hpp"

namespace mray::algorithms
{
    template <class T>
    requires (requires(T x) { {x + x} -> std::same_as<T>; }&& std::is_constructible_v<T, uint32_t>)
    void Iota(Span<T> dOut, const T& hInitialValue, const GPUQueue& queue);

    template <class T, class UnaryFunction>
    requires requires(UnaryFunction f, T x) { {f(x)} -> std::convertible_to<T>; }
    void Transform(Span<T> dOut, Span<const T> dIn,
                   const GPUQueue& queue,
                   UnaryFunction&&);

    template <class T, class UnaryFunction>
    requires requires(UnaryFunction f, T x) { {f(x)} -> std::convertible_to<T>; }
    void InPlaceTransform(Span<T> dInOut,
                          const GPUQueue& queue,
                          UnaryFunction&&);

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
}

namespace DeviceAlgorithms
{
    inline namespace DeviceAgnostic{ using namespace ::mray::algorithms; }
}