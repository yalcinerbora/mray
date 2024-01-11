#pragma once

//template<class T>
//concept Add
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
        queue.IssueLambda(p, [=] MRAY_HYBRID(KernelCallParams kp)
        {
            for(uint32_t i = kp.GlobalId();
                i < static_cast<uint32_t>(dOut.size());
                i += kp.TotalSize())
            {
                dOut[i] = T(i) + hInitialValue;
            }
        });
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
        queue.IssueLambda(p, [=] MRAY_HYBRID(KernelCallParams kp)
        {
            for(uint32_t i = kp.GlobalId();
                i < static_cast<uint32_t>(dOut.size());
                i += kp.TotalSize())
            {
                dOut[i] = TransFunction(dIn[i]);
            }
        });
    }
}

#ifdef MRAY_GPU_BACKEND_CUDA

    #include "AlgReduceCUDA.h"
    #include "AlgRadixSortCUDA.h"
    #include "AlgBinPartitionCUDA.h"

    namespace DeviceAlgorithms
    {
        inline namespace DeviceSpecific{ using namespace ::mray::cuda::algorithms; }
        inline namespace DeviceAgnostic{ using namespace ::mray::algorithms; }
    }

//#elif defined MRAY_GPU_BACKEND_SYCL
//    // TODO:
//    //#include "GPUSystemSycl.hpp"
#else
    #error Please define a GPU Backend!
#endif