#pragma once

#include "GPUSystemForward.h"

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
    void Transform(Span<T> dOut, Span<const T> dIn,
                   const GPUQueue& queue,
                   UnaryFunction&&);

    template <class T, class UnaryFunction>
    requires requires(UnaryFunction f, T x) { {f(x)} -> std::convertible_to<T>; }
    void InPlaceTransform(Span<T> dInOut,
                          const GPUQueue& queue,
                          UnaryFunction&&);

}

#ifdef MRAY_GPU_BACKEND_CUDA

    #include "CUDA/AlgForwardCUDA.h"

    namespace DeviceAlgorithms
    {
        inline namespace DeviceSpecific{ using namespace ::mray::cuda::algorithms; }
        inline namespace DeviceAgnostic{ using namespace ::mray::algorithms; }
    }

#else
    #error Please define a GPU Backend!
#endif