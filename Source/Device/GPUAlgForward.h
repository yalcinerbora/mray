#pragma once

#include <cstdint>
#include <type_traits>

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
    void Transform(Span<OutT> dOut, Span<const InT> dIn,
                   const GPUQueue& queue,
                   UnaryFunction&&);

    template <class T, class UnaryFunction>
    requires requires(UnaryFunction f, T& x) { {f(x)} -> std::same_as<void>; }
    void InPlaceTransform(Span<T> dInOut, const GPUQueue& queue, UnaryFunction&&);

    template <class T, class UnaryFunction>
    requires requires(UnaryFunction f, T& x) { { f(x) } -> std::same_as<void>; }
    void InPlaceTransformIndirect(Span<T> dInOut, Span<const uint32_t> dIndices,
                                  const GPUQueue& queue, UnaryFunction&&);

}

#ifdef MRAY_GPU_BACKEND_CUDA

    #include "CUDA/AlgForwardCUDA.h" // IWYU pragma: export

    namespace DeviceAlgorithms
    {
        inline namespace DeviceSpecific{ using namespace ::mray::cuda::algorithms; }
        inline namespace DeviceAgnostic{ using namespace ::mray::algorithms; }
    }
#elif defined MRAY_GPU_BACKEND_HIP

    #include "HIP/AlgForwardHIP.h" // IWYU pragma: export

    namespace DeviceAlgorithms
    {
        inline namespace DeviceSpecific{ using namespace ::mray::hip::algorithms; }
        inline namespace DeviceAgnostic{ using namespace ::mray::algorithms; }
    }
#elif defined MRAY_GPU_BACKEND_CPU
    #include "CPU/AlgForwardCPU.h" // IWYU pragma: export

    namespace DeviceAlgorithms
    {
        inline namespace DeviceSpecific{ using namespace ::mray::host::algorithms; }
        inline namespace DeviceAgnostic{ using namespace ::mray::algorithms; }
    }
#else
    #error Please define a GPU Backend!
#endif