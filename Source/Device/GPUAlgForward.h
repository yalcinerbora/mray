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

    #define MRAY_DEVICE_ALGO_NAMESPACE ::mray::cuda::algorithms

    namespace DeviceAlgorithms
    {
        inline namespace DeviceSpecific{ using namespace ::mray::cuda::algorithms; }
        inline namespace DeviceAgnostic{ using namespace ::mray::algorithms; }
    }
#elif defined MRAY_GPU_BACKEND_HIP

    #include "HIP/AlgForwardHIP.h" // IWYU pragma: export

    #define MRAY_DEVICE_ALGO_NAMESPACE ::mray::hip::algorithms

    namespace DeviceAlgorithms
    {
        inline namespace DeviceSpecific{ using namespace ::mray::hip::algorithms; }
        inline namespace DeviceAgnostic{ using namespace ::mray::algorithms; }
    }
#elif defined MRAY_GPU_BACKEND_CPU
    #include "CPU/AlgForwardCPU.h" // IWYU pragma: export

    #define MRAY_DEVICE_ALGO_NAMESPACE ::mray::host::algorithms

    namespace DeviceAlgorithms
    {
        inline namespace DeviceSpecific{ using namespace ::mray::host::algorithms; }
        inline namespace DeviceAgnostic{ using namespace ::mray::algorithms; }
    }
#else
    #error Please define a GPU Backend!
#endif

// TODO: Add as needed
#define MRAY_DEVICE_ALGO_BINARY_PARTITION_TM_SIZE_SIGNATURE(T) \
    size_t (MRAY_DEVICE_ALGO_NAMESPACE::BinPartitionTMSize<T>) \
    (                                                          \
        size_t, const GPUQueue&                                \
    )

#define MRAY_DEVICE_ALGO_BINARY_PARTITION_SIGNATURE(T, F)      \
    void (MRAY_DEVICE_ALGO_NAMESPACE::BinaryPartition<T, F>)   \
    (                                                          \
        Span<T>, Span<uint32_t, 1>, Span<Byte>, Span<const T>, \
        const GPUQueue&, F&&                                   \
    )

#define MRAY_DEVICE_ALGO_SEG_TRANSFORM_REDUCE_TM_SIZE_SIGNATURE(T0, T1)         \
    size_t (MRAY_DEVICE_ALGO_NAMESPACE::SegmentedTransformReduceTMSize<T0, T1>) \
    (                                                                           \
        size_t, const GPUQueue&                                                 \
    )

#define MRAY_DEVICE_ALGO_SEG_TRANSFORM_REDUCE_SIGNATURE(T0, T1, F0, F1)         \
    void (MRAY_DEVICE_ALGO_NAMESPACE::SegmentedTransformReduce<T0, T1, F0, F1>) \
    (                                                                           \
        Span<T0>,                                                               \
        Span<Byte>,                                                             \
        Span<const T0>,                                                         \
        Span<const uint32_t>,                                                   \
        const T0&,                                                              \
        const GPUQueue&,                                                        \
        F0&&,                                                                   \
        F1&&                                                                    \
    )

#define MRAY_DEVICE_ALGO_SEG_RADIX_SORT_TM_SIZE_SIGNATURE(B, T0, T1)         \
    size_t (MRAY_DEVICE_ALGO_NAMESPACE::SegmentedRadixSortTMSize<B, T0, T1>) \
    (                                                                        \
        size_t, size_t, const GPUQueue&                                      \
    )

#define MRAY_DEVICE_ALGO_SEG_RADIX_SORT_SIGNATURE(B, T0, T1)             \
    uint32_t (MRAY_DEVICE_ALGO_NAMESPACE::SegmentedRadixSort<B, T0, T1>) \
    (                                                                    \
        Span<Span<T0>, 2>,                                               \
        Span<Span<T1>, 2>,                                               \
        Span<Byte>,                                                      \
        Span<const uint32_t>,                                            \
        const GPUQueue&,                                                 \
        const Vector2ui&                                                 \
    )
