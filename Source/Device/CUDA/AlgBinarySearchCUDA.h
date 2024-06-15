#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCUDA.h"

#include <cub/thread/thread_search.cuh>

namespace mray::cuda::algorithms
{

template <class T>
MRAY_HYBRID
size_t LowerBound(Span<const T>, const T& value);

}

namespace mray::cuda::algorithms
{

template <class T>
MRAY_HYBRID MRAY_CGPU_INLINE
size_t LowerBound(Span<const T> range, const T& value)
{
    return cub::LowerBound(range.data(), range.size(), value);
}

}