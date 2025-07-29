#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCUDA.h"

#include <cub/thread/thread_search.cuh>

namespace mray::cuda::algorithms
{

template <class T>
MR_HF_DECL
size_t LowerBound(Span<const T> range, const T& value)
{
    return cub::LowerBound(range.data(), range.size(), value);
}

}