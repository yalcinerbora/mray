#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemHIP.h"

#include <rocprim/thread/thread_search.hpp>

namespace mray::hip::algorithms
{

template <class T>
MR_HF_DECL
size_t LowerBound(Span<const T> range, const T& value)
{
    return rocprim::lower_bound(range.data(), range.size(), value);
}

}