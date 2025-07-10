#pragma once
// IWYU pragma: private; include "AlgBinarySearch.h"

#include "Core/Definitions.h"
#include "Core/Types.h"

namespace mray::host::algorithms
{

template <class T>
MRAY_HYBRID MRAY_CGPU_INLINE
size_t LowerBound(Span<const T> range, const T& value)
{
    auto it = std::lower_bound(range.data(), range.data() + range.size(), value);
    return size_t(std::distance(range.data(), it));
}

}