#pragma once
// IWYU pragma: private; include "AlgBinarySearch.h"

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "Core/Span.h"

namespace mray::host::algorithms
{

template <class T>
MR_HF_DECL
size_t LowerBound(Span<const T> range, const T& value)
{
    auto it = std::lower_bound(range.data(), range.data() + range.size(), value);
    return size_t(std::distance(range.data(), it));
}

}