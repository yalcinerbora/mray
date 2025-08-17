#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "GPUSystemCUDA.h"

// TODO: Dropping cub thread search
// since it takes too long to include.
// Also it is basic lower bound implementation anyway,
// so we just implement it
//#include <cub/thread/thread_search.cuh>

namespace mray::cuda::algorithms
{

template <class T>
MR_HF_DECL
size_t LowerBound(Span<const T> range, const T& value)
{
    // Check the include description
    //return cub::LowerBound(range.data(), range.size(), value);

    uint32_t curOffset = 0;
    for(uint32_t i = uint32_t(range.size()); i > 0u;)
    {
        uint32_t mid = i >> 1;
        // Item is on the right
        if(range[curOffset + mid] < value)
        {
            curOffset += (mid + 1);
            i -= (mid + 1);
        }
        // Items is on the left
        else i = mid;
    }
    return curOffset;
}

}