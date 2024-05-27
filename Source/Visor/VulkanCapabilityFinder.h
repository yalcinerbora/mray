#pragma once

#include "Core/DataStructures.h"

//
template <std::contiguous_iterator ItL,
          std::contiguous_iterator ItS,
          class BinaryOp>
bool CheckAllInList(ItL lBegin, ItL lEnd,
                    ItS sBegin, ItS sEnd,
                    BinaryOp&& op)
{
    size_t itemCount = static_cast<size_t>(std::distance(sBegin, sEnd));
    using ValueTypeS = typename std::iterator_traits<ItS>::value_type;
    using ValueTypeL = typename std::iterator_traits<ItL>::value_type;

    uint32_t result = 0;
    for(ItL j = lBegin; j != lEnd; j++)
    {
        const ValueTypeL& item = *j;
        auto loc = std::find_if(sBegin, sEnd,
        [&item, op = std::forward<BinaryOp>(op)](const ValueTypeS& s)
        {
            return op(item, s);
        });
        if(loc != sEnd) result++;
    }
    return result == itemCount;
}