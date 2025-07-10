#pragma once


#include <iterator>
#include <vector>
#include <algorithm>

#include "MathForward.h"

namespace Algo
{
    template <std::contiguous_iterator It, class Compare = std::less<typename It::value_type>>
    constexpr
    std::vector<Vector2ul> PartitionRange(It first, It last,
                                          Compare&& cmp = std::less<typename It::value_type>{});
}

template <std::contiguous_iterator It, class Compare>
constexpr
std::vector<Vector2ul> Algo::PartitionRange(It first, It last, Compare&& cmp)
{
    assert(std::is_sorted(first, last, cmp));
    std::vector<Vector2ul> result;
    It start = first;
    while(start != last)
    {
        It end = std::upper_bound(start, last, *start, cmp);
        Vector2ul r(std::distance(first, start),
                    std::distance(first, end));
        result.push_back(r);
        start = end;
    }
    return result;
}