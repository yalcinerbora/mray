#pragma once

#include "GenericGroup.h"
#include "Core/MemAlloc.h"
#include "Core/Log.h"

template<class ID, class AI>
template <class... Args>
void GenericGroupT<ID, AI>::GenericCommit(Tuple<Span<Args>&...> output,
                                          std::array<int32_t, sizeof...(Args)> countLookup)
{
    constexpr size_t TypeCount = sizeof...(Args);
    if(isCommitted)
    {
        MRAY_WARNING_LOG("{:s}:{:d}: is in committed state, "
                         "you cannot re-commit!", this->Name(), groupId);
        return;
    }
    // Cactuate offsets
    std::array<size_t, TypeCount> offsets = {0ull};
    for(const auto& c : itemCounts)
    {
        const auto& counts = c.second;
        AttributeRanges range;
        for(size_t i = 0; i < TypeCount; i++)
        {
            size_t count = (countLookup[i] == -1)
                    ? 1
                    : counts[static_cast<uint32_t>(countLookup[i])];
            range.emplace_back(offsets[i], offsets[i] + count);
            offsets[i] += count;
        }

        [[maybe_unused]]
        auto r = itemRanges.emplace(c.first, range);
        assert(r.second);
    }
    // Rename for clarity
    const auto& totalSize = offsets;
    if(std::reduce(totalSize.cbegin(), totalSize.cend()) == 0)
    {
        MRAY_WARNING_LOG("{:s}:{:d}: committing as empty, "
                         "is this correct?", this->Name(), groupId);
        isCommitted = true;
        return;
    }

    using namespace MemAlloc;
    AllocateMultiData<DeviceMemory, Args...>(output, deviceMem, totalSize);
    isCommitted = true;
}
