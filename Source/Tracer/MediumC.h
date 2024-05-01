#pragma once

#include "Device/GPUSystem.h"
#include "Core/TracerI.h"
#include "TracerTypes.h"
#include "GenericGroup.h"
#include <map>

using GenericGroupMediumT = GenericTexturedGroupT<MediumKey, MediumAttributeInfo>;

template<class Child>
class GenericGroupMedium : public GenericGroupMediumT
{
    public:
                        GenericGroupMedium(uint32_t groupId,
                                           const GPUSystem&,
                                           const TextureViewMap&,
                                           size_t allocationGranularity = 2_MiB,
                                           size_t initialReservartionSize = 4_MiB);
    std::string_view    Name() const override;
};

template <class C>
GenericGroupMedium<C>::GenericGroupMedium(uint32_t groupId,
                                          const GPUSystem& sys,
                                          const TextureViewMap& map,
                                          size_t allocationGranularity,
                                          size_t initialReservartionSize)
    : GenericGroupMediumT(groupId, sys, map,
                          allocationGranularity,
                          initialReservartionSize)
{}

template <class C>
std::string_view GenericGroupMedium<C>::Name() const
{
    return C::TypeName();
}