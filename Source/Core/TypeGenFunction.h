#pragma once

#include <memory>

#include "Core/Map.h"
#include "Core/Types.h"

template<class BaseType, class...Args>
using GeneratorFuncType = std::unique_ptr<BaseType>(*)(Args&&... args);

template<class BaseType, class Type, class...Args>
std::unique_ptr<BaseType> GenerateType(Args&&... args)
{
    return std::make_unique<Type>(std::forward<Args>(args)...);
}

template<class GeneratorType, class BaseType,
         class... Args, class... Types>
void GenerateMapping(Map<std::string_view, GeneratorType>& map,
                     // These types are here for overload resolution,
                     // these will not be used directly
                     Tuple<Args...>*, Tuple<Types...>*)
{
    // I've come up with this to move the Args... pack away
    // from expansion of Types...
    // If you have asked me how does it work, I've no idea :)
    auto GenTypeWrapper = [&map]<typename T>()
    {
        map.emplace(T::TypeName(), &GenerateType<BaseType, T, Args...>);
    };

    // Use comma operator expansion
    // Trick from here
    //https://codereview.stackexchange.com/questions/201106/iteration-over-zipped-tuples-for-each-in-tuples
    ((void)GenTypeWrapper.template operator()<Types>(), ...);
}