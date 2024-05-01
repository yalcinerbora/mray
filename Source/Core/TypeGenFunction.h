#pragma once

#include <memory>

template<class BaseType, class...Args>
using GeneratorFuncType = std::unique_ptr<BaseType>(*)(Args&&... args);

template<class BaseType, class Type, class...Args>
std::unique_ptr<BaseType> GenerateType(Args&&... args)
{
    return std::make_unique<Type>(std::forward<Args>(args)...);
}