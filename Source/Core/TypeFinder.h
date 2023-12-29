#pragma once

#include <type_traits>
#include <tuple>
#include <cstddef>

// Compile time recursion on a Tuple of KeyValueType types.
//
// Returns a compile-time value (in this case a function)
// of with given type. This is used couple of times in the MRay
// when pairing material and primitive.
// Primitive can generate multitude of surfaces but material can only
// act on one. Fo example, material has texture(s) and requires a UV surface
// and primitive has UV coordinates and provides a surface generator function
// that generate differential surface using hit coordinates (barycentrics for
// triangle) primitiveId and transformId.
//
// Similarly, a primitive can have multiple transform appliers (identity, rigid,
// skinned mesh etc.) and another function can request the application function.
// This example is for optix accelerator, or again the surface. Generated surface
//
//
//
//
// I'm not good at template metaprogramming
// probably this can be cleaner but it works...
namespace TypeFinder
{
    template<class KeyT, class ReturnT, auto RT>
    struct KeyFuncT
    {
        using KeyType = KeyT;
        using ReturnType = ReturnT;
        static constexpr auto Function = RT;
    };

    namespace Detail
    {
        template<class CheckType, class Tuple, size_t I>
        requires (I == std::tuple_size_v<Tuple>)
        constexpr auto LoopAndFind(Tuple&&)
        {
            static_assert(I != std::tuple_size_v<Tuple>, "Unable to find type in tuple");
        }

        template<class CheckType, class Tuple, size_t I>
        requires (I < std::tuple_size_v<Tuple>)
        constexpr auto LoopAndFind(Tuple&& t) -> std::tuple_element_t<I, Tuple>
        {
            using KeyType = typename std::tuple_element_t<I, Tuple>::KeyType;
            // Check if the key type is same
            if constexpr(std::is_same_v<KeyType, CheckType>)
            {
                return std::tuple_element_t<I, Tuple>{};
            }
            else
                return LoopAndFind<CheckType, Tuple, I + 1>(std::forward<Tuple>(t));
        }
    }

    // Finds the function in given tuple
    template<class CheckType, class Tuple>
    constexpr auto GetFunctionTuple(Tuple&& tuple)
    {
        using namespace Detail;
        return LoopAndFind<CheckType, Tuple, 0>(std::forward<Tuple>(tuple));
    }

    // Finds the function in given variadic template parameters
    template<class Surface, class... KVTypes>
    constexpr auto GetFunctionVariadic()
    {
        constexpr std::tuple<KVTypes...> GenFuncList = std::make_tuple(KVTypes{}...);
        return GetFunctionTuple<Surface, decltype(GenFuncList)>(std::move(GenFuncList));
    }
}
