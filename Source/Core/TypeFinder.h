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
    template <class T>
    concept HasKeyAsTypeC = requires
    {
        typename T::KeyType;
    };

    template<class KeyT, class ReturnT, auto RT>
    struct KeyTFuncPair
    {
        using KeyType = KeyT;
        using ReturnType = ReturnT;
        static constexpr auto Function = RT;
    };

    template <class T>
    concept HasKeyAsValueC = requires
    {
        T::Key;
    };

    template<auto KeyV, class ResultType>
    struct ValKeyTypePair
    {
        static constexpr auto Key = KeyV;
        using Result = ResultType;
    };

    namespace Detail
    {
        template<class CheckType, class Tuple, size_t I>
        requires (I == std::tuple_size_v<Tuple>)
        constexpr size_t LoopAndFind(Tuple&&)
        {
            static_assert(I != std::tuple_size_v<Tuple>, "Unable to find type in tuple");
            return 0;
        }

        template<class CheckType, class Tuple, size_t I>
        requires (I < std::tuple_size_v<Tuple> && HasKeyAsTypeC<std::tuple_element_t<I, Tuple>>)
        constexpr size_t LoopAndFind(Tuple&& t)
        {
            using KeyType = typename std::tuple_element_t<I, Tuple>::KeyType;
            // Check if the key type is same
            if constexpr(std::is_same_v<KeyType, CheckType>)
            {
                return I;
            }
            else
                return LoopAndFind<CheckType, Tuple, I + 1>(std::forward<Tuple>(t));
        }

        template<auto CheckValue, class Tuple, size_t I>
        requires (I == std::tuple_size_v<Tuple>)
        constexpr size_t LoopAndFindV(Tuple&&)
        {
            static_assert(I != std::tuple_size_v<Tuple>, "Unable to find type in tuple");
            return 0;
        }

        template<auto CheckValue, class Tuple, size_t I>
        requires (I < std::tuple_size_v<Tuple>) //&& HasKeyAsValueC<std::tuple_element_t<I, Tuple>>)
        constexpr size_t LoopAndFindV(Tuple&& t)
        {
            constexpr auto Key = std::tuple_element_t<I, Tuple>::Key;
            // Compare using "operator=="
            if constexpr(Key == CheckValue)
            {
                return I;
            }
            else
                return LoopAndFindV<CheckValue, Tuple, I + 1>(std::forward<Tuple>(t));
        }
    }

    // Finds the "CheckedType" in the given tuple
    // returns the tuple element that contains the "CheckType"
    template<class CheckType, class Tuple>
    constexpr auto GetTupleElement(Tuple tuple)
    {
        using namespace Detail;
        return std::tuple_element_t<LoopAndFind<CheckType, Tuple, 0>(std::forward<Tuple>(tuple)), Tuple>{};
    }
    // Same as above, but variadic template version
    template<class CheckType, class... KVTypes>
    constexpr auto GetTupleElementVariadic()
    {
        constexpr std::tuple<KVTypes...> GenFuncList = std::make_tuple(KVTypes{}...);
        return GetTupleElement<CheckType, std::tuple<KVTypes...>>(std::move(GenFuncList));
    }

    // Concrete value version "CheckType" is some form of value
    // (Most of the time it is enum)
    template<auto CheckValue, class Tuple>
    constexpr auto GetTupleElement(Tuple tuple)
    {
        using namespace Detail;
        return std::tuple_element_t<LoopAndFindV<CheckValue, Tuple, 0>(std::forward<Tuple>(tuple)), Tuple>{};
    }
    // Variadic version
    template<auto CheckValue, class... KVTypes>
    constexpr auto GetTupleElementVariadic()
    {
        constexpr std::tuple<KVTypes...> GenFuncList = std::make_tuple(KVTypes{}...);
        return GetTupleElement<CheckValue, std::tuple<KVTypes...>>(std::move(GenFuncList));
    }
}
