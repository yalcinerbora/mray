#pragma once

// Compile time recursion on a std::tuple of KeyValueType types.
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
// I'm not good at template metaprograming
// probably this can be cleaner but it works...
//
// Old implementation were using recursive function instantiations
// I've found this gem on stack overflow
// Uses in class function overload resolution which is quite fast
// compared to the old implementation
// https://stackoverflow.com/questions/68668956/c-how-to-implement-a-compile-time-mapping-from-types-to-types
namespace TypeFinder
{
    // Enum to Type Mapper
    template<class EnumType>
    struct E_TMapper
    {
        template<EnumType E>
        struct Tag { static constexpr auto Enum = E; };

        template<class R>
        struct ResultTag { using Result = R; };

        template<EnumType E, class T>
        struct ETPair
        {
            static constexpr EnumType Enum = E;
            using Type = T;
        };

        template<typename Pair>
        struct Node
        {
            static auto Get(Tag<Pair::Enum>) -> ResultTag<typename Pair::Type>;
        };

        template<typename... Pairs>
        struct Map : Node<Pairs>...
        {
            // I did not know you can do this
            // public inheritance probably slower?
            using Node<Pairs>::Get...;

            template<EnumType E>
            using Find = typename decltype(Map::Get(Tag<E>{}))::Result;
        };
    };

    // This can be a namespace but making it a class
    // to be consistent with the above type
    // Type to Type Mapper
    struct T_TMapper
    {
        template<class T>
        struct Tag { using Type = T; };

        template<class R>
        struct ResultTag { using Result = R; };

        template<class K, class V>
        struct TTPair { using From = K; using To = V; };

        template<typename Pair>
        struct Node
        {
            static auto Get(Tag<typename Pair::From>) -> ResultTag<typename Pair::To>;
        };

        template<typename... Pairs>
        struct Map : Node<Pairs>...
        {
            // I did not know you can do this
            // public inheritance probably slower?
            using Node<Pairs>::Get...;

            template<class T>
            using Find = typename decltype(Map::Get(Tag<T>{}))::Result;
        };
    };

    struct T_VMapper
    {
        template<class T>
        struct Tag { using Type = T; };

        template<auto R>
        struct ResultTag { static constexpr auto Result = R; };

        template<class K, auto V>
        struct TVPair { using Key = K; static constexpr auto Value = V; };

        template<typename Pair>
        struct Node
        {
            static auto Get(Tag<typename Pair::Key>) -> ResultTag<Pair::Value>;
        };

        template<typename... Pairs>
        struct Map : Node<Pairs>...
        {
            // I did not know you can do this
            // public inheritance probably slower?
            using Node<Pairs>::Get...;

            template<class T>
            static constexpr auto Find = decltype(Map::Get(Tag<T>{}))::Result;
        };
    };

    //template <class T>
    //concept HasKeyAsTypeC = requires
    //{
    //    typename T::KeyType;
    //};

    //template<class KeyT, class ReturnT, auto RT>
    //struct KeyTFuncPair
    //{
    //    using KeyType = KeyT;
    //    using ReturnType = ReturnT;
    //    static constexpr auto Function = RT;
    //};

    //template <class T>
    //concept HasKeyAsValueC = requires
    //{
    //    T::Key;
    //};

    //template<auto KeyV, class ResultType>
    //struct ValKeyTypePair
    //{
    //    static constexpr auto Key = KeyV;
    //    using Result = ResultType;
    //};

    //namespace Detail
    //{
    //    template<class CheckType, class std::tuple, size_t I>
    //    requires (I == std::tuple_size_v<std::tuple>)
    //    constexpr size_t LoopAndFind(std::tuple&&)
    //    {
    //        static_assert(I != std::tuple_size_v<std::tuple>, "Unable to find type in tuple");
    //        return 0;
    //    }

    //    template<class CheckType, class std::tuple, size_t I>
    //    requires (I < std::tuple_size_v<std::tuple> && HasKeyAsTypeC<std::tuple_element_t<I, std::tuple>>)
    //    constexpr size_t LoopAndFind(std::tuple&& t)
    //    {
    //        using KeyType = typename std::tuple_element_t<I, std::tuple>::KeyType;
    //        // Check if the key type is same
    //        if constexpr(std::is_same_v<KeyType, CheckType>)
    //        {
    //            return I;
    //        }
    //        else
    //            return LoopAndFind<CheckType, std::tuple, I + 1>(std::forward<std::tuple>(t));
    //    }

    //    template<auto CheckValue, class std::tuple, size_t I>
    //    requires (I == std::tuple_size_v<std::tuple>)
    //    constexpr size_t LoopAndFindV(std::tuple&&)
    //    {
    //        static_assert(I != std::tuple_size_v<std::tuple>, "Unable to find type in tuple");
    //        return 0;
    //    }

    //    template<auto CheckValue, class std::tuple, size_t I>
    //    requires (I < std::tuple_size_v<std::tuple> && HasKeyAsValueC<std::tuple_element_t<I, std::tuple>>)
    //    constexpr size_t LoopAndFindV(std::tuple&& t)
    //    {
    //        constexpr auto Key = std::tuple_element_t<I, std::tuple>::Key;
    //        // Compare using "operator=="
    //        if constexpr(Key == CheckValue)
    //        {
    //            return I;
    //        }
    //        else
    //            return LoopAndFindV<CheckValue, std::tuple, I + 1>(std::forward<std::tuple>(t));
    //    }
    //}

    //// Finds the "CheckedType" in the given tuple
    //// returns the tuple element that contains the "CheckType"
    //template<class CheckType, class std::tuple>
    //constexpr auto Getstd::tupleElement(std::tuple tuple)
    //{
    //    using namespace Detail;
    //    return std::tuple_element_t<LoopAndFind<CheckType, std::tuple, 0>(std::forward<std::tuple>(tuple)), std::tuple>{};
    //}
    //// Same as above, but variadic template version
    //template<class CheckType, class... KVTypes>
    //constexpr auto Getstd::tupleElementVariadic()
    //{
    //    constexpr std::tuple<KVTypes...> GenFuncList = std::make_tuple(KVTypes{}...);
    //    return Getstd::tupleElement<CheckType, std::tuple<KVTypes...>>(std::move(GenFuncList));
    //}

    //// Concrete value version "CheckType" is some form of value
    //// (Most of the time it is enum)
    //template<auto CheckValue, class std::tuple>
    //constexpr auto Getstd::tupleElement(std::tuple tuple)
    //{
    //    using namespace Detail;
    //    return std::tuple_element_t<LoopAndFindV<CheckValue, std::tuple, 0>(std::forward<std::tuple>(tuple)), std::tuple>{};
    //}
    //// Variadic version
    //template<auto CheckValue, class... KVTypes>
    //constexpr auto Getstd::tupleElementVariadic()
    //{
    //    constexpr std::tuple<KVTypes...> GenFuncList = std::make_tuple(KVTypes{}...);
    //    return Getstd::tupleElement<CheckValue, std::tuple<KVTypes...>>(std::move(GenFuncList));
    //}
}
