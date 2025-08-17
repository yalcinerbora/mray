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
}
