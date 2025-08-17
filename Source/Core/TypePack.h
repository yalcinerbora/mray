#pragma once

#include<cstddef>

// Simple type package, usefull while metaprograming
template<class... T> struct TypePack {};

namespace TypePackDetail
{
    // https://ldionne.com/2015/11/29/efficient-parameter-pack-indexing/
    // Basic Recursive Impl
    template <size_t I, typename T, typename ...Ts>
    struct FindLinearT
    {
        using Type = typename FindLinearT<I - 1, Ts...>::Type;
    };
    template <typename T, typename ...Ts>
    struct FindLinearT<0, T, Ts...> { using Type = T; };

    // Overload Resolution trick (NVCC ctadvisor likes this better)
    template<class...>
    struct FindViaOverloadT;

    template<size_t I, class T>
    struct OverloadTag { using Type = T; };

    template<size_t... Is, class... Ts>
    struct FindViaOverloadT<std::index_sequence<Is...>, Ts...>
        : public OverloadTag <Is, Ts> ...
    {};

    template<size_t I, class T>
    static auto Finder(OverloadTag<I, T>&&) -> OverloadTag<I, T>;
}

// ======================== //
//    Type Pack Element     //
// ======================== //
template<size_t, class>
struct TypePackElementT;

template<size_t I, class... Ts>
struct TypePackElementT<I, TypePack<Ts...>>
{
    #ifdef MRAY_MSVC
        //using Type = TypePackDetail::FindLinearT<I, Ts...>::Type;

        using  TF = TypePackDetail::FindViaOverloadT<std::index_sequence_for<Ts...>, Ts...>;
        using Type = decltype(TypePackDetail::Finder<I>(std::declval<TF>()))::Type;

    #elif defined(MRAY_CLANG) || defined(MRAY_GCC)
        #if __has_builtin(__type_pack_element)
            using Type = __type_pack_element<I, Ts...>;
        #else
            using Type = PackElementDetail::TypePackElementT<I, Ts...>::Type;
        #endif
    #endif
};

template<size_t I, class T>
using TypePackElement = typename TypePackElementT<I, T>::Type;

// ======================== //
//     Type Pack Size       //
// ======================== //
template<class>
struct TypePackSizeV;

template<class... Ts>
struct TypePackSizeV<TypePack<Ts...>>
    : std::integral_constant<size_t, sizeof...(Ts)>
{};

template<class T>
constexpr size_t TypePackSize = TypePackSizeV<T>::value;