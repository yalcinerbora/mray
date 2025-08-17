#pragma once

#include <utility>
#include <cstddef>
#include <cstdint>

namespace TupleDetail
{
    template<class T>
    concept HasSpaceshipOpC = requires(const T & a, const T & b)
    {
        a<=>b;
    };

    template<size_t I, class T>
    struct IndexedBase
    {
        using Type = T;
        static constexpr size_t Index = I;
        //
        T data;

        // Constructors & Destructor
        template <typename... Args>
        requires(sizeof...(Args) != 0)
        explicit constexpr IndexedBase(Args&&... args)
            : data(std::forward<Args>(args)...)
        {}
        //
        constexpr               IndexedBase()                   = default;
        constexpr               IndexedBase(const IndexedBase&) = default;
        constexpr               IndexedBase(IndexedBase&&)      = default;
        constexpr IndexedBase&  operator=(const IndexedBase&)   = default;
        constexpr IndexedBase&  operator=(IndexedBase&&)        = default;
        constexpr               ~IndexedBase()                  = default;
        // NVCC bug?
        // "assertion failed at : "expr.c", line 33498 in determine_defaulted_spaceship_return_type"
        constexpr auto operator<=>(const IndexedBase& right) const requires(HasSpaceshipOpC<Type>)
        {
            return data <=> right.data;
        }
    };

    template<class...>
    struct TupleStorage;

    template<size_t... Is, class... Ts>
    struct TupleStorage<std::index_sequence<Is...>, Ts...>
        : public IndexedBase<Is, Ts>...
    {
        // Constructors & Destructor
        template <typename... Args>
        requires(sizeof...(Args) != 0)
        explicit constexpr TupleStorage(Args&&... args)
            : IndexedBase<Is, Ts>(std::forward<Args>(args))...
        {}
        //
        constexpr               TupleStorage()                      = default;
        constexpr               TupleStorage(const TupleStorage&)   = default;
        constexpr               TupleStorage(TupleStorage&&)        = default;
        constexpr TupleStorage& operator=(const TupleStorage&)      = default;
        constexpr TupleStorage& operator=(TupleStorage&&)           = default;
        constexpr               ~TupleStorage()                     = default;
        //
        constexpr auto          operator<=>(const TupleStorage&) const = default;
    };

}

template<class... Ts>
struct Tuple : public TupleDetail::TupleStorage<std::index_sequence_for<Ts...>, Ts...>
{
    using Storage = TupleDetail::TupleStorage<std::index_sequence_for<Ts...>, Ts...>;
    static constexpr auto TypeCount = sizeof...(Ts);

    // Constructors & Destructor
    template<class... Args>
    requires(sizeof...(Args) != 0)
    explicit constexpr Tuple(Args&& ...ts)
        : Storage(std::forward<Args>(ts)...)
    {}
    //
    constexpr           Tuple()                 = default;
    constexpr           Tuple(const Tuple&)     = default;
    constexpr           Tuple(Tuple&&)          = default;
    constexpr Tuple&    operator=(const Tuple&) = default;
    constexpr Tuple&    operator=(Tuple&&)      = default;
    constexpr           ~Tuple()                = default;
    //
    constexpr auto      operator<=>(const Tuple&) const = default;

};

template<class... Args>
Tuple(Args&&...) -> Tuple<std::remove_cvref_t<Args>...>;

// Specialization of pack element for our tuple
template<size_t, class>
struct TupleElementT;

template<size_t I, class... Ts>
struct TupleElementT<I, Tuple<Ts...>>
{
    // So we can also use the parameter pack expansion inheritance trick
    // for n_th element!
    // https://ldionne.com/2015/11/29/efficient-parameter-pack-indexing/
    // Since we define tuple with the same structure, we can use it to find
    // the Nth element. If no built-in is specified.
    #ifdef MRAY_MSVC
        template<class T>
        static auto NthElementF(TupleDetail::IndexedBase<I, T>&&) -> TupleDetail::IndexedBase<I, T>;
        using Type = decltype(NthElementF(std::declval<Tuple<Ts...>>()))::Type;
    #elif defined(MRAY_CLANG) || defined(MRAY_GCC)
        #if __has_builtin(__type_pack_element)
            using Type = __type_pack_element<I, Ts...>;
        #else
            template<class T>
            static auto NthElementF(TupleDetail::IndexedBase<I, T>&&) -> TupleDetail::IndexedBase<I, T>;
            using Type = decltype(NthElementF(std::declval<Tuple<Ts...>>()))::Type;
        #endif
    #endif
};

template<size_t I, class... Ts>
using TupleElement = typename TupleElementT<I, Ts...>::Type;

namespace TupleDetail
{
    // TODO: Implement const version later?
    template<class... Args, std::size_t... Is>
    constexpr Tuple<Args&...> ToTupleRef(Tuple<Args...>& t,
                                         std::index_sequence<Is...>) noexcept;

    template<typename Func, class... Args, size_t... Is>
    constexpr bool InvokeAt(size_t idx, const Tuple<Args...>& t, Func&& F,
                            std::index_sequence<Is...>);
}

template<class... Ts>
constexpr Tuple<Ts&...> Tie(Ts&... args) noexcept;

template<class... Args, typename Indices = std::index_sequence_for<Args...>>
constexpr Tuple<Args&...> ToTupleRef(Tuple<Args...>& t) noexcept;

// https://stackoverflow.com/questions/28997271/c11-way-to-index-tuple-at-runtime-without-using-switch
// From here, recursive expansion of if(i == I) F(std::get<I>(tuple));
// returning bool here to show out of bounds access (so not function is called)
// Predicate function must return bool to abuse short circuiting.
template<class Func, class... Args>
requires(std::is_same_v<std::invoke_result_t<Func, Args>, bool> && ...)
constexpr bool InvokeAt(uint32_t index, const Tuple<Args...>& t, Func&& F);

template<class... Ts>
constexpr Tuple<Ts&...> Tie(Ts&... args) noexcept;

template<class Func, class TupleT>
constexpr decltype(auto) Apply(TupleT&&, Func&& F);

// We need to specialize these for
// "auto [a, b, c] = TupleReturningFunc(...)"
// syntax (forgot the name). It is "structured binding"
//
// get is specifically c++ naming convention (g is lowercase)
// since it is required for structured binding
template <size_t I, typename ...Ts>
constexpr TupleElement<I, Tuple<Ts...>>&
get(Tuple<Ts...>& t) noexcept
{
    using T = TupleElement<I, Tuple<Ts...>>;
    return static_cast<TupleDetail::IndexedBase<I, T>&>(t).data;
}

template <std::size_t I, typename ...Ts>
constexpr const TupleElement<I, Tuple<Ts...>>&
get(const Tuple<Ts...>& t) noexcept
{
    using T = TupleElement<I, Tuple<Ts...>>;
    return static_cast<const TupleDetail::IndexedBase<I, T>&>(t).data;
}

template <std::size_t I, typename ...Ts>
constexpr TupleElement<I, Tuple<Ts...>>&&
get(Tuple<Ts...>&& t) noexcept
{
    using T = TupleElement<I, Tuple<Ts...>>;
    return static_cast<TupleDetail::IndexedBase<I, T>&&>(t).data;
}

namespace std
{
    template <class... Args>
    struct tuple_size<Tuple<Args...>> : std::integral_constant<size_t, sizeof...(Args)>
    {};

    template <size_t I, class... Ts>
    struct tuple_element<I, Tuple<Ts...>>
    {
        using type = TupleElement<I, Tuple<Ts...>>;
    };
}

template<class... Args, std::size_t... I>
constexpr Tuple<Args&...> TupleDetail::ToTupleRef(Tuple<Args...>& t,
                                                  std::index_sequence<I...>) noexcept
{
    return Tie(get<I>(t)...);
}

template<typename Func, class... Args, size_t... Is>
constexpr bool TupleDetail::InvokeAt(size_t idx, const Tuple<Args...>& t, Func&& F,
                                     std::index_sequence<Is...>)
{
    // Parameter pack expansion (comma separator abuse)
    bool r = false;
    (
        // Abusing short circuit of "&&"
        static_cast<void>(r |= (Is == idx) && F(get<Is>(t))),
        // And expand
        ...
    );
    return r;
}

template<class... Args>
constexpr Tuple<Args&...> ToTupleRef(Tuple<Args...>& t) noexcept
{
    return TupleDetail::ToTupleRef(t, std::index_sequence_for<Args...>{});
}

template<class Func, class... Args>
requires(std::is_same_v<std::invoke_result_t<Func, Args>, bool> && ...)
constexpr bool InvokeAt(uint32_t index, const Tuple<Args...>& t, Func&& F)
{
    return TupleDetail::InvokeAt(index, t, std::forward<Func>(F),
                                 std::index_sequence_for<Args...>{});
}

template<class... Ts>
constexpr Tuple<Ts&...> Tie(Ts&... args) noexcept
{
    return Tuple<Ts&...>(args...);
}

template<class Func, class TupleT>
constexpr decltype(auto) Apply(TupleT&& t, Func&& F)
{
    using T = std::remove_cvref_t<TupleT>;
    auto Impl = [&]<size_t... Is>(std::index_sequence<Is...>)
    {
        return std::invoke(std::forward<Func>(F),
                           get<Is>(std::forward<TupleT>(t))...);
    };
    //
    Impl(std::make_index_sequence<T::TypeCount>{});
}