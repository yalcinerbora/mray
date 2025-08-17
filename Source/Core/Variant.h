#pragma once

// Variant implementation
// It is implemented to reduce compilation
// times (some portion of the std::variant takes
// too long to instantiate (for example, std::conjunction).
// Additionally it has conditional inheritance for trivial/non-trivial
// move copy construct etc. due to non c++20 implementation.
//
// It is still constexpr and still union of union of ... union
// solution. Specifically, we do not do storage-based design
// since NVCC is reluctant to lift byte array to registers
// and amlost always loads it from GMem
// (and such implementation can't be constexpr, but its not
// important for MRay's use-case).
//
// Currently it has quite a bit of warnings on NVCC so its not used.
// We may improve it later and use it on important parts of the
// system (such as textures).
//
// A better, fast to compile and constexpr implementation
// could've been achieved if unions accept inheritance (see Tuple.h).
// Alas, it is not allowed :(

#include <utility>
#include <cstddef>
#include <cstdint>
#include <array>

#include "Definitions.h"
#include "TypePack.h"

#include <variant>

template<class... Types>
class Variant;

namespace VariantDetail
{
    template<uint32_t I>
    using UIntTConst = std::integral_constant<uint32_t, I>;

    template<template<class> class X, class... Args>
    constexpr bool AllTrait(TypePack<Args...>);

    template<size_t C>
    static constexpr size_t IndexByteSize() noexcept
    {
        constexpr bool FITS_CHAR = (C <= std::numeric_limits<uint8_t>::max());
        constexpr bool FITS_SHORT = (C <= std::numeric_limits<uint16_t>::max());
        //
             if constexpr(FITS_CHAR ) return 1;
        else if constexpr(FITS_SHORT) return 2;
        else                          return 3;
    }
    template<size_t ByteCount> struct SelectVariantIndexT;
    template<> struct SelectVariantIndexT<1> { using type = uint8_t; };
    template<> struct SelectVariantIndexT<2> { using type = uint16_t; };
    template<> struct SelectVariantIndexT<3> { using type = uint32_t; };
    template<size_t S>
    using SelectVariantIndex = SelectVariantIndexT<IndexByteSize<S>()>::type;

    template <bool TC, bool TD, uint32_t S, uint32_t E, class TypePack >
    union UnionStorage;

    // Generic Case
    template <bool TC, bool TD, uint32_t S, uint32_t E, class... Args>
    union UnionStorage<TC, TD, S, E, TypePack<Args...>>
    {
        using TP = TypePack<Args...>;
        static constexpr uint32_t MID = S + (E - S) / 2;
        static constexpr bool IS_LEAF_L = (S + 1 == MID);
        static constexpr bool IS_LEAF_R = (E - 1 == MID);
        using LeftT = std::conditional_t
        <
            IS_LEAF_L,
            TypePackElement<S, TP>,
            UnionStorage<TC, TD, S, MID, TP>
        >;
        using RightT = std::conditional_t
        <
            IS_LEAF_R,
            TypePackElement<MID, TP>,
            UnionStorage<TC, TD, MID, E, TP>
        >;

        public:
        LeftT   l;
        RightT  r;
        // TODO: Do we really need move on r-value context?
        constexpr LeftT& Left() &               requires(IS_LEAF_L) { return l; }
        constexpr const LeftT& Left() const&    requires(IS_LEAF_L) { return l; }
        constexpr LeftT&& Left() &&             requires(IS_LEAF_L) { return std::move(l); }
        constexpr RightT& Right() &             requires(IS_LEAF_R) { return r; }
        constexpr const RightT& Right() const&  requires(IS_LEAF_R) { return r; }
        constexpr RightT&& Right() &&           requires(IS_LEAF_R) { return std::move(r); }
        // Constructors & Destructor
        // These can't be defaulted easily, since these will be implicitly called
        // whenever (like actually constructing the Variant.
        // And even if we explicitly define a destructor, destructors of base/members are called anyway).
        // TC (aka. "Trivially Default Constructible") and
        // TD (aka. "Trivially Default Destructible")
        // come from the heavens since we do not want to instantiate these every time
        // (saves couple of seconds according to the CUDA 13 compile time analyzer).
        // Lifetime and logistics will be handled by the "Variant" class.
        constexpr               UnionStorage() noexcept requires(TC) = default;
        constexpr               UnionStorage() noexcept requires(!TC) {}
        constexpr              ~UnionStorage() noexcept requires(TD) = default;
        constexpr              ~UnionStorage() noexcept requires(!TD) {}
        // These can be defaulted since actual variant will manually move etc.
        // Even we state it sdefault it may be deleted.
        constexpr               UnionStorage(const UnionStorage& other) = default;
        constexpr               UnionStorage(UnionStorage&& other)      = default;
        constexpr UnionStorage& operator=(const UnionStorage& other)    = default;
        constexpr UnionStorage& operator=(UnionStorage&& other)         = default;
    };

    template<uint32_t I, uint32_t S, uint32_t E, class StorageT>
    constexpr decltype(auto) MetaGet(StorageT&& s);

    // TODO: Do not rely on variant
    template<size_t I, class... Types>
    using VariantAlternative = TypePackElement<I, TypePack<Types...>>;

    template<class>
    struct VariantSizeV;

    template<class... Types>
    struct VariantSizeV<Variant<Types...>>
        : std::integral_constant<size_t, sizeof...(Types)>
    {};

    template<class R, class T, class... Types>
    constexpr R IndexOfType();

    template<uint32_t O, class VariantT, class Func>
    constexpr auto IfElseVisitImpl(UIntTConst<O>, VariantT&& v, Func&& f) -> decltype(auto);
}

template<class... Types>
class Variant
{
    // Logistics stuff (Lifetime and data movement)
    public:
    using TP = TypePack<Types...>;
    static constexpr bool TC  = VariantDetail::template AllTrait<std::is_trivially_constructible>(TP{});
    static constexpr bool TD  = VariantDetail::template AllTrait<std::is_trivially_destructible >(TP{});
    static constexpr bool TCC = VariantDetail::template AllTrait<std::is_trivially_copy_constructible>(TP{});
    static constexpr bool TMC = VariantDetail::template AllTrait<std::is_trivially_move_constructible>(TP{});
    static constexpr bool TCA = VariantDetail::template AllTrait<std::is_trivially_copy_assignable>(TP{});
    static constexpr bool TMA = VariantDetail::template AllTrait<std::is_trivially_move_assignable>(TP{});

    private:
    // Index type related
    using VariantIndex = VariantDetail::SelectVariantIndex<sizeof...(Types)>;
    static constexpr auto INVALID_INDEX = std::numeric_limits<VariantIndex>::max();
    // Storage type related
    using StorageType = VariantDetail::UnionStorage<TC, TD, 0, sizeof...(Types), TypePack<Types...>>;
    // Is first type default constructible ?
    static constexpr bool FIRST_DC = std::is_default_constructible_v<TypePackElement<0, TP>>;
    // Friends
    template <class T, class... Ts>
    friend constexpr bool HoldsAlternative(const Variant<Ts...>&) noexcept;

    template <uint32_t I, class VariantT>
    friend constexpr decltype(auto) Alternative(VariantT&& v) noexcept;

    private:
    StorageType     storage;
    VariantIndex    tag;

    // Helpers
    template<class T, class... Args>
    void            ConstrcutAlternative(Args&&... args);
    void            DestroyAlternative();

    public:
    template<class T>
    static constexpr auto IndexOfType = VariantDetail::IndexOfType<VariantIndex, T, Types...>();
    static constexpr auto TypeCount = sizeof...(Types);

    // Constructors & Destructor
    constexpr           Variant() noexcept requires(FIRST_DC);
    template<class T>
    constexpr           Variant(T&&) noexcept requires(!std::is_same_v<std::remove_cvref_t<T>, Variant>);
    template<class T>
    constexpr Variant&  operator=(T&&) noexcept;
    // Logistics
    constexpr           Variant(const Variant& other) noexcept   requires(!TCC);
    constexpr           Variant(Variant&& other) noexcept        requires(!TMC);
    constexpr Variant&  operator=(const Variant& other) noexcept requires(!TCA);
    constexpr Variant&  operator=(Variant&& other) noexcept      requires(!TMA);
    // TODO: MSVC Bug? Cant define outside class. It says it is ambiguous
    constexpr           ~Variant() noexcept                      requires(!TD)
    {
        if(tag != INVALID_INDEX) DestroyAlternative();
    }
    //
    constexpr           Variant(const Variant& other) noexcept   requires(TCC) = default;
    constexpr           Variant(Variant&& other) noexcept        requires(TMC) = default;
    constexpr Variant&  operator=(const Variant& other) noexcept requires(TCA) = default;
    constexpr Variant&  operator=(Variant&& other) noexcept      requires(TMA) = default;
    constexpr           ~Variant() noexcept                      requires(TD)  = default;
    //
    constexpr VariantIndex  Index() const { return tag; };
    constexpr VariantIndex  Index() { return tag; };

    constexpr VariantIndex  index() const { return tag; };
    constexpr VariantIndex  index() { return tag; };
};

template<size_t I, class... Types>
using VariantAlternative = VariantDetail::VariantAlternative<I, Types...>;

template<class T>
static constexpr auto VariantSize = VariantDetail::VariantSizeV<T>::value;

// Try to utilize execution unit of the compiler instead of
// instantiation unit. (NVCC reports too many conjunction instantiations)
template<template<class> class X, class... Args>
constexpr bool VariantDetail::AllTrait(TypePack<Args...>)
{
    constexpr auto N = sizeof...(Args);
    constexpr std::array<bool, N> TraitList = {X<Args>::value ...};

    for(size_t i = 0; i < N; i++)
        if(!TraitList[i]) return false;
    return true;
}

template<uint32_t I, uint32_t S, uint32_t E, class StorageT>
constexpr decltype(auto)
VariantDetail::MetaGet(StorageT&& s)
{
    constexpr uint32_t MID = S + (E - S) / 2;
    constexpr bool IS_LEAF_L = (S + 1 == MID && I == S);
    constexpr bool IS_LEAF_R = (E - 1 == MID && I == MID);
    // TODO: Expand it a little to reduce compile times maybe?
    using US = StorageT;
    if constexpr(IS_LEAF_L) return std::forward<US>(s).Left();
    else if constexpr(IS_LEAF_R) return std::forward<US>(s).Right();
    else if constexpr(I < MID)   return MetaGet<I, S, MID>(std::forward<US>(s).l);
    else                         return MetaGet<I, MID, E  >(std::forward<US>(s).r);
}

template<class R, class T, class... Types>
constexpr R VariantDetail::IndexOfType()
{
    R INVALID_INDEX = std::numeric_limits<R>::max();
    constexpr auto N = sizeof...(Types);
    std::array<bool, N> isAvail = {std::is_same_v<T, Types>...};

    R duplicate = 0;
    R tag = INVALID_INDEX;
    for(R i = 0; i < N; i++)
    {
        if(!isAvail[i]) continue;
        duplicate++;
        tag = i;
    }
    if(duplicate != 1 || tag == INVALID_INDEX)
        return INVALID_INDEX;
    return tag;
}

template<uint32_t O, class VariantT, class Func>
constexpr auto VariantDetail::IfElseVisitImpl(UIntTConst<O>, VariantT&& v, Func&& f) -> decltype(auto)
{
    using V = std::remove_cvref_t<VariantT>;
    constexpr uint32_t STAMP_COUNT = 16;
    constexpr uint32_t VSize = uint32_t(VariantSize<V>);
    uint32_t index = uint32_t(v.Index());
    // I dunno how to make this compile time
    // so we check it runtime
    #define COND_INVOKE(I)                                   \
        if constexpr(VSize > (I + O)) if(index == O + I)     \
        {                                                    \
            return std::invoke                               \
            (                                                \
                std::forward<Func>(f),                       \
                Alternative<O + I>(std::forward<VariantT>(v))\
            );                                               \
        }
    // End COND_INVOKE
         COND_INVOKE(0)
    else COND_INVOKE(1)
    else COND_INVOKE(2)
    else COND_INVOKE(3)
    else COND_INVOKE(4)
    else COND_INVOKE(5)
    else COND_INVOKE(6)
    else COND_INVOKE(7)
    else COND_INVOKE(8)
    else COND_INVOKE(9)
    else COND_INVOKE(10)
    else COND_INVOKE(11)
    else COND_INVOKE(12)
    else COND_INVOKE(13)
    else COND_INVOKE(14)
    else COND_INVOKE(15)
    else if constexpr(VSize > O + STAMP_COUNT)
        return IfElseVisitImpl(UIntTConst<O + STAMP_COUNT>{},
                               std::forward<VariantT>(v),
                               std::forward<Func>(f));
    MRAY_UNREACHABLE;

    #undef COND_INVOKE
}

template<class ... Types>
template<class TT, class... Args>
void Variant<Types...>::ConstrcutAlternative(Args&&... args)
{
    // Since we are going to call the visit.
    // check that the tag is pre set.
    assert(tag == IndexOfType<TT>);

    // Skip if this type is trivially default constructible
    // and deduced "ConstrcutAlternative" is called to default construct
    // the alternative
    constexpr auto IsTypeTDC = std::is_trivially_default_constructible_v<TT>;
    if constexpr(IsTypeTDC && sizeof...(Args) == 0)
        return;
    //
    // Or do the if else dance...
    else Visit(*this, [&](auto& t)
    {
        using T = std::remove_cvref_t<decltype(t)>;
        if constexpr(std::is_same_v<TT, T>)
            std::construct_at<T>(&t, std::forward<Args>(args)...);
    });
}

template<class ... Types>
void Variant<Types...>::DestroyAlternative()
{
    assert(tag != INVALID_INDEX);
    // If all the types are trivially destructible just skip
    if constexpr(!TD) return;
    //
    // Or do the if else dance...
    else Visit(*this, [](auto&& t)
    {
        using TIn = decltype(t);
        std::destroy_at(&std::forward<TIn>(t));
    });
}

template<class... Types>
constexpr Variant<Types...>::Variant() noexcept requires(FIRST_DC)
    : tag(VariantIndex(0))
{
    using FirstT = VariantAlternative<0, Types...>;
    constexpr bool FIRST_TDC = std::is_trivially_default_constructible_v<FirstT>;
    if constexpr(FIRST_TDC)
    {
        std::construct_at<FirstT>(&VariantDetail::MetaGet<0, 0, TypeCount>(storage));
    }
}

template<class... Types>
template<class T>
constexpr Variant<Types...>::Variant(T&& other) noexcept requires(!std::is_same_v<std::remove_cvref_t<T>, Variant>)
{
    using TBase = std::remove_cvref_t<T>;
    constexpr auto I = VariantDetail::IndexOfType<VariantIndex, TBase, Types...>();
    static_assert(I != INVALID_INDEX, "Unable to Construct Variant with type T");
    tag = I;
    ConstrcutAlternative<TBase>(std::forward<T>(other));
}

template<class ... Types>
template<class T>
constexpr Variant<Types...>& Variant<Types...>::operator=(T&& t) noexcept
{
    using TBase = std::remove_cvref_t<T>;
    constexpr auto I = VariantDetail::IndexOfType<VariantIndex, T, Types...>();
    static_assert(I != INVALID_INDEX, "Unable to Construct Variant with type T");
    if(tag != INVALID_INDEX) DestroyAlternative();
    tag = I;
    ConstrcutAlternative<TBase>(std::forward<T>(t));
    return *this;
}

template<class ... Types>
constexpr Variant<Types...>::Variant(const Variant& other) noexcept requires(!TCC)
{
    tag = other.tag;
    //
    if(tag != INVALID_INDEX)
    {
        Visit(*this, [&](auto&& left)
        {
            using Left = decltype(left);
            using LeftT = std::remove_cvref_t<Left>;
            std::forward<Left>(left) = Alternative<LeftT>(other);
        });
    }
}

template<class ... Types>
constexpr Variant<Types...>::Variant(Variant&& other) noexcept requires(!TMC)
{
    tag = other.tag;
    if(tag != INVALID_INDEX)
    {
        Visit(*this, [&](auto&& left)
        {
            using Left = decltype(left);
            using LeftT = std::remove_cvref_t<Left>;
            // This probably do not work? (Not guaranteed to be copy since we dereference ptr?)
            std::forward<Left>(left) = std::move(Alternative<LeftT>(std::forward<Variant>(other)));
        });
    }
    // Other is in "moved from" state. Revert back to monostate?
    other.tag = INVALID_INDEX;
}

template<class ... Types>
constexpr Variant<Types...>&
Variant<Types...>::operator=(const Variant& other) noexcept requires(!TCA)
{
    assert(this != &other);
    if(tag != INVALID_INDEX) DestroyAlternative();
    //
    tag = other.tag;
    if(tag != INVALID_INDEX)
    {
        Visit(*this, [&](auto&& left)
        {
            using Left = decltype(left);
            using LeftT = std::remove_cvref_t<Left>;
            std::forward<Left>(left) = *Get<LeftT>(other);
        });
    }
    return *this;
}

template<class ... Types>
constexpr Variant<Types...>&
Variant<Types...>::operator=(Variant&& other) noexcept requires(!TMA)
{
    assert(this != &other);
    if(tag != INVALID_INDEX) DestroyAlternative();
    //
    tag = other.tag;
    if(tag != INVALID_INDEX)
    {
        Visit(*this, [&](auto&& left)
        {
            using Left = decltype(left);
            using LeftT = std::remove_cvref_t<Left>;
            // This probably do not work? (Not guaranteed to be copy since we dereference ptr?)
            std::forward<Left>(left) = std::move(Alternative<LeftT>(std::forward<Variant>(other)));
        });
    }
    // Other is in "moved from" state, or in other term "valueless"
    // state.
    other.tag = INVALID_INDEX;
    return *this;
}

template <class T, class... Types>
constexpr bool HoldsAlternative(const Variant<Types...>& v) noexcept
{
    using VariantIndex = typename Variant<Types...>::VariantIndex;
    return v.tag == Variant<Types...>::template IndexOfType<T>;
}

template <uint32_t I, class VariantT>
constexpr decltype(auto)
Alternative(VariantT&& v) noexcept
{
    using V = std::remove_cvref_t<VariantT>;
    constexpr uint32_t TypeCount = V::TypeCount;
    assert(v.tag == I);
    return VariantDetail::MetaGet<I, 0u, TypeCount>(std::forward<VariantT>(v).storage);
}

template <class T, class VariantT>
constexpr decltype(auto)
Alternative(VariantT&& v) noexcept
{
    using V = std::remove_cvref_t<VariantT>;
    constexpr auto I = V::template IndexOfType<T>;
    return Alternative<I>(std::forward<VariantT>(v));
}

template<class VariantT, class Func>
constexpr auto Visit(VariantT&& v, Func&& f) -> decltype(auto)
{
    using namespace VariantDetail;
    assert(v.Index() < VariantSize<std::remove_cvref_t<VariantT>>);
    return IfElseVisitImpl(UIntTConst<0>{}, std::forward<VariantT>(v), std::forward<Func>(f));
}

// TODO: Overloading std functions is very fragile (can be UB etc.)
// But this is temporary to just swap the type with "std::variant" and "Variant"
// and check compilation times. It will be removed later
namespace std
{
    template <class Visitor, class... Args>
    constexpr auto visit(Visitor&& f, Variant<Args...>& v) -> decltype(auto)
    {
        return Visit(v, std::forward<Visitor>(f));
    }

    template <class Visitor, class... Args>
    constexpr auto visit(Visitor&& f, const Variant<Args...>& v) -> decltype(auto)
    {
        return Visit(v, std::forward<Visitor>(f));
    }

    template<class T, class... Args>
    constexpr bool holds_alternative(const Variant<Args...>& v)
    {
        return HoldsAlternative<T>(v);
    }

    template <uint32_t I, class... Args>
    constexpr VariantAlternative<I, Args...>&
    get(Variant<Args...>& v)
    {
        return Alternative<I>(v);
    }

    template <uint32_t I, class... Args>
    constexpr const VariantAlternative<I, Args...>&
    get(const Variant<Args...>& v)
    {
        return Alternative<I>(v);
    }

    template<class... Args>
    struct variant_size<Variant<Args...>>
        : public integral_constant<size_t, sizeof...(Args)>
    {};
}
