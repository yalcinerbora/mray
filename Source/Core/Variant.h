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
//
// ==================================================== //
//   DO NOT THRUST AND COPY THIS CODE IT WORKS BUT      //
//   IT IS NOT OPTIMAL. YOU'VE BEEN WARNED.             //
//                                                      //
//  Writing this because its on the internet.           //
//                                                      //
//   Issues:                                            //
//    -- Probably we do copy/assign too much            //
//    -- Also we construct/destruct via visit which     //
//       may not be optimal given type(s).              //
//    -- Variant of single type is not allowed.         //
//    -- We do not utilize move/copy assignment         //
//    -- We destroy object then construct via move/copy //
//       construct.                                     //
//    --                                                //
// ==================================================== //

#include <utility>
#include <cstddef>
#include <cstdint>
#include <array>
#include <limits>
#include <functional>

#include "TypePack.h"

// TODO: We remove this when we fully confident of this
// variant to be perfectly usable. Until then, we specialize
// STL std::visit / std::variant_alternative etc. and
// swtich between std::variant and this variant.
#include <variant>

// TODO: Check if this works (and nvcc does not include it
// while compiling cuda code)
#ifndef MRAY_DEVICE_CODE_PATH
    #include "Error.h"
#endif

template<class... Types>
struct Variant;

template <class T, class... Types>
constexpr bool HoldsAlternative(const Variant<Types...>& v) noexcept;

template <uint32_t I, class VariantT>
constexpr decltype(auto) Alternative(VariantT&& v) noexcept;

template <class T, class VariantT>
constexpr decltype(auto) Alternative(VariantT&& v) noexcept;

template<class VariantT, class Func>
constexpr auto Visit(VariantT&& v, Func&& f) -> decltype(auto);

namespace VariantDetail
{
    template<uint32_t I>
    using UIntTConst = std::integral_constant<uint32_t, I>;

    template<class T>
    struct IndexOfResult
    {
        T    Index;
        bool IsUnique;
    };

    template<template<class...> class X, class... Args>
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
        // Even we state it is default, it may be deleted by the compiler.
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
    constexpr IndexOfResult<R> IndexOfTypeImpl();

    template<uint32_t O, class VariantT, class Func>
    constexpr auto SwitchCaseVisitImpl(UIntTConst<O>, VariantT&& v, Func&& f) -> decltype(auto);

    template<uint32_t O, class VariantT, class Func>
    constexpr auto IfElseVisitImpl(UIntTConst<O>, VariantT&& v, Func&& f) -> decltype(auto);

    template<uint32_t O, class VariantT, class Func>
    constexpr void IfElseIndexImpl(UIntTConst<O>, VariantT&& v, Func&& f);


    template <class Tp>
    struct TypePackToVariant;

    template <class... Ts>
    struct TypePackToVariant<TypePack<Ts...>>
    {
        using Type = Variant<Ts...>;
    };

    template<class... Types>
    class VariantImpl
    {
        // Logistics stuff (Lifetime and data movement)
        public:
        using TP = TypePack<Types...>;
        static constexpr bool TC  = AllTrait<std::is_trivially_constructible>(TP{});
        static constexpr bool TD  = AllTrait<std::is_trivially_destructible >(TP{});
        static constexpr bool TCC = AllTrait<std::is_trivially_copy_constructible>(TP{});
        static constexpr bool TMC = AllTrait<std::is_trivially_move_constructible>(TP{});
        static constexpr bool TCA = AllTrait<std::is_trivially_copy_assignable>(TP{});
        static constexpr bool TMA = AllTrait<std::is_trivially_move_assignable>(TP{});
        // Is first type default constructible ?
        using FirstType = TypePackElement<0, TP>;
        static constexpr bool FIRST_DC = std::is_default_constructible_v<FirstType>;
        static constexpr bool FIRST_TDC = std::is_trivially_default_constructible_v<FirstType>;

        private:
        // Index type related
        using VariantIndex = SelectVariantIndex<sizeof...(Types)>;
        static constexpr auto INVALID_INDEX = std::numeric_limits<VariantIndex>::max();
        // Storage type related
        using StorageType = UnionStorage<TC, TD, 0, sizeof...(Types), TypePack<Types...>>;
        // Friends
        template <class T, class... Ts>
        friend constexpr bool ::HoldsAlternative(const Variant<Ts...>&) noexcept;

        template <uint32_t I, class VariantT>
        friend constexpr decltype(auto) ::Alternative(VariantT&& v) noexcept;

        private:
        StorageType     storage;
        VariantIndex    tag = 0;

        // Helpers
        template<class T, uint32_t I, class... Args>
        constexpr void  ConstrcutAlternative(Args&&... args);
        constexpr void  DestroyAlternative();

        public:
        template<class T>
        static constexpr auto IndexOfType = IndexOfTypeImpl<VariantIndex, T, Types...>();
        template<class T>
        static constexpr bool TypeIsInPack = (IndexOfType<std::remove_cvref_t<T>>.Index != INVALID_INDEX);
        static constexpr auto TypeCount = sizeof...(Types);

        // Constructors & Destructor
        // TODO: MSVC Bug? Cant define outside class. It says it is ambiguous
        constexpr ~VariantImpl() noexcept requires(!TD)
        {
            if(tag != INVALID_INDEX) DestroyAlternative();
        }
        //
        constexpr             VariantImpl() noexcept requires(FIRST_DC);
        template<class T>
        constexpr             VariantImpl(T&&) noexcept requires(TypeIsInPack<T>);
        template<size_t I, class... Args>
        constexpr             VariantImpl(std::in_place_index_t<I>, Args&&... args);
        template<class T>
        constexpr VariantImpl& operator=(T&&) noexcept requires(TypeIsInPack<T>);
        // Logistics
        constexpr              VariantImpl(const VariantImpl& other) noexcept requires(!TCC);
        constexpr              VariantImpl(VariantImpl&& other) noexcept      requires(!TMC);
        constexpr VariantImpl& operator=(const VariantImpl& other) noexcept   requires(!TCA);
        constexpr VariantImpl& operator=(VariantImpl&& other) noexcept        requires(!TMA);
        //
        constexpr              VariantImpl() noexcept                         requires(!FIRST_DC) = delete;
        constexpr              VariantImpl(const VariantImpl& other) noexcept requires(TCC) = default;
        constexpr              VariantImpl(VariantImpl&& other) noexcept      requires(TMC) = default;
        constexpr VariantImpl& operator=(const VariantImpl& other) noexcept   requires(TCA) = default;
        constexpr VariantImpl& operator=(VariantImpl&& other) noexcept        requires(TMA) = default;
        constexpr              ~VariantImpl() noexcept                        requires(TD)  = default;
        //
        constexpr VariantIndex Index() const { return tag; };
        constexpr VariantIndex Index() { return tag; };

        constexpr VariantIndex index() const { return tag; };
        constexpr VariantIndex index() { return tag; };
    };

    template<class... Types>
    struct VariantSizeV<VariantImpl<Types...>>
        : std::integral_constant<size_t, sizeof...(Types)>
    {};
}

// I guess we need to wrap the actual implementation
// to an inner/base class, for nested template generations
// (Variant<Variant<int>, int>  etc.)
template<class... Types>
struct Variant : public VariantDetail::VariantImpl<Types...>
{
    using Base = VariantDetail::VariantImpl<Types...>;
    using Base::Base;

    template<class T>
    static constexpr auto IndexOfType = Base::template IndexOfType<T>;
    static constexpr auto TypeCount = sizeof...(Types);
};

template<size_t I, class... Types>
using VariantAlternative = VariantDetail::VariantAlternative<I, Types...>;

template<class T>
static constexpr auto VariantSize = VariantDetail::VariantSizeV<T>::value;

template <class... Ts>
using UniqueVariant = typename VariantDetail::TypePackToVariant<UniqueTypePack<Ts...>>::Type;

// Try to utilize execution unit of the compiler instead of
// instantiation unit. (NVCC reports too many conjunction instantiations)
template<template<class...> class X, class... Args>
constexpr bool VariantDetail::AllTrait(TypePack<Args...>)
{
    using SecondType = TypePackElement<1, TypePack<Args...>>;
    if constexpr(!X<SecondType>::value)
        return false;

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
constexpr VariantDetail::IndexOfResult<R>
VariantDetail::IndexOfTypeImpl()
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
    return IndexOfResult<R>{ .Index = tag, .IsUnique = (duplicate == R(1)) };
}

// std::visit fails on NVCC when STL is GCC's stl.
// (see: https://godbolt.org/z/fM811b4cx,
// for the compiled down result. It traps immediately)
//
// We use if/else chain instead of switch case also we
// expand the template by 16 to reduce instantiation count.
//
// This is technically not O(1) but on GPU switch may not be
// O(1) anyway, also NVCC crashed with swtich/case in the past
// so I'am reluctant to use it.
// Anyway there is an implementation down below, we may change
// to it if it is better.
template<uint32_t O, class VariantT, class Func>
constexpr auto VariantDetail::IfElseVisitImpl(UIntTConst<O>, VariantT&& v, Func&& f) -> decltype(auto)
{
    using V = std::remove_cvref_t<VariantT>;
    constexpr uint32_t STAMP_COUNT = 16;
    constexpr uint32_t VSize = uint32_t(V::TypeCount);
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
    COND_INVOKE(1)
    COND_INVOKE(2)
    COND_INVOKE(3)
    COND_INVOKE(4)
    COND_INVOKE(5)
    COND_INVOKE(6)
    COND_INVOKE(7)
    COND_INVOKE(8)
    COND_INVOKE(9)
    COND_INVOKE(10)
    COND_INVOKE(11)
    COND_INVOKE(12)
    COND_INVOKE(13)
    COND_INVOKE(14)
    COND_INVOKE(15)
    #undef COND_INVOKE
    if constexpr(VSize > O + STAMP_COUNT)
        return IfElseVisitImpl(UIntTConst<O + STAMP_COUNT>{},
                               std::forward<VariantT>(v),
                               std::forward<Func>(f));

    // On GPU we crash and burn...
    // on CPU we throw
    #ifdef MRAY_DEVICE_CODE_PATH
        MRAY_UNREACHABLE;
    #else
        else throw MRayError("Unable to invoke visit over a variant!");
    #endif
}

template<uint32_t O, class VariantT, class Func>
constexpr void VariantDetail::IfElseIndexImpl(UIntTConst<O>, VariantT&& v, Func&& f)
{
    using V = std::remove_cvref_t<VariantT>;
    constexpr uint32_t STAMP_COUNT = 16;
    constexpr uint32_t VSize = uint32_t(V::TypeCount);
    uint32_t index = uint32_t(v.Index());
    // I dunno how to make this compile time
    // so we check it runtime
    #define COND_INVOKE(I)                               \
        if constexpr(VSize > (I + O)) if(index == O + I) \
        {                                                \
            return std::invoke                           \
            (                                            \
                std::forward<Func>(f),                   \
                std::in_place_index<O + I>               \
            );                                           \
        }
    // End COND_INVOKE
    COND_INVOKE(0)
    COND_INVOKE(1)
    COND_INVOKE(2)
    COND_INVOKE(3)
    COND_INVOKE(4)
    COND_INVOKE(5)
    COND_INVOKE(6)
    COND_INVOKE(7)
    COND_INVOKE(8)
    COND_INVOKE(9)
    COND_INVOKE(10)
    COND_INVOKE(11)
    COND_INVOKE(12)
    COND_INVOKE(13)
    COND_INVOKE(14)
    COND_INVOKE(15)
    #undef COND_INVOKE
    if constexpr(VSize > O + STAMP_COUNT)
        return IfElseIndexImpl(UIntTConst<O + STAMP_COUNT>{},
                               std::forward<VariantT>(v),
                               std::forward<Func>(f));
    // On GPU we crash and burn...
    // on CPU we throw
    #ifdef MRAY_DEVICE_CODE_PATH
        MRAY_UNREACHABLE;
    #else
        else throw MRayError("Unable to invoke visit over a variant!");
    #endif
}

template<uint32_t O, class VariantT, class Func>
constexpr auto VariantDetail::SwitchCaseVisitImpl(UIntTConst<O>, VariantT&& v, Func&& f) -> decltype(auto)
{
    using V = std::remove_cvref_t<VariantT>;
    constexpr uint32_t STAMP_COUNT = 16;
    constexpr uint32_t VSize = uint32_t(V::TypeCount);
    // I dunno how to make this compile time
    // so we check it runtime
    #define CASE_INVOKE(I)                        \
        case(O + I):                              \
        {                                         \
            if constexpr(VSize > (I + O))         \
                return std::invoke                \
                (                                 \
                    std::forward<Func>(f),        \
                    Alternative<O + I>            \
                    (                             \
                        std::forward<VariantT>(v) \
                    )                             \
                );                                \
        }
    //
    switch(v.Index())
    {
        CASE_INVOKE(0)
        CASE_INVOKE(1)
        CASE_INVOKE(2)
        CASE_INVOKE(3)
        CASE_INVOKE(4)
        CASE_INVOKE(5)
        CASE_INVOKE(6)
        CASE_INVOKE(7)
        CASE_INVOKE(8)
        CASE_INVOKE(9)
        CASE_INVOKE(10)
        CASE_INVOKE(11)
        CASE_INVOKE(12)
        CASE_INVOKE(13)
        CASE_INVOKE(14)
        CASE_INVOKE(15)
        #undef CASE_INVOKE
        default:
        {
            if constexpr(VSize > O + STAMP_COUNT)
                return SwitchCaseVisitImpl
                (
                    UIntTConst<O + STAMP_COUNT>{},
                    std::forward<VariantT>(v),
                    std::forward<Func>(f)
                );
            // On GPU we crash and burn...
            // on CPU we throw
            #ifdef MRAY_DEVICE_CODE_PATH
                MRAY_UNREACHABLE;
            #else
                else throw MRayError("Unable to invoke visit over a variant!");
            #endif
        }
    }
}

template<class ... Types>
template<class TT, uint32_t I, class... Args>
constexpr
void VariantDetail::VariantImpl<Types...>::ConstrcutAlternative(Args&&... args)
{
    // Skip if this type is trivially default constructible
    // and deduced "ConstrcutAlternative" is called to default construct
    // the alternative
    constexpr auto IsTypeTDC = std::is_trivially_default_constructible_v<TT>;
    if constexpr(IsTypeTDC && sizeof...(Args) == 0)
        return;
    //
    // Or do the visitor...
    TT& loc = MetaGet<I, 0, TypeCount>(storage);
    std::construct_at<TT>(&loc, std::forward<Args>(args)...);

    // else Visit(*this, [&args...](auto& t)
    // {
    //     using T = std::remove_cvref_t<decltype(t)>;
    //     if constexpr(std::is_same_v<TT, T>)
    //         std::construct_at<T>(&t, std::forward<Args>(args)...);
    // });
}

template<class ... Types>
constexpr
void VariantDetail::VariantImpl<Types...>::DestroyAlternative()
{
    assert(tag != INVALID_INDEX);
    // If all the types are trivially destructible just skip
    if constexpr(TD) return;
    //
    // Or do the if else dance...
    else Visit(*this, [](auto& t)
    {
        std::destroy_at(&t);
    });
}

template<class... Types>
constexpr
VariantDetail::VariantImpl<Types...>::VariantImpl() noexcept requires(FIRST_DC)
    : tag(VariantIndex(0))
{
    // TODO: Do we really need this?
    if constexpr(FIRST_TDC)
    {
        std::construct_at<FirstType>(&MetaGet<0, 0, TypeCount>(storage));
    }
}

template<class... Types>
template<class T>
constexpr
VariantDetail::VariantImpl<Types...>::VariantImpl(T&& other) noexcept requires(TypeIsInPack<T>)
{
    using TBase = std::remove_cvref_t<T>;
    constexpr auto R = IndexOfType<TBase>;
    static_assert(R.IsUnique, "Given type is not unique in variant! "
                  "Cannot call \"Variant(T&&)\"");
    static_assert(R.Index != INVALID_INDEX, "Unable to Construct Variant with type T");
    //
    tag = R.Index;
    ConstrcutAlternative<TBase, R.Index>(std::forward<T>(other));
}

template<class... Types>
template<size_t I, class... Args>
constexpr
VariantDetail::VariantImpl<Types...>::VariantImpl(std::in_place_index_t<I>, Args&&... args)
{
    static_assert(I < TypeCount, "Given \"I\" is out of range!");
    using T = TypePackElement<I, TP>;
    //
    tag = I;
    ConstrcutAlternative<T, I>(std::forward<Args>(args)...);
}

template<class ... Types>
template<class T>
constexpr VariantDetail::VariantImpl<Types...>&
VariantDetail::VariantImpl<Types...>::operator=(T&& t) noexcept requires(TypeIsInPack<T>)
{
    using TBase = std::remove_cvref_t<T>;
    constexpr auto R = IndexOfType<TBase>;
    static_assert(R.IsUnique, "Given type is not unique in variant! "
                  "Cannot call \"operator=(T&&)\"");
    static_assert(R.Index != INVALID_INDEX, "Unable to Construct Variant with type T");
    //
    if(tag != INVALID_INDEX) DestroyAlternative();
    tag = R.Index;
    ConstrcutAlternative<TBase, R.Index>(std::forward<T>(t));
    return *this;
}

template<class ... Types>
constexpr
VariantDetail::VariantImpl<Types...>::VariantImpl(const VariantImpl& other) noexcept
requires(!TCC)
{
    tag = other.tag;
    //
    if(tag != INVALID_INDEX)
    {
        IfElseIndexImpl(UIntTConst<0>{}, *this, [&]<size_t I>(std::in_place_index_t<I>)
        {
            std::construct_at(&Alternative<I>(*this), Alternative<I>(other));
        });
    }
}

template<class ... Types>
constexpr
VariantDetail::VariantImpl<Types...>::VariantImpl(VariantImpl&& other) noexcept
requires(!TMC)
{
    tag = other.tag;
    if(tag != INVALID_INDEX)
    {
        IfElseIndexImpl(UIntTConst<0>{}, * this, [&]<size_t I>(std::in_place_index_t<I>)
        {
            std::construct_at(&Alternative<I>(*this),
                              Alternative<I>(std::forward<VariantImpl>(other)));
        });
    }
    // Other is in "moved from" state. Revert back to monostate?
    other.tag = INVALID_INDEX;
}

template<class ... Types>
constexpr VariantDetail::VariantImpl<Types...>&
VariantDetail::VariantImpl<Types...>::operator=(const VariantImpl& other) noexcept
requires(!TCA)
{
    assert(this != &other);
    if(tag != INVALID_INDEX) DestroyAlternative();
    //
    tag = other.tag;
    if(tag != INVALID_INDEX)
    {
        IfElseIndexImpl(UIntTConst<0>{}, * this, [&]<size_t I>(std::in_place_index_t<I>)
        {
            std::construct_at(&Alternative<I>(*this), Alternative<I>(other));
        });
    }
    return *this;
}

template<class ... Types>
constexpr VariantDetail::VariantImpl<Types...>&
VariantDetail::VariantImpl<Types...>::operator=(VariantImpl&& other) noexcept
requires(!TMA)
{
    assert(this != &other);
    if(tag != INVALID_INDEX) DestroyAlternative();
    //
    tag = other.tag;
    if(tag != INVALID_INDEX)
    {
        IfElseIndexImpl(UIntTConst<0>{}, * this, [&]<size_t I>(std::in_place_index_t<I>)
        {
            std::construct_at(&Alternative<I>(*this),
                              Alternative<I>(std::forward<VariantImpl>(other)));
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
    constexpr auto R = Variant<Types...>::template IndexOfType<T>;
    static_assert(R.IsUnique, "Given type is not unique in variant! "
                  "Cannot call \"HoldsAlternative\"");
    return v.tag == R.Index;
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
    constexpr auto R = V::template IndexOfType<T>;
    static_assert(R.IsUnique, "Given type is not unique in variant! "
                  "Use index version of \"Alternative\"");
    return Alternative<R.Index>(std::forward<VariantT>(v));
}

template<class VariantT, class Func>
constexpr auto Visit(VariantT&& v, Func&& f) -> decltype(auto)
{
    using namespace VariantDetail;
    assert(v.Index() < VariantSize<std::remove_cvref_t<VariantT>>);
    return IfElseVisitImpl(UIntTConst<0>{}, std::forward<VariantT>(v), std::forward<Func>(f));
    // TODO: Enable / Disable this depending on the performance.
    //return SwitchCaseVisitImpl(UIntTConst<0>{}, std::forward<VariantT>(v), std::forward<Func>(f));
}

template<class Func, class... Args>
constexpr auto Visit(std::variant<Args...>&& v, Func&& f) -> decltype(auto)
{
    return std::visit(f, v);
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

    template <class T, class... Args>
    constexpr T&
    get(Variant<Args...>& v)
    {
        return Alternative<T>(v);
    }

    template <class T, class... Args>
    constexpr const T&
    get(const Variant<Args...>& v)
    {
        return Alternative<T>(v);
    }

    template<class... Args>
    struct variant_size<Variant<Args...>>
        : public integral_constant<size_t, sizeof...(Args)>
    {};
}
