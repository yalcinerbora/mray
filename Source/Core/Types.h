#pragma once

#include "Definitions.h"
#include <array>
#include <string_view>

// For some recent compilers (for examlpe, clang-18)
// CTAD for alias templates (C++20 feature) is not implemented
// yet. so this does not work.
//
//template<class First, class Second>
//using Pair = std::pair<First, Second>;
//
// So we inherit the pair
template<class First, class Second>
struct Pair : public std::pair<First, Second>
{
    using Base = std::pair<First, Second>;
    using Base::Base;
};

#ifndef MRAY_GCC

template<class F, class S>
Pair(F&&, S&&) -> Pair<std::remove_cvref_t<F>,
                       std::remove_cvref_t<S>>;

#endif

// TODO: reference_wrapper<T> vs. span<T,1> which is better?
template <class T>
using Ref = std::reference_wrapper<T>;

template<class E>
using EnumNameArray = std::array<std::string_view, static_cast<uint32_t>(E::END)>;

template <class T>
struct SampleT
{
    T       value;
    Float   pdf;
};

// TODO: Move this later
template <class T>
struct IdentityFunctor
{
    MR_PF_DECL T operator()(const T& t) const noexcept { return t; }
};

// Forward Declarations of Tuple / Variant etc..

// Please see "Span.h" why dynamic extent marker is zero insteat of
// "INT32_MAX".
static constexpr uint32_t DynamicExtent = uint32_t(0);

template <class T, uint32_t Extent = DynamicExtent> class Span;

template<class... Ts> struct Tuple;

template<class... Types> struct Variant;

template<class T> struct Optional;
