#pragma once

#include <span>
#include <optional>
#include <variant>
#include <functional>

#include "Definitions.h"

#include "Variant.h"

// Rename the std::optional, gpu may not like it
// most(all after c++20) of optional is constexpr
// so the "relaxed-constexpr" flag of nvcc will be able to compile it
// Just to be sure, aliasing here to ease refactoring
template <class T>
using Optional = std::optional<T>;

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

template<class F, class S>
Pair(F&&, S&&) -> Pair<std::remove_cvref_t<F>,
                       std::remove_cvref_t<S>>;


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