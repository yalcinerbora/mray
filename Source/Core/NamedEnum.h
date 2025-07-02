#pragma once

#include <array>
#include <cstdint>

#include "Error.h"

template<class Enum, const std::array<const char*, static_cast<size_t>(Enum::END)>& NamesIn>
class NamedEnum
{
    static constexpr std::array Names = NamesIn;
    public:
    using E = Enum;
    // This do not work :(
    //using enum E;

    private:
    E e;

    public:
                NamedEnum() = default;
    constexpr   NamedEnum(E eIn);
    constexpr   NamedEnum(std::string_view sv);

    constexpr   operator E() const;
    constexpr   operator E();

    constexpr std::string_view ToString() const;
    constexpr std::strong_ordering operator<=>(const NamedEnum&) const;
};

template<class Enum, const std::array<const char*, static_cast<size_t>(Enum::END)>& NamesIn>
constexpr NamedEnum<Enum, NamesIn>::NamedEnum(E eIn)
    : e(eIn)
{}

template<class Enum, const std::array<const char*, static_cast<size_t>(Enum::END)>& NamesIn>
constexpr NamedEnum<Enum, NamesIn>::NamedEnum(std::string_view sv)
{
    auto loc = std::find_if(Names.cbegin(), Names.cend(),
                            [&](std::string_view r)
    {
        return sv == r;
    });
    if(loc == Names.cend())
        throw MRayError("Bad enum name");

    e = E(std::distance(Names.cbegin(), loc));
}

template<class Enum, const std::array<const char*, static_cast<size_t>(Enum::END)>& NamesIn>
constexpr NamedEnum<Enum, NamesIn>::operator E() const
{
    return e;
}

template<class Enum, const std::array<const char*, static_cast<size_t>(Enum::END)>& NamesIn>
constexpr NamedEnum<Enum, NamesIn>::operator E()
{
    return e;
}

template<class Enum, const std::array<const char*, static_cast<size_t>(Enum::END)>& NamesIn>
constexpr std::strong_ordering NamedEnum<Enum, NamesIn>::operator<=>(const NamedEnum& right) const
{
    return std::underlying_type_t<Enum>{e} <=> std::underlying_type_t<Enum>{right.e};
}

template<class Enum, const std::array<const char*, static_cast<size_t>(Enum::END)>& NamesIn>
constexpr std::string_view
NamedEnum<Enum, NamesIn>::ToString() const
{
    assert(e < E::END);
    return Names[static_cast<uint32_t>(e)];
}

