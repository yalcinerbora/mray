#pragma once

#include <bitset>

template<class Enum, unsigned int S> class Flag;
template<class Enum, unsigned int S> Flag<Enum, S> operator|(Enum e1, Enum e2);

template<class Enum, unsigned int S = static_cast<int>(Enum::END)>
class Flag
{
    public:
        // I mean... lets just be sure
        static_assert(sizeof(Enum) <= sizeof(size_t),
                      "Flag -> Enum may have more than size_t amount of data");

        using F         = Enum;
        using BitRef    = typename std::bitset<S>::reference;

    private:
        std::bitset<S> flagData;

    protected:

    public:
        // Constructors & Destructor
                        Flag() = default;
                        Flag(uint64_t);
                        Flag(Enum);
                        template <unsigned int C>
                        Flag(const std::array<Enum, C>& vals);
                        Flag(const Flag&) = default;
                        Flag(Flag&&) = default;
        Flag&           operator=(const Flag&) = default;
        Flag&           operator=(Flag&&) = default;
                        ~Flag() = default;

        BitRef          operator[](Enum);
        bool            operator[](Enum) const;

        Flag&           operator|(Enum);
        Flag&           operator|=(Enum);

        template<class E, int Sz>
        friend Flag     operator|(E, E);
};

template <class Enum, unsigned int S = static_cast<int>(Enum::END)>
Flag<Enum, S> operator|(Enum e1, Enum e2)
{
    return Flag<Enum, S>(std::array<Enum, 2>{e1, e2});
}

template<class Enum, unsigned int S>
Flag<Enum, S>::Flag(uint64_t v)
 : flagData(v)
{}

template<class Enum, unsigned int S>
Flag<Enum, S>::Flag(Enum e)
{
    flagData[static_cast<size_t>(e)] = true;
}

template<class Enum, unsigned int S>
template <unsigned int C>
Flag<Enum, S>::Flag(const std::array<Enum, C>& vals)
{
    for(Enum e : vals)
        flagData.set(static_cast<size_t>(e));
}

template<class Enum, unsigned int S>
typename std::bitset<S>::reference Flag<Enum, S>::operator[](Enum e)
{
    return flagData[static_cast<size_t>(e)];
}

template<class Enum, unsigned int S>
bool Flag<Enum, S>::operator[](Enum e) const
{
    return flagData[static_cast<size_t>(e)];
}

template<class Enum, unsigned int S>
Flag<Enum, S>& Flag<Enum, S>::operator|(Enum e)
{
    // Set is fine here
    flagData.set(static_cast<size_t>(e));
    return *this;
}

template<class Enum, unsigned int S>
Flag<Enum, S>& Flag<Enum, S>::operator|=(Enum e)
{
    // Set is fine here
    flagData.set(static_cast<size_t>(e));
    return *this;
}