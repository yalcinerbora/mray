#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include "Core/Types.h"
#include "Core/Tuple.h"
#include "Core/Variant.h"

// ========================================= //
//   TODO: IMPLEMENT MORE TESTS!             //
//         THIS IS NOWHERE NEAR COMPLETE.    //
//                                           //
// We use variant quite extensivley and it   //
// has its own logistic operations           //
// (constructors / assignment / destructor)  //
// so we need to test these extensively      //
// ========================================= //

// To change the type for variant etc
// this is templated.
template<int I>
struct Lifetimer
{
    static uint32_t cCount;
    static uint32_t ccCount;
    static uint32_t mcCount;
    static uint32_t caCount;
    static uint32_t maCount;
    static uint32_t dCount;
    // Constructors & Destructor
                Lifetimer()                 {  cCount++; }
                Lifetimer(const Lifetimer&) { ccCount++; }
                Lifetimer(Lifetimer&&)      { mcCount++; }
    Lifetimer&  operator=(const Lifetimer&) { caCount++; return *this; }
    Lifetimer&  operator=(Lifetimer&&)      { maCount++; return *this; }
                ~Lifetimer()                {  dCount++; }

    static void ResetCounters()
    {
        cCount  = 0;
        ccCount = 0;
        mcCount = 0;
        caCount = 0;
        maCount = 0;
        dCount  = 0;
    }
    static void CheckCounters(std::array<int, 6> expected)
    {
        EXPECT_EQ(cCount,  expected[0]) << "     Constructor";
        EXPECT_EQ(ccCount, expected[1]) << "Copy Constructor";
        EXPECT_EQ(mcCount, expected[2]) << "Move Constructor";
        EXPECT_EQ(caCount, expected[3]) << "Copy Assign";
        EXPECT_EQ(maCount, expected[4]) << "Move Assign";
        EXPECT_EQ(dCount,  expected[5]) << "Destructor!";
    }
    static void CheckAndResetCounters(std::array<int, 6> expected)
    {
        CheckCounters(expected);
        ResetCounters();
    }

};
template<int I> uint32_t Lifetimer<I>::cCount  = 0;
template<int I> uint32_t Lifetimer<I>::ccCount = 0;
template<int I> uint32_t Lifetimer<I>::mcCount = 0;
template<int I> uint32_t Lifetimer<I>::caCount = 0;
template<int I> uint32_t Lifetimer<I>::maCount = 0;
template<int I> uint32_t Lifetimer<I>::dCount  = 0;

struct NoDefaultConstructClass
{
    NoDefaultConstructClass() = delete;
};

template<class V>
concept VariantDefaultUnconstructible = requires() { V(); };

TEST(VariantTest, Lifetime)
{
    // First type cannot be default constructed
    {
        using V = Variant<NoDefaultConstructClass, Lifetimer<0>>;
        static_assert(!VariantDefaultUnconstructible<V>, "This should not be default constructible");
    }
    // First type is default constructed
    //using TestVariant = Variant<Lifetimer<0>, Lifetimer<1>, Lifetimer<2>>;
    using TestVariant = std::variant<Lifetimer<0>, Lifetimer<1>, Lifetimer<2>>;
    //
    {
        static_assert(VariantDefaultUnconstructible<TestVariant>,
                      "This should be default constructible");
        TestVariant v;
    }
    Lifetimer<0>::CheckAndResetCounters({1, 0, 0, 0, 0, 1});
    Lifetimer<1>::CheckAndResetCounters({0, 0, 0, 0, 0, 0});
    Lifetimer<2>::CheckAndResetCounters({0, 0, 0, 0, 0, 0});

    // These two should be the same
    // TODO: Does standard guarantees this???
    {
        TestVariant v = Lifetimer<1>();
    }
    Lifetimer<0>::CheckAndResetCounters({0, 0, 0, 0, 0, 0});
    Lifetimer<1>::CheckAndResetCounters({1, 0, 1, 0, 0, 2});
    Lifetimer<2>::CheckAndResetCounters({0, 0, 0, 0, 0, 0});

    {
        const Lifetimer<2> c;
        TestVariant v = c;
    }
    Lifetimer<0>::CheckAndResetCounters({0, 0, 0, 0, 0, 0});
    Lifetimer<1>::CheckAndResetCounters({0, 0, 0, 0, 0, 0});
    Lifetimer<2>::CheckAndResetCounters({1, 1, 0, 0, 0, 2});

    {
        TestVariant v = Lifetimer<2>();
        v = Lifetimer<2>();
    }
    Lifetimer<0>::CheckAndResetCounters({0, 0, 0, 0, 0, 0});
    Lifetimer<1>::CheckAndResetCounters({0, 0, 0, 0, 0, 0});
    Lifetimer<2>::CheckAndResetCounters({2, 0, 1, 0, 1, 3});


}