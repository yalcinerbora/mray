#pragma once

// Half library
// GPU and CPU and NVIDIA AMD agnostic type
//
// This implementation is quite cumbersome but w/e

// Now we have matrix rows of MSVC/CLANG/GCC
// on the other side (columns) CPU (x86 only) / NVCC / HIP,
// (sycl in future).
//
// We probably want constexpr support just cherry on top.
// (Since Vector<N, T> is defined as constexpr)
//
// So yeah... The design is uint16_t defined type for all
// platforms, and we bit-cast to actual CPU type _Float16 if avail
// while we do any operations on them.
//
// We hope this approach does not break compiler optimization etc.
// We will directly lift the data to float everytime we
// do a calculation on the CPU side.
//
// For GPU, we will do a bitcast and do the operation, so it shouldn't
// be a problem. If it is this code will be a good demo to improve the
// bitcast of compilers maybe.
//
// __________________________________
// Update: I've checked the code, CPU-side half should not be used
// (It has many f16<->fp32 promotions demotions and some registers
// are not considered as transient.)
//
// For GPU side, it is fine, storage->native bitcasts does not cost anyhing
// codegen-wise, (highly probably hinders compilation time)
//
// Here is the godbolt:
// https://godbolt.org/z/hrEn3vPnr
//
// When instruction set supports half (AVX Half extension or smth.)
// codegen is kinda fine except for ABI since we pass uint16_t technically,
// so no xmm register passing etc.
//
// But codegen is awful when only 32-bit float is supported
// so take care

#include <cstdint>
#include "Definitions.h"
#include "BitFunctions.h"

#if __cplusplus >= 202302L
    #include <stdfloat> // IWYU pragma: keep
#endif

// Check if we have the native fp16 from the compiler
#if defined(__STDCPP_FLOAT16_T__)
    using CPUNativeFP16 = std::float16_t;
#elif defined MRAY_CLANG
    using CPUNativeFP16 = _Float16;
#elif defined MRAY_GCC
    using CPUNativeFP16 = _Float16;
#else // vvvv MSVC vvvvv
    using CPUNativeFP16 = uint16_t;
#endif

// When code compiles for device
#ifdef MRAY_DEVICE_CODE_PATH_CUDA
    #include <cuda_fp16.h>
    using NativeHalf = __half;
#elif defined MRAY_DEVICE_CODE_PATH_HIP
    #include <hip/hip_fp16.h>
    using NativeHalf = __half;
#else
    using NativeHalf = CPUNativeFP16;
#endif // DEBUG

static constexpr bool MRAY_CPU_EMULATE_FP16 = std::is_same_v<CPUNativeFP16, uint16_t>;

// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
// Get "_cvtss_sh" and "_cvtsh_ss"
// Hipster MSVC ofc. does not have these (checked many headers).
//
// It has the wide version so we emulate. Hope it does not break
// vectorization etc. (probably will but w/e)
#ifdef MRAY_MSVC
    #include <immintrin.h>

    MRAY_FORCE_INLINE_DECL
    unsigned short _cvtss_sh(float a, int)
    {
        auto reg = _mm_cvtps_ph(_mm_set_ss(a), _MM_FROUND_TO_NEAREST_INT);
        return unsigned short(_mm_extract_epi16(reg, 0));
    }
    MRAY_FORCE_INLINE_DECL
    float _cvtsh_ss(unsigned short a)
    {
        auto reg = _mm_set1_epi16(Bit::BitCast<short>(a));
        return _mm_cvtss_f32(_mm_cvtph_ps(reg));
    }

#else
    #include <immintrin.h>
    #include <emmintrin.h>
    #include <cmath>
#endif

namespace HalfDetail
{
    MR_PF_DECL uint16_t FloatToHalf(float f);
    MR_PF_DECL float HalfToFloat(uint16_t f);
}

struct alignas(2) Half
{
    uint16_t val;
    // Constructors & Destructor;
    MR_PF_DEF_V explicit Half(uint16_t);
    MR_PF_DEF_V explicit Half(float);
    MR_PF_DEF   Half&    operator=(float);
    MR_PF_DEF   explicit operator float() const;

    static constexpr Half NaN();
    static constexpr Half Inf();
    static constexpr Half Max();
    static constexpr Half Min();
    static constexpr Half Epsilon();
};

MR_PF_DECL Half operator-(const Half&) noexcept;

MR_PF_DECL Half operator+(const Half&, const Half&) noexcept;
MR_PF_DECL Half operator-(const Half&, const Half&) noexcept;
MR_PF_DECL Half operator*(const Half&, const Half&) noexcept;
MR_PF_DECL Half operator/(const Half&, const Half&) noexcept;

MR_PF_DECL Half& operator+=(Half&, const Half&) noexcept;
MR_PF_DECL Half& operator+=(Half&, const Half&) noexcept;
MR_PF_DECL Half& operator*=(Half&, const Half&) noexcept;
MR_PF_DECL Half& operator/=(Half&, const Half&) noexcept;

MR_PF_DECL bool operator==(const Half&, const Half&) noexcept;
MR_PF_DECL bool operator!=(const Half&, const Half&) noexcept;
MR_PF_DECL bool operator> (const Half&, const Half&) noexcept;
MR_PF_DECL bool operator< (const Half&, const Half&) noexcept;
MR_PF_DECL bool operator>=(const Half&, const Half&) noexcept;
MR_PF_DECL bool operator<=(const Half&, const Half&) noexcept;

MR_PF_DEF
uint16_t HalfDetail::FloatToHalf(float f)
{
    if(std::is_constant_evaluated())
    {
        // TODO:
        return Half::NaN().val;
    }
    // TODO: Dirty fix, this should not reach to device compiler since it is
    // host code.
    #ifndef MRAY_DEVICE_CODE_PATH
        else if constexpr(MRAY_CPU_EMULATE_FP16)
        {
            return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
    #endif
    else return Bit::BitCast<uint16_t>(CPUNativeFP16(f));
}

MR_PF_DEF
float HalfDetail::HalfToFloat(uint16_t f)
{
    if(std::is_constant_evaluated())
    {
        // TODO:
        return NAN;
    }
    // TODO: Dirty fix, this should not reach to device compiler since it is
    // host code.
    #ifndef MRAY_DEVICE_CODE_PATH
        else if constexpr(MRAY_CPU_EMULATE_FP16)
        {
            return _cvtsh_ss(f);
        }
    #endif
    else return float(Bit::BitCast<CPUNativeFP16>(f));
}

MR_PF_DEF_V
Half::Half(uint16_t v)
    : val{v}
{}

MR_PF_DECL_V
Half::Half(float f)
    : val(HalfDetail::FloatToHalf(f))
{}

MR_PF_DECL
Half& Half::operator=(float v)
{
    val = HalfDetail::FloatToHalf(v);
    return *this;
}

MR_PF_DECL
Half::operator float() const
{
    return HalfDetail::HalfToFloat(val);
}

constexpr Half Half::NaN()     { return Half(uint16_t(0x7FFF)); }
constexpr Half Half::Inf()     { return Half(uint16_t(0x7C00)); }
constexpr Half Half::Max()     { return Half(uint16_t(0x7BFF)); }
constexpr Half Half::Min()     { return Half(uint16_t(0x0400)); }
constexpr Half Half::Epsilon() { return Half(uint16_t(0x1400)); }

// ============================== //
//      ARITHMETIC OPERATORS      //
// ============================== //
MR_PF_DEF
Half operator-(const Half& l) noexcept
{
    using namespace Bit;
    using H = NativeHalf;
    if(std::is_constant_evaluated())    return Half(-float(l));
    if constexpr(MRAY_CPU_EMULATE_FP16) return Half(-float(l));
     // See operator+ for the redundant cast
    else return BitCast<Half>(H(-BitCast<H>(l.val)));
}

MR_PF_DEF
Half operator+(const Half& l, const Half& r) noexcept
{
    using namespace Bit;
    using H = NativeHalf;
    if(std::is_constant_evaluated())    return Half(float(l) + float(r));
    if constexpr(MRAY_CPU_EMULATE_FP16) return Half(float(l) + float(r));
    // The Bit-Cast below does not compile even though
    // it should not be reachable when "MRAY_CPU_EMULATE_FP16" false.
    // when it is true "NativeHalf" (which is "H" in this case)
    // is uint16_t and when we convert and do operator+ it is promoted to
    // a int. Then BitCast int->Half fails.
    //
    // We also directly convert the ".val" instead of the Half itself
    // since BitCast will translate it to the intrinsic
    // ("__half_as_ushort" etc.)
    else return BitCast<Half>(H(BitCast<H>(l.val) + BitCast<H>(r.val)));
}

MR_PF_DEF
Half operator-(const Half& l, const Half& r) noexcept
{
    using namespace Bit;
    using H = NativeHalf;
    if(std::is_constant_evaluated())    return Half(float(l) - float(r));
    if constexpr(MRAY_CPU_EMULATE_FP16) return Half(float(l) - float(r));
    // See operator+ for the redundant cast
    else return BitCast<Half>(H(BitCast<H>(l.val) - BitCast<H>(r.val)));
}

MR_PF_DEF
Half operator*(const Half& l, const Half& r) noexcept
{
    using namespace Bit;
    using H = NativeHalf;
    if(std::is_constant_evaluated())    return Half(float(l) * float(r));
    if constexpr(MRAY_CPU_EMULATE_FP16) return Half(float(l) * float(r));
    // See operator+ for the redundant cast
    else return BitCast<Half>(H(BitCast<H>(l.val) * BitCast<H>(r.val)));
}

MR_PF_DEF
Half operator/(const Half& l, const Half& r) noexcept
{
    using namespace Bit;
    using H = NativeHalf;
    if(std::is_constant_evaluated())    return Half(float(l) / float(r));
    if constexpr(MRAY_CPU_EMULATE_FP16) return Half(float(l) / float(r));
    // See operator+ for the redundant cast
    else return BitCast<Half>(H(BitCast<H>(l.val) / BitCast<H>(r.val)));
}

// ============================== //
//      ASSIGNMENT OPERATORS      //
// ============================== //
MR_PF_DEF
Half& operator+=(Half& l, const Half& r) noexcept
{
    using namespace Bit;
    if(std::is_constant_evaluated())
    {
        float v = float(l);
        l = Half(v += float(r));
    }
    else
    {
        using H = NativeHalf;
        if constexpr(MRAY_CPU_EMULATE_FP16)
        {
            float v = float(l);
            l = Half(v += float(r));
        }
        else
        {
            H lH = BitCast<H>(l.val);
            l.val = BitCast<uint16_t>(lH += BitCast<H>(r.val));
        }
    }
    return l;
}

MR_PF_DEF
Half& operator-=(Half& l, const Half& r) noexcept
{
    using namespace Bit;
    if(std::is_constant_evaluated())
    {
        float v = float(l);
        l = Half(v -= float(r));
    }
    else
    {
        using H = NativeHalf;
        if constexpr(MRAY_CPU_EMULATE_FP16)
        {
            float v = float(l);
            l = Half(v -= float(r));
        }
        else
        {
            H lH = BitCast<H>(l.val);
            l.val = BitCast<uint16_t>(lH -= BitCast<H>(r.val));
        }
    }
    return l;
}

MR_PF_DEF
Half& operator*=(Half& l, const Half& r) noexcept
{
    using namespace Bit;
    if(std::is_constant_evaluated())
    {
        float v = float(l);
        l = Half(v *= float(r));
    }
    else
    {
        using H = NativeHalf;
        if constexpr(MRAY_CPU_EMULATE_FP16)
        {
            float v = float(l);
            l = Half(v *= float(r));
        }
        else
        {
            H lH = BitCast<H>(l.val);
            l.val = BitCast<uint16_t>(lH *= BitCast<H>(r.val));
        }
    }
    return l;
}

MR_PF_DEF
Half& operator/=(Half& l, const Half& r) noexcept
{
    using namespace Bit;
    if(std::is_constant_evaluated())
    {
        float v = float(l);
        l = Half(v /= float(r));
    }
    else
    {
        using H = NativeHalf;
        if constexpr(MRAY_CPU_EMULATE_FP16)
        {
            float v = float(l);
            l = Half(v /= float(r));
        }
        else
        {
            H lH = BitCast<H>(l.val);
            l.val = BitCast<uint16_t>(lH /= BitCast<H>(r.val));
        }
    }
    return l;
}

// ============================== //
//      COMPARISON OPERATORS      //
// ============================== //
MR_PF_DEF
bool operator==(const Half& l, const Half& r) noexcept
{
    using namespace Bit;
    using H = NativeHalf;
    if(std::is_constant_evaluated())    return float(l) == float(r);
    if constexpr(MRAY_CPU_EMULATE_FP16) return float(l) == float(r);
    else                                return BitCast<H>(l.val) == BitCast<H>(r.val);
}

MR_PF_DEF
bool operator!=(const Half& l, const Half& r) noexcept
{
    using namespace Bit;
    using H = NativeHalf;
    if(std::is_constant_evaluated())    return float(l) != float(r);
    if constexpr(MRAY_CPU_EMULATE_FP16) return float(l) != float(r);
    else                                return BitCast<H>(l.val) != BitCast<H>(r.val);
}

MR_PF_DEF
bool operator>(const Half& l, const Half& r) noexcept
{
    using namespace Bit;
    using H = NativeHalf;
    if(std::is_constant_evaluated())    return float(l) > float(r);
    if constexpr(MRAY_CPU_EMULATE_FP16) return float(l) > float(r);
    else                                return BitCast<H>(l.val) > BitCast<H>(r.val);
}

MR_PF_DEF
bool operator<(const Half& l, const Half& r) noexcept
{
    using namespace Bit;
    using H = NativeHalf;
    if(std::is_constant_evaluated())    return float(l) < float(r);
    if constexpr(MRAY_CPU_EMULATE_FP16) return float(l) < float(r);
    else                                return BitCast<H>(l.val) < BitCast<H>(r.val);
}

MR_PF_DEF
bool operator>=(const Half& l, const Half& r) noexcept
{
    using namespace Bit;
    using H = NativeHalf;
    if(std::is_constant_evaluated())    return float(l) >= float(r);
    if constexpr(MRAY_CPU_EMULATE_FP16) return float(l) >= float(r);
    else                                return BitCast<H>(l.val) >= BitCast<H>(r.val);
}

MR_PF_DEF
bool operator<=(const Half& l, const Half& r) noexcept
{
    using namespace Bit;
    using H = NativeHalf;
    if(std::is_constant_evaluated())    return float(l) <= float(r);
    if constexpr(MRAY_CPU_EMULATE_FP16) return float(l) <= float(r);
    else                                return BitCast<H>(l.val) <= BitCast<H>(r.val);
}

// And numeric limits to the std for generic code.
// TODO: Add volatile / const all that bs.
namespace std
{
    template<>
    class numeric_limits<Half>
    {
        public:
        static constexpr bool is_specialized = true;

        static constexpr Half min() noexcept { return Half::Min(); }
        static constexpr Half max() noexcept { return Half::Max(); }
        static constexpr Half lowest() noexcept { return Half(uint16_t(0xFBFF)); }

        static constexpr int digits       = 11;
        static constexpr int digits10     = 3;
        static constexpr int max_digits10 = 5;

        static constexpr bool is_signed  = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact   = false;

        static constexpr int radix = 2;
        static constexpr Half epsilon() noexcept { return Half::Epsilon(); }
        static constexpr Half round_error() noexcept { return Half(uint16_t(0x3800)); }

        static constexpr int min_exponent   = -13;
        static constexpr int min_exponent10 = -4;
        static constexpr int max_exponent   = 16;
        static constexpr int max_exponent10 = 4;

        static constexpr bool has_infinity  = true;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = true;
        static constexpr bool has_denorm_loss = false;

        static constexpr Half infinity()      noexcept { return Half::Inf(); }
        static constexpr Half quiet_NaN()     noexcept { return Half::NaN(); }
        static constexpr Half signaling_NaN() noexcept { return Half(uint16_t(0x7E01)); }
        static constexpr Half denorm_min()    noexcept { return Half(uint16_t(0x0001)); }

        static constexpr bool is_iec559       = true;
        static constexpr bool is_bounded      = true;
        static constexpr bool is_modulo       = false;
        static constexpr bool traps           = false;
        static constexpr bool tinyness_before = true;

        static constexpr float_round_style round_style = round_to_nearest;
    };
}