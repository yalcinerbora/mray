#pragma once

#include "TypeFinder.h"
#include "Definitions.h"
#include "NormTypes.h"

// Compilers successfully get away with
// forward declaration, but if user forgot to
// include these, hell break loose. ("Domain expert"-level
// compiler diagnostics...)
// So we pre-include these here
#include "Vector.h"     // IWYU pragma: keep
#include "Matrix.h"     // IWYU pragma: keep
#include "Quaternion.h" // IWYU pragma: keep
#include "Ray.h"        // IWYU pragma: keep
#include "AABB.h"       // IWYU pragma: keep

#include <functional>

// This implementation is quite taxing on compilation times.
// It is used to "automatically" create switch/case statements
// for the given Pixel/Data type.
// Here is the generated code (Only copy pasted the Pixel portion)
// https://godbolt.org/z/d99s1v7zP
//
// MSVC is slightly better at optimizing these
// ; merges same case statements into one.
// Clang/GCC does not do that (Maybe it does not worth it?)
namespace MRayDataDetail
{
    using M = TypeFinder::E_TMapper<MRayDataEnum>;
    using Mapper = M::Map
    <
        typename M:: template ETPair<MRayDataEnum::MR_INT8,         int8_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2C,    Vector2c>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3C,    Vector3c>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4C,    Vector4c>,

        typename M:: template ETPair<MRayDataEnum::MR_INT16,        int16_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2S,    Vector2s>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3S,    Vector3s>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4S,    Vector4s>,

        typename M:: template ETPair<MRayDataEnum::MR_INT32,        int32_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2I,    Vector2i>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3I,    Vector3i>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4I,    Vector4i>,

        typename M:: template ETPair<MRayDataEnum::MR_INT64,        int64_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2L,    Vector2l>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3L,    Vector3l>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4L,    Vector4l>,

        typename M:: template ETPair<MRayDataEnum::MR_UINT8,        uint8_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2UC,   Vector2uc>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3UC,   Vector3uc>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4UC,   Vector4uc>,

        typename M:: template ETPair<MRayDataEnum::MR_UINT16,       uint16_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2US,   Vector2us>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3US,   Vector3us>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4US,   Vector4us>,

        typename M:: template ETPair<MRayDataEnum::MR_UINT32,       uint32_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2UI,   Vector2ui>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3UI,   Vector3ui>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4UI,   Vector4ui>,

        typename M:: template ETPair<MRayDataEnum::MR_UINT64,       uint64_t>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2UL,   Vector2ul>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3UL,   Vector3ul>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4UL,   Vector4ul>,

        typename M:: template ETPair<MRayDataEnum::MR_FLOAT,        float>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_2,     Vector2>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_3,     Vector3>,
        typename M:: template ETPair<MRayDataEnum::MR_VECTOR_4,     Vector4>,

        typename M:: template ETPair<MRayDataEnum::MR_QUATERNION,   Quaternion>,
        typename M:: template ETPair<MRayDataEnum::MR_MATRIX_4x4,   Matrix4x4>,
        typename M:: template ETPair<MRayDataEnum::MR_MATRIX_3x3,   Matrix2x2>,
        typename M:: template ETPair<MRayDataEnum::MR_AABB3,        AABB3>,
        typename M:: template ETPair<MRayDataEnum::MR_RAY,          Ray>,

        typename M:: template ETPair<MRayDataEnum::MR_UNORM_4x8,    UNorm4x8>,
        typename M:: template ETPair<MRayDataEnum::MR_UNORM_2x16,   UNorm2x16>,
        typename M:: template ETPair<MRayDataEnum::MR_SNORM_4x8,    SNorm4x8>,
        typename M:: template ETPair<MRayDataEnum::MR_SNORM_2x16,   SNorm2x16>,

        typename M:: template ETPair<MRayDataEnum::MR_STRING,       std::string_view>,
        typename M:: template ETPair<MRayDataEnum::MR_BOOL,         bool>
    >;

    template<MRayDataEnum E>
    using Type = typename Mapper::Find<E>;

}

namespace MRayPixelDetail
{
    using M = TypeFinder::E_TMapper<MRayPixelEnum>;
    using Mapper = M::Map
    <
        typename M:: template ETPair<MRayPixelEnum::MR_R8_UNORM,         uint8_t>,
        typename M:: template ETPair<MRayPixelEnum::MR_RG8_UNORM,        Vector2uc>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGB8_UNORM,       Vector3uc>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGBA8_UNORM,      Vector4uc>,

        typename M:: template ETPair<MRayPixelEnum::MR_R16_UNORM,        uint16_t>,
        typename M:: template ETPair<MRayPixelEnum::MR_RG16_UNORM,       Vector2us>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGB16_UNORM,      Vector3us>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGBA16_UNORM,     Vector4us>,

        typename M:: template ETPair<MRayPixelEnum::MR_R8_SNORM,         int8_t>,
        typename M:: template ETPair<MRayPixelEnum::MR_RG8_SNORM,        Vector2c>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGB8_SNORM,       Vector3c>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGBA8_SNORM,      Vector4c>,

        typename M:: template ETPair<MRayPixelEnum::MR_R16_SNORM,        int16_t>,
        typename M:: template ETPair<MRayPixelEnum::MR_RG16_SNORM,       Vector2s>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGB16_SNORM,      Vector3s>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGBA16_SNORM,     Vector4s>,

        // TODO: Find a proper half lib and change this
        typename M:: template ETPair<MRayPixelEnum::MR_R_HALF,           uint16_t>,
        typename M:: template ETPair<MRayPixelEnum::MR_RG_HALF,          Vector2us>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGB_HALF,         Vector3us>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGBA_HALF,        Vector4us>,

        typename M:: template ETPair<MRayPixelEnum::MR_R_FLOAT,          float>,
        typename M:: template ETPair<MRayPixelEnum::MR_RG_FLOAT,         Vector2f>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGB_FLOAT,        Vector3f>,
        typename M:: template ETPair<MRayPixelEnum::MR_RGBA_FLOAT,       Vector4f>,

        typename M:: template ETPair<MRayPixelEnum::MR_BC1_UNORM,        PixelBC1>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC2_UNORM,        PixelBC2>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC3_UNORM,        PixelBC3>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC4_UNORM,        PixelBC4U>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC4_SNORM,        PixelBC4S>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC5_UNORM,        PixelBC5U>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC5_SNORM,        PixelBC5S>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC6H_UFLOAT,      PixelBC6U>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC6H_SFLOAT,      PixelBC6S>,
        typename M:: template ETPair<MRayPixelEnum::MR_BC7_UNORM,        PixelBC7>
    >;

    template<MRayPixelEnum E>
    using Type = typename Mapper::Find<E>;

}

template<MRayDataEnum E>
struct MRayDataType
{
    using Type = MRayDataDetail::Type<E>;
    static constexpr size_t Size       = sizeof(Type);
    static constexpr size_t Alignment  = alignof(Type);
    static constexpr MRayDataEnum Name = E;
};

template<MRayPixelEnum E>
struct MRayPixelType
{
    private:
    template<class T>
    requires (std::is_integral_v<T> || std::is_floating_point_v<T>)
    static constexpr size_t FindChannelCount();
    template<class T>
    requires (!std::is_integral_v<T> && !std::is_floating_point_v<T>)
    static constexpr size_t FindChannelCount();
    template<class T>
    requires (std::is_integral_v<T> || std::is_floating_point_v<T>)
    static constexpr bool FindIsSigned();
    template<class T>
    requires (!std::is_integral_v<T> && !std::is_floating_point_v<T>)
    static constexpr bool FindIsSigned();

    public:
    using Type = MRayPixelDetail::Type<E>;
    static constexpr size_t ChannelCount = FindChannelCount<Type>();
    static constexpr MRayPixelEnum Name = E;
    static constexpr bool IsBCPixel = IsBlockCompressedPixel<Type>;
    static constexpr bool IsSigned = FindIsSigned<Type>();
    static constexpr size_t PixelSize = sizeof(Type);
    static constexpr size_t PaddedPixelSize = (ChannelCount != 3)
                                                ? PixelSize
                                                : PixelSize / 3 * 4;
};

template<MRayPixelEnum E>
template<class T>
requires (std::is_integral_v<T> || std::is_floating_point_v<T>)
constexpr size_t MRayPixelType<E>::FindChannelCount()
{
    return 1u;
}

template<MRayPixelEnum E>
template<class T>
requires (!std::is_integral_v<T> && !std::is_floating_point_v<T>)
constexpr size_t MRayPixelType<E>::FindChannelCount()
{
    return Type::Dims;
}

template<MRayPixelEnum E>
template<class T>
requires (std::is_integral_v<T> || std::is_floating_point_v<T>)
constexpr bool MRayPixelType<E>::FindIsSigned()
{
    return std::is_signed_v<T>;
}

template<MRayPixelEnum E>
template<class T>
requires (!std::is_integral_v<T> && !std::is_floating_point_v<T>)
constexpr bool MRayPixelType<E>::FindIsSigned()
{
    return std::is_signed_v<typename T::InnerType>;
}

namespace DataTypeDetail
{

template<uint32_t I>
using UIntTConst = std::integral_constant<uint32_t, I>;

template<class EnumType, template<auto> class DataType, class Func, uint32_t O>
auto StampIfOverEnum(UIntTConst<O>, Func&& F, unsigned int name) -> decltype(auto)
{
    constexpr uint32_t STAMP_COUNT = 16;
    constexpr uint32_t VSize = uint32_t(EnumType::MR_END);
    [[maybe_unused]] int invokeCount = 0;
    // TODO: Check this at godbolt for code generation.
    #define COND_INVOKE(I)                                  \
        if constexpr(MRAY_IS_DEBUG) invokeCount++;          \
        if constexpr(VSize > (I + O))                       \
        if(name == O + I)                                   \
        {                                                   \
            return std::invoke                              \
            (                                               \
                std::forward<Func>(F),                      \
                DataType<static_cast<EnumType>(I + O)>{}    \
            );                                              \
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
    assert(invokeCount == STAMP_COUNT && "Invalid Visit implementation, "
            "add more\"COND_INVOKE\"s");
    if constexpr(VSize > O + STAMP_COUNT)
        return StampIfOverEnum<EnumType, DataType>
        (
            UIntTConst<O + STAMP_COUNT>{},
            std::forward<Func>(F), name
        );
    MRAY_UNREACHABLE;
}

template<class EnumType, template<auto> class DataType, class Func>
auto SwitchCaseAsIfStatements(Func&& F, unsigned int name) -> decltype(auto)
{
    return StampIfOverEnum<EnumType, DataType>
    (
        UIntTConst<0u>{},
        std::forward<Func>(F), name
    );
}

}

struct MRayDataTypeRT
{
    private:
    MRayDataEnum name;

    public:
    using enum MRayDataEnum;

    constexpr MRayDataEnum  Name() const;
    size_t                  Size() const;
    size_t                  Alignment() const;

    constexpr           MRayDataTypeRT() = default;
    constexpr explicit  MRayDataTypeRT(MRayDataEnum e);

    //
    template<class Func>
    constexpr auto SwitchCase(Func&&) const -> decltype(auto);
    constexpr bool operator==(MRayDataTypeRT other) const;
};

struct MRayPixelTypeRT
{
    private:
    MRayPixelEnum name;

    public:
    using enum MRayPixelEnum;

    constexpr MRayPixelEnum Name() const;
    size_t                  ChannelCount() const;
    bool                    IsBlockCompressed() const;
    bool                    IsSigned() const;
    size_t                  PixelSize() const;
    size_t                  PaddedPixelSize() const;

    // Constructors & Destructor
    constexpr           MRayPixelTypeRT() = default;
    constexpr explicit  MRayPixelTypeRT(MRayPixelEnum e);

    template<class Func>
    constexpr auto SwitchCase(Func&&) const -> decltype(auto);
    constexpr bool operator==(MRayPixelTypeRT other) const;
};

constexpr MRayDataTypeRT::MRayDataTypeRT(MRayDataEnum e)
    : name(e)
{}

constexpr MRayDataEnum MRayDataTypeRT::Name() const
{
    return name;
}

template<class Func>
constexpr auto MRayDataTypeRT::SwitchCase(Func&& F) const -> decltype(auto)
{
    assert(name < MRayDataEnum::MR_END);
    return DataTypeDetail::SwitchCaseAsIfStatements<MRayDataEnum, MRayDataType>
    (
        std::forward<Func>(F),
        static_cast<unsigned int>(name)
    );
}

constexpr bool MRayDataTypeRT::operator==(MRayDataTypeRT other) const
{
    return name == other.name;
}

constexpr MRayPixelTypeRT::MRayPixelTypeRT(MRayPixelEnum e)
    : name(e)
{}

constexpr MRayPixelEnum MRayPixelTypeRT::Name() const
{
    return name;
}

template<class Func>
constexpr auto MRayPixelTypeRT::SwitchCase(Func&& F) const -> decltype(auto)
{
    assert(name < MRayPixelEnum::MR_END);
    return DataTypeDetail::SwitchCaseAsIfStatements<MRayPixelEnum, MRayPixelType>
    (
        std::forward<Func>(F),
        static_cast<unsigned int>(name)
    );
}

constexpr bool MRayPixelTypeRT ::operator==(MRayPixelTypeRT other) const
{
    return name == other.name;
}
