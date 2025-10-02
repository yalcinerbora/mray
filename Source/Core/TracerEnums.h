#pragma once

#include "Definitions.h"

class PrimitiveAttributeLogic
{
    public:
    enum E
    {
        POSITION,
        INDEX,
        NORMAL,
        RADIUS,
        TANGENT,
        BITANGENT,
        UV0,
        UV1,
        WEIGHT,
        WEIGHT_INDEX,

        MR_ENUM_END
    };

    private:
    static constexpr std::array<std::string_view, size_t(MR_ENUM_END)> Names =
    {
        "Position",
        "Index",
        "Normal",
        "Radius",
        "Tangent",
        "BiTangent",
        "UV0",
        "UV1",
        "Weight",
        "Weight Index"
    };

    public:
    E e;
    static constexpr std::string_view   ToString(E);
    static constexpr E                  FromString(std::string_view);

    constexpr PrimitiveAttributeLogic(E e) : e(e) {}
    constexpr PrimitiveAttributeLogic& operator=(E eIn) { e = eIn; return *this; }
};

// Accelerators are responsible for accelerating ray/surface interactions
// This is abstracted away but exposed to the user for prototyping different
// accelerators. This is less useful due to hw-accelerated ray tracing.
//
// The old design supported mixing and matching "Base Accelerator"
// (TLAS, IAS on other APIs) with bottom-level accelerators. This
// design only enables the user to set a single accelerator type
// for the entire scene. Hardware acceleration APIs, such asOptiX,
// did not support it anyway so it is removed. Internally software stack
// utilizes the old implementation for simulating two-level acceleration
// hierarchy.
//
// Currently Accepted types are
//
//  -- SOFTWARE_NONE :  No acceleration, ray caster does a !LINEAR SEARCH! over the
//                      primitives (Should not be used, it is for debugging,
//                      testing etc.)
//  -- SOFTWARE_BVH  :  Very basic LBVH. Each triangle's center is converted to
//                      a morton code and these are sorted. Provided for completeness
//                      sake and should not be used.
//  -- HARDWARE      :  On CUDA, it utilizes OptiX for hardware acceleration.
class AcceleratorType
{
    public:
    enum E
    {
        SOFTWARE_NONE,
        SOFTWARE_BASIC_BVH,
        HARDWARE,

        MR_ENUM_END
    };

    private:
    static constexpr std::array<std::string_view, size_t(MR_ENUM_END)> Names =
    {
        "Linear",
        "BVH",
        "Hardware"
    };

    public:
    E e;

    // We use this on a map, so overload less
    bool operator<(AcceleratorType t) const;

    static constexpr std::string_view   ToString(E);
    static constexpr E                  FromString(std::string_view);

    constexpr AcceleratorType(E e) : e(e) {}
    constexpr AcceleratorType& operator=(E eIn) { e = eIn; return *this; }
};

class SamplerType
{
    public:
    enum E
    {
        INDEPENDENT,
        Z_SOBOL,

        MR_ENUM_END
    };

    private:
    static constexpr std::array<std::string_view, size_t(MR_ENUM_END)> Names =
    {
        "Independent",
        "ZSobol"
    };

    public:
    E e;

    static constexpr std::string_view   ToString(E);
    static constexpr E                  FromString(std::string_view);

    constexpr SamplerType(E e) : e(e) {}
    constexpr SamplerType& operator=(E eIn) { e = eIn; return *this; }
};

// TODO: Move this to a user facing system (user can set this via config etc.
class WavelengthSampleMode
{
    public:
    enum E
    {
        UNIFORM,
        GAUSSIAN_MIS,
        HYPERBOLIC_PBRT,

        MR_ENUM_END
    };
    private:
    static constexpr std::array<std::string_view, size_t(MR_ENUM_END)> Names =
    {
        "Uniform",
        "GaussianMIS",
        "HyperbolicPBRT"
    };

    public:
    E e;

    static constexpr std::string_view   ToString(E);
    static constexpr E                  FromString(std::string_view);

    constexpr WavelengthSampleMode(E e) : e(e) {}
    constexpr WavelengthSampleMode& operator=(E eIn) { e = eIn; return *this; }
};

class FilterType
{
    public:
    enum E
    {
        BOX,
        TENT,
        GAUSSIAN,
        MITCHELL_NETRAVALI,

        MR_ENUM_END
    };

    private:
    static constexpr std::array<std::string_view, size_t(MR_ENUM_END)> Names =
    {
        "Box",
        "Tent",
        "Gaussian",
        "Mitchell-Netravali"
    };

    public:
    E       type;
    Float   radius;

    static constexpr auto TYPE_NAME = "type";
    static constexpr auto RADIUS_NAME = "radius";

    static constexpr std::string_view   ToString(E);
    static constexpr E                  FromString(std::string_view);
};

enum class AttributeOptionality : uint8_t
{
    MR_MANDATORY,
    MR_OPTIONAL
};

enum class AttributeTexturable : uint8_t
{
    MR_CONSTANT_ONLY,
    MR_TEXTURE_OR_CONSTANT,
    MR_TEXTURE_ONLY
};

enum class AttributeIsColor : uint8_t
{
    IS_COLOR,
    IS_PURE_DATA
};

enum class AttributeIsArray : uint8_t
{
    IS_SCALAR,
    IS_ARRAY
};

// We use this on a map, so overload less
inline bool AcceleratorType::operator<(AcceleratorType t) const
{
    return e < t.e;
}

constexpr
std::string_view AcceleratorType::ToString(AcceleratorType::E e)
{
    assert(e >= 0 && e < uint32_t(MR_ENUM_END));
    return Names[static_cast<uint32_t>(e)];
}

constexpr typename AcceleratorType::E
AcceleratorType::FromString(std::string_view sv)
{
    using IntType = std::underlying_type_t<typename AcceleratorType::E>;
    IntType i = 0;
    for(const std::string_view& checkSV : Names)
    {
        if(checkSV == sv) return AcceleratorType::E(i);
        i++;
    }
    return MR_ENUM_END;
}

constexpr
std::string_view SamplerType::ToString(typename SamplerType::E e)
{
    assert(e >= 0 && e < uint32_t(MR_ENUM_END));
    return Names[static_cast<uint32_t>(e)];
}

constexpr typename SamplerType::E
SamplerType::FromString(std::string_view sv)
{
    using IntType = std::underlying_type_t<typename SamplerType::E>;
    IntType i = 0;
    for(const std::string_view& checkSV : Names)
    {
        if(checkSV == sv) return SamplerType::E(i);
        i++;
    }
    return MR_ENUM_END;
}

constexpr
std::string_view FilterType::ToString(typename FilterType::E e)
{
    assert(e >= 0 && e < uint32_t(MR_ENUM_END));
    return Names[static_cast<uint32_t>(e)];
}

constexpr typename FilterType::E
FilterType::FromString(std::string_view sv)
{
    using IntType = std::underlying_type_t<typename FilterType::E>;
    IntType i = 0;
    for(const std::string_view& checkSV : Names)
    {
        if(checkSV == sv) return FilterType::E(i);
        i++;
    }
    return MR_ENUM_END;
}

constexpr
std::string_view PrimitiveAttributeLogic::ToString(typename PrimitiveAttributeLogic::E e)
{
    assert(e >= 0 && e < uint32_t(MR_ENUM_END));
    return Names[static_cast<uint32_t>(e)];
}

constexpr typename PrimitiveAttributeLogic::E
PrimitiveAttributeLogic::FromString(std::string_view sv)
{
    using IntType = std::underlying_type_t<typename PrimitiveAttributeLogic::E>;
    IntType i = 0;
    for(const std::string_view& checkSV : Names)
    {
        if(checkSV == sv) return PrimitiveAttributeLogic::E(i);
        i++;
    }
    return MR_ENUM_END;
}

constexpr
std::string_view WavelengthSampleMode::ToString(E e)
{
    assert(e >= 0 && e < uint32_t(MR_ENUM_END));
    return Names[static_cast<uint32_t>(e)];
}

constexpr
typename WavelengthSampleMode::E
WavelengthSampleMode::FromString(std::string_view sv)
{
    using IntType = std::underlying_type_t<typename WavelengthSampleMode::E>;
    IntType i = 0;
    for(const std::string_view& checkSV : Names)
    {
        if(checkSV == sv) return WavelengthSampleMode::E(i);
        i++;
    }
    return MR_ENUM_END;
}