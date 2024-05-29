#pragma once

#include <array>
#include <string_view>
#include "TracerI.h"

namespace TypeNameGen
{

namespace Detail
{
    // From here
    // https://stackoverflow.com/questions/38955940/how-to-concatenate-static-strings-at-compile-time
    template <const std::string_view&... Strs>
    struct ct_str_join
    {
        // Join all strings into a single std::array of chars
        static constexpr auto impl() noexcept
        {
            constexpr std::size_t len = (Strs.size() + ... + 0);
            std::array<char, len + 1> arr{};
            auto append = [i = 0u, &arr](auto const& s) mutable {
                for(auto c : s) arr[i++] = c;
            };
            (append(Strs), ...);
            arr[len] = 0;
            return arr;
        }
        // Give the joined string static storage
        static constexpr auto Arr = impl();
        // View as a std::string_view
        static constexpr std::string_view value{Arr.data(), Arr.size() - 1};
    };
    // Helper to get the value out
    template <const std::string_view&... Strs>
    static constexpr auto ct_str_join_v = ct_str_join<Strs...>::value;

    using namespace std::string_view_literals;
    static constexpr auto PRIM_SV = "Prim"sv;
}

namespace CompTime
{
    using namespace std::string_view_literals;
    using namespace TracerConstants;

    template <const std::string_view& Name>
    static constexpr
    std::string_view PrimTypeName = Detail::ct_str_join_v<PRIM_PREFIX, Name>;

    template <const std::string_view& Name>
    static constexpr
    std::string_view TransformTypeName = Detail::ct_str_join_v<TRANSFORM_PREFIX, Name>;

    template <const std::string_view& Name>
    static constexpr
    std::string_view MaterialTypeName = Detail::ct_str_join_v<MAT_PREFIX, Name>;

    template <const std::string_view& Name>
    static constexpr
    std::string_view CameraTypeName = Detail::ct_str_join_v<CAM_PREFIX, Name>;

    template <const std::string_view& Name>
    static constexpr
    std::string_view MediumTypeName = Detail::ct_str_join_v<MEDIUM_PREFIX, Name>;

    template <const std::string_view& Name>
    static constexpr
    std::string_view RendererTypeName = Detail::ct_str_join_v<RENDERER_PREFIX, Name>;

    template <const std::string_view& Name>
    static constexpr
    std::string_view LightTypeName = Detail::ct_str_join_v<LIGHT_PREFIX, Name>;

    // Hard to make these two constexpr
    static std::string PrimLightTypeName(std::string_view PrimName)
    {
        return (std::string(LIGHT_PREFIX) +
                std::string(Detail::PRIM_SV) +
                std::string(PrimName));
    };

    template <const std::string_view& Name>
    static constexpr
    std::string_view BaseAccelTypeName = Detail::ct_str_join_v<ACCEL_PREFIX, Name>;

    static std::string AccelGroupTypeName(std::string_view BaseAccelName,
                                          std::string_view PrimName)
    {
        return std::string(BaseAccelName) + std::string(PrimName);
    }

    static std::string AccelWorkTypeName(const std::string_view& AccelGroupName,
                                         const std::string_view& TransformName)
    {
        return std::string(AccelGroupName) + std::string(TransformName);
    }
}

namespace Runtime
{

    inline
    std::string AddPrimitivePrefix(std::string_view primType)
    {
        return (std::string(TracerConstants::PRIM_PREFIX) +
                std::string(primType));
    }

    inline
    std::string AddLightPrefix(std::string_view lightType)
    {
        return (std::string(TracerConstants::LIGHT_PREFIX) +
                std::string(lightType));
    }

    inline
    std::string AddTransformPrefix(std::string_view transformType)
    {
        return (std::string(TracerConstants::TRANSFORM_PREFIX) +
                std::string(transformType));
    }

    inline
    std::string AddMaterialPrefix(std::string_view matType)
    {
        return (std::string(TracerConstants::MAT_PREFIX) +
                std::string(matType));
    }

    inline
    std::string AddCameraPrefix(std::string_view camType)
    {
        return (std::string(TracerConstants::CAM_PREFIX) +
                std::string(camType));
    }

    inline
    std::string AddMediumPrefix(std::string_view medType)
    {
        return (std::string(TracerConstants::MEDIUM_PREFIX) +
                std::string(medType));
    }

    inline
    std::string CreatePrimBackedLightTypeName(std::string_view primType)
    {
        using namespace std::string_literals;
        return ("Prim"s + std::string(TracerConstants::PRIM_PREFIX) +
                std::string(primType));
    }

    inline
    bool IsPrimBackedLightType(std::string_view lightTypeName)
    {
        using namespace std::string_view_literals;
        return lightTypeName.find("(L)Prim"sv) != std::string_view::npos;
    }

    inline
    std::string CreateAcceleratorType(std::string_view accelType,
                                      std::string_view primType)
    {
        using namespace std::string_literals;
        return (std::string(TracerConstants::ACCEL_PREFIX) +
                std::string(accelType) +
                std::string(TracerConstants::PRIM_PREFIX) +
                std::string(primType));
    }

    inline
    std::string CreateAcceleratorInstanceType(std::string_view accelType,
                                              std::string_view transType)
    {
        using namespace std::string_literals;
        return (std::string(accelType) +
                std::string(TracerConstants::TRANSFORM_PREFIX) +
                std::string(transType));
    }

}
}