#pragma once

#include "Device/GPUSystem.h"
#include "Core/DeviceVisit.h"
#include "TextureCommon.h"

MRAY_GPU MRAY_GPU_INLINE
Vector4 GenericRead(const Vector2ui& pixCoords,
                    const SurfViewVariant& surf)
{
    Vector4 v = DeviceVisit
    (
        std::as_const(surf),
        // Visitor
        [pixCoords](auto&& readSurf) -> Vector4
        {
            Vector4 result = Vector4::Zero();
            using VariantType = std::remove_cvref_t<decltype(readSurf)>;
            if constexpr(!std::is_same_v<VariantType, std::monostate>)
            {
                using ReadType = typename VariantType::Type;
                constexpr uint32_t C = VariantType::Channels;
                ReadType rPix = readSurf(pixCoords);
                // We need to manually loop over the channels here
                if constexpr(C != 1)
                {
                    UNROLL_LOOP
                    for(uint32_t c = 0; c < C; c++)
                        result[c] = static_cast<Float>(rPix[c]);
                }
                else result[0] = static_cast<Float>(rPix);
            }
            return result;
        }
    );
    return v;
}

MRAY_GPU MRAY_GPU_INLINE
void GenericWrite(SurfViewVariant& surf,
                  const Vector4& value,
                  const Vector2ui& pixCoords)
{
    DeviceVisit
    (
        surf,
        // Visitor
        [value, pixCoords](auto&& writeSurf) -> void
        {
            using VariantType = std::remove_cvref_t<decltype(writeSurf)>;
            if constexpr(!std::is_same_v<VariantType, std::monostate>)
            {
                using WriteType = typename VariantType::Type;
                constexpr uint32_t C = VariantType::Channels;
                WriteType writeVal;
                if constexpr(C != 1)
                {
                    using InnerT = typename WriteType::InnerType;
                    UNROLL_LOOP
                    for(uint32_t c = 0; c < C; c++)
                    {
                        Float v = value[c];
                        if constexpr(std::is_integral_v<InnerT>)
                        {
                            using MathFunctions::Clamp;
                            v = Clamp(std::round(v),
                                      Float(std::numeric_limits<InnerT>::min()),
                                      Float(std::numeric_limits<InnerT>::max()));
                        }
                        writeVal[c] = static_cast<InnerT>(v);
                    }
                }
                else
                {
                    Float v = value[0];
                    if constexpr(std::is_integral_v<WriteType>)
                    {
                        using MathFunctions::Clamp;
                        v = Clamp(std::round(v),
                                  Float(std::numeric_limits<WriteType>::min()),
                                  Float(std::numeric_limits<WriteType>::max()));
                    }
                    writeVal = static_cast<WriteType>(v);
                }
                writeSurf(pixCoords) = writeVal;
            }
        }
    );
}