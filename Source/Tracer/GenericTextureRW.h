#pragma once

#include "Device/GPUTextureView.h"  // IWYU pragma: keep
#include "TextureCommon.h"
#include "SurfaceView.h"

MR_GF_DECL
Vector4 GenericRead(const Vector2ui& pixCoords,
                    const TracerSurfView& surf);

MR_GF_DECL
Vector4 GenericReadFromBuffer(const Span<const Byte>& dBufferImage,
                              const TracerSurfView& surfToVisit,
                              uint32_t pixCoordLinear);

MR_GF_DECL
void GenericWrite(TracerSurfView& surf, const Vector4& value,
                  const Vector2ui& pixCoords);

MR_GF_DEF
Vector4 GenericRead(const Vector2ui& pixCoords,
                    const TracerSurfView& surf)
{
    Vector4 v = Visit
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
                    MRAY_UNROLL_LOOP
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

MR_GF_DEF
Vector4 GenericReadFromBuffer(const Span<const Byte>& dBufferImage,
                              const TracerSurfView& surfToVisit,
                              uint32_t pixCoordLinear)
{
    return Visit
    (
        surfToVisit,
        [pixCoordLinear, dBufferImage](auto&& s) -> Vector4
        {
            Vector4 out = Vector4::Zero();
            using VariantType = std::remove_cvref_t<decltype(s)>;
            // Skip monostate
            if constexpr(!std::is_same_v<VariantType, std::monostate>)
            {
                // Data is in padded channel type
                using ReadType = typename VariantType::PaddedChannelType;
                const ReadType* dPtr = reinterpret_cast<const ReadType*>(dBufferImage.data());
                dPtr = std::launder(dPtr);
                constexpr uint32_t C = VariantType::Channels;
                if constexpr(C != 1)
                {
                    // Somewhat hard part, first read the data
                    // then convert
                    ReadType data = dPtr[pixCoordLinear];
                    MRAY_UNROLL_LOOP
                    for(uint32_t c = 0; c < C; c++)
                        out[c] = static_cast<Float>(data[c]);
                }
                // Easy, directly subscript the pointer and cast
                else out[0] = static_cast<Float>(dPtr[pixCoordLinear]);
            }
            return out;
        }
    );
}

MR_GF_DEF
void GenericWrite(TracerSurfView& surf,
                  const Vector4& value,
                  const Vector2ui& pixCoords)
{
    Visit
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
                    MRAY_UNROLL_LOOP
                    for(uint32_t c = 0; c < C; c++)
                    {
                        Float v = value[c];
                        if constexpr(std::is_integral_v<InnerT>)
                        {
                            using Math::Clamp;
                            v = Clamp(Math::Round(v),
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
                        using Math::Clamp;
                        v = Clamp(Math::Round(v),
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