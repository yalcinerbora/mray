#pragma once

#include "Core/Types.h"
#include "ParamVaryingData.h"
#include "TransformC.h"
#include "LightC.h"

#include "Device/GPUSystem.h"
#include "Device/GPUSystem.hpp"

// Meta Light Class
// This will be used for routines that requires
// every light type at hand. (The example for this is the NEE
// routines that will sample a light source from the lights all over the scene).
// For that case either we would require virtual polymorphism, or a variant
// type/visit. This class utilizes the second one.
// All these functions should decay to "switch case" / "if else"
// statements.
template<class CommonHitT, class MetaLightT,
         class SpectrumTransformer = SpectrumConverterContextIdentity>
class MetaLightViewT
{
    using SpectrumConverter = typename SpectrumTransformer::Converter;
    private:
    const MetaLightT&        light;
    const SpectrumConverter& sConverter;

    public:
    MRAY_HYBRID         MetaLightViewT(const SpectrumConverter& sTransContext,
                                       const MetaLightT&);

    MRAY_HYBRID
    SampleT<Vector3>    SampleSolidAngle(RNGDispenser& dispenser,
                                         const Vector3& distantPoint) const;
    MRAY_HYBRID
    Float               PdfSolidAngle(const CommonHitT& hit,
                                      const Vector3& distantPoint,
                                      const Vector3& dir) const;
    MRAY_HYBRID
    uint32_t            SampleSolidAngleRNCount() const;
    MRAY_HYBRID
    SampleT<Ray>        SampleRay(RNGDispenser& dispenser) const;
    MRAY_HYBRID
    Float               PdfRay(const Ray&) const;
    MRAY_HYBRID
    uint32_t            SampleRayRNCount() const;

    MRAY_HYBRID
    Spectrum            EmitViaHit(const Vector3& wO,
                                   const CommonHitT& hit) const;
    MRAY_HYBRID
    Spectrum            EmitViaSurfacePoint(const Vector3& wO,
                                            const Vector3& surfacePoint) const;

    MRAY_HYBRID bool    IsPrimitiveBackedLight() const;
};

template<class... >
class MetaLightArray;

// Specialize the array
template<TransformContextC... TContexts, LightC... Lights>
class MetaLightArray<Variant<TContexts...>, Variant<Lights...>>
{
    using LightVariant          = Variant<std::monostate, Lights...>;
    using PrimVariant           = UniqueVariant<std::monostate, typename Lights::Primitive...>;
    using TContextVariant       = Variant<std::monostate, TContexts...>;
    using IdentitySConverter    = typename SpectrumConverterContextIdentity::Converter;

    public:
    using MetaLight             = LightVariant;
    template<class CommonHitT, class SpectrumTransformer = SpectrumConverterContextIdentity>
    using MetaLightView         = MetaLightViewT<CommonHitT, MetaLight, SpectrumTransformer>;

    private:
    const GPUSystem&            system;
    Span<PrimVariant>           dLightPrimitiveList;
    Span<LightVariant>          dLightList;
    Span<TContextVariant>       dTContextList;
    Span<IdentitySConverter>    dSConverter;

    DeviceMemory                memory;

    public:
                                MetaLightArray(const GPUSystem&);

    template<LightGroupC LightGroup, TransformGroupC TransformGroup>
    void                        AddBatch(const LightGroup& lg, const TransformGroup& tg,
                                         const Span<const PrimitiveKey>& primitiveKeys,
                                         const Span<const LightKey>& lightKeys,
                                         const Span<const TransformKey>& transformKeys,
                                         const Vector2ui& batchRange);
    Span<const LightVariant>    Array() const;

};

#include "MetaLight.hpp"
