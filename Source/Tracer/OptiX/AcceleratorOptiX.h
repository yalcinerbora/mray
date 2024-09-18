#pragma once

#include "AcceleratorC.h"
#include "OptiXPTX.h"

#include <optix_host.h>

void OptiXAssert(OptixResult code, const char* file, int line);

#ifdef MRAY_DEBUG
    #define OPTIX_CHECK(func) OptiXAssert((func), __FILE__, __LINE__)
    #define OPTIX_CHECK_ERROR(err) OptiXAssert(err, __FILE__, __LINE__)
    #define OPTIX_LAUNCH_CHECK() \
            CUDA_CHECK(cudaDeviceSynchronize()); \
            CUDA_CHECK(cudaGetLastError())
#else
    #define OPTIX_CHECK(func) func
    #define OPTIX_CHECK_ERROR(err)
    #define OPTIX_LAUNCH_CHECK()
#endif

namespace OptiXAccelDetail
{
    static constexpr OptixModuleCompileOptions MODULE_OPTIONS_OPTIX =
    {
        .maxRegisterCount = 0,
        .optLevel = (MRAY_IS_DEBUG)
            ? OPTIX_COMPILE_OPTIMIZATION_LEVEL_0
            : OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
        .debugLevel = (MRAY_IS_DEBUG)
            ? OPTIX_COMPILE_DEBUG_LEVEL_FULL
            : OPTIX_COMPILE_DEBUG_LEVEL_NONE,
        .boundValues = nullptr,
        .numBoundValues = 0,
        .numPayloadTypes = 0,
        .payloadTypes = nullptr
    };

    static constexpr OptixPipelineCompileOptions PIPELINE_OPTIONS_OPTIX =
    {
        .usesMotionBlur = 0,
        .traversableGraphFlags =
            (OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING |
             // We will need this for SSS
             OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS),
        .numPayloadValues = (sizeof(BackupRNGState) / sizeof(uint32_t) +
                             sizeof(RayIndex)       / sizeof(RayIndex)),
        .numAttributeValues = MetaHit::MaxDim,
        .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
        .pipelineLaunchParamsVariableName = "params",
        // TODO: New OptiX has other fancy types (but in software I gusess?)
        // specialize these instead of implementing our own later.
        .usesPrimitiveTypeFlags = std::bit_cast<uint32_t>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
                                                          OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM),
        .allowOpacityMicromaps = 0
    };
    static_assert(PIPELINE_OPTIONS_OPTIX.numPayloadValues == 2,
                  "Currently OptiX shaders are compiled with "
                  "single word for index and another sigle word for backup RNG state. "
                  "Type sizes do not match with this assumption.");

    // Phony type to satisfy the concept
    template<PrimitiveGroupC PrimGroup,
             TransformGroupC TransGroup = TransformGroupIdentity>
    class AcceleratorOptiX
    {
        public:
        using PrimHit       = typename PrimGroup::Hit;
        using PrimDataSoA   = typename PrimGroup::DataSoA;
        using TransDataSoA  = typename TransGroup::DataSoA;
        using HitResult     = HitResultT<PrimHit>;
        using DataSoA       = EmptyType;

        public:
        MRAY_GPU MRAY_GPU_INLINE
        AcceleratorOptiX(TransDataSoA, PrimDataSoA,
                         DataSoA, AcceleratorKey) {}

        MRAY_GPU MRAY_GPU_INLINE
        Optional<HitResult>
        ClosestHit(BackupRNG& rng, const Ray&, const Vector2&) const
        { return std::nullopt; }

        MRAY_GPU MRAY_GPU_INLINE
        Optional<HitResult>
        FirstHit(BackupRNG& rng, const Ray&, const Vector2&) const
        { return std::nullopt; }

        MRAY_GPU MRAY_GPU_INLINE
        TransformKey
        GetTransformKey() const { return TransformKey::InvalidKey(); }
    };
}

struct ComputeCapabilityTypePackOptiX
{
    OptixModule         optixModule = nullptr;
    OptixPipeline       pipeline    = nullptr;
    std::string         computeCapability;

    // Constructors & Destructor
    ComputeCapabilityTypePackOptiX(const std::string&);
    ComputeCapabilityTypePackOptiX(const ComputeCapabilityTypePackOptiX&) = delete;
    ComputeCapabilityTypePackOptiX(ComputeCapabilityTypePackOptiX&&) noexcept;

    ComputeCapabilityTypePackOptiX&
    operator=(const ComputeCapabilityTypePackOptiX&) = delete;

    ComputeCapabilityTypePackOptiX&
    operator=(ComputeCapabilityTypePackOptiX&&) noexcept;

    ~ComputeCapabilityTypePackOptiX();

    auto operator<=>(const ComputeCapabilityTypePackOptiX& right) const;
};

class ContextOptiX
{
    private:
    OptixDeviceContext optixContext;

    public:
                    ContextOptiX();
                    ContextOptiX(const ContextOptiX&) = delete;
    ContextOptiX&   operator=(const ContextOptiX&) = delete;
                    ~ContextOptiX();

    operator OptixDeviceContext() const;
    operator OptixDeviceContext();

};

template<PrimitiveGroupC PrimitiveGroupType>
class AcceleratorGroupOptiX final
    : public AcceleratorGroupT<AcceleratorGroupOptiX<PrimitiveGroupType>, PrimitiveGroupType>
{
    using Base = AcceleratorGroupT<AcceleratorGroupOptiX <PrimitiveGroupType>, PrimitiveGroupType>;

    public:
    static std::string_view TypeName();
    using PrimitiveGroup    = PrimitiveGroupType;
    using DataSoA           = EmptyType;

    template<class TG = TransformGroupIdentity>
    using Accelerator = OptiXAccelDetail::AcceleratorOptiX<PrimitiveGroup, TG>;

    static constexpr auto TransformLogic = PrimitiveGroup::TransformLogic;
    //static constexpr auto IsTriangle = PrimitiveGroup::IsTriangleGroup;

    private:
    OptixDeviceContext optixContext;

    public:
    // Constructors & Destructor
    AcceleratorGroupOptiX(uint32_t accelGroupId,
                          BS::thread_pool&,
                          const GPUSystem&,
                          const GenericGroupPrimitiveT& pg,
                          const AccelWorkGenMap&);
    //
    void    Construct(AccelGroupConstructParams, const GPUQueue&) override;
    void    WriteInstanceKeysAndAABBs(Span<AABB3> dAABBWriteRegion,
                                      Span<AcceleratorKey> dKeyWriteRegion,
                                      const GPUQueue&) const override;

    // Functionality
    void    CastLocalRays(// Output
                          Span<HitKeyPack> dHitIds,
                          Span<MetaHit> dHitParams,
                          // I-O
                          Span<BackupRNGState> dRNGStates,
                          Span<RayGMem> dRays,
                          // Input
                          Span<const RayIndex> dRayIndices,
                          Span<const CommonKey> dAccelKeys,
                          // Constants
                          uint32_t workId,
                          const GPUQueue& queue) override;

    DataSoA SoA() const;
    size_t  GPUMemoryUsage() const override;
};

class BaseAcceleratorOptiX final : public BaseAcceleratorT<BaseAcceleratorOptiX>
{
    public:
    static std::string_view TypeName();
    private:
    ContextOptiX optixContext;

    std::vector<ComputeCapabilityTypePackOptiX> optixTypesPerCC;

    DeviceMemory            accelMem;
    Span<ArgumentPackOpitX> dLaunchArgPack;
    OptixShaderBindingTable commonSBT;

    OptixTraversableHandle  baseAccelerator;

    protected:
    AABB3 InternalConstruct(const std::vector<size_t>& instanceOffsets) override;

     public:
    // Constructors & Destructor
     BaseAcceleratorOptiX(BS::thread_pool&, const GPUSystem&,
                          const AccelGroupGenMap&,
                          const AccelWorkGenMap&);

    //
    void    CastRays(// Output
                     Span<HitKeyPack> dHitIds,
                     Span<MetaHit> dHitParams,
                     // I-O
                     Span<BackupRNGState> dRNGStates,
                     Span<RayGMem> dRays,
                     // Input
                     Span<const RayIndex> dRayIndices,
                     const GPUQueue& queue) override;

    void    CastShadowRays(// Output
                           Bitspan<uint32_t> dIsVisibleBuffer,
                           Bitspan<uint32_t> dFoundMediumInterface,
                           // I-O
                           Span<BackupRNGState> dRNGStates,
                           // Input
                           Span<const RayIndex> dRayIndices,
                           Span<const RayGMem> dShadowRays,
                           const GPUQueue& queue) override;

    void    CastLocalRays(// Output
                          Span<HitKeyPack> dHitIds,
                          Span<MetaHit> dHitParams,
                          // I-O
                          Span<BackupRNGState> dRNGStates,
                          // Input
                          Span<const RayGMem> dRays,
                          Span<const RayIndex> dRayIndices,
                          Span<const AcceleratorKey> dAccelIdPacks,
                          const GPUQueue& queue) override;

    void    Construct(BaseAccelConstructParams) override;
    void    AllocateForTraversal(size_t maxRayCount) override;
    size_t  GPUMemoryUsage() const override;
};

inline ContextOptiX::operator OptixDeviceContext() const
{
    return optixContext;
}

inline ContextOptiX::operator OptixDeviceContext()
{
    return optixContext;
}

#include "AcceleratorOptiX.hpp"