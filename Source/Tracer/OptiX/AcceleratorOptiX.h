#pragma once

#include "AcceleratorC.h"
#include "OptiXPTX.h"

#include <optix_host.h>
#include <optix_stubs.h>

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
    struct ShaderTypeNames
    {
        std::string_view    primName;
        std::string_view    transformName;
        bool                isTriangle = false;

        auto operator<=>(const ShaderTypeNames& t) const;
    };

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

    static constexpr OptixAccelBuildOptions BUILD_OPTIONS_OPTIX =
    {
        .buildFlags = std::bit_cast<uint32_t>(OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                                              OPTIX_BUILD_FLAG_PREFER_FAST_TRACE),
        .operation = OPTIX_BUILD_OPERATION_BUILD,
        .motionOptions = OptixMotionOptions{ 1 }
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
    OptixModule                     optixModule     = nullptr;
    std::vector<OptixProgramGroup>  programGroups;
    OptixPipeline                   pipeline        = nullptr;
    std::string                     computeCapability;

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
    OptixDeviceContext contextOptiX = nullptr;

    public:
                    ContextOptiX();
                    ContextOptiX(const ContextOptiX&) = delete;
    ContextOptiX&   operator=(const ContextOptiX&) = delete;
                    ~ContextOptiX();

    operator OptixDeviceContext() const;
    operator OptixDeviceContext();

};

// Add Optix specific functions
class AcceleratorGroupOptixI : public AcceleratorGroupI
{
    public:
    virtual ~AcceleratorGroupOptixI() = default;

    virtual void AcquireIASConstructionParams(Span<OptixTraversableHandle> dTraversableHandles,
                                              Span<Matrix4x4> dInstanceMatrices,
                                              Span<uint32_t> dSBTCounts,
                                              Span<uint32_t> dFlags,
                                              const GPUQueue& queue) const = 0;
    //
    virtual std::vector<OptiXAccelDetail::ShaderTypeNames>
    GetShaderTypeNames() const = 0;
    //
    virtual std::vector<GenericHitRecord<>>
    GetHitRecords() const = 0;

    virtual std::vector<uint32_t>
    GetShaderOffsets() const = 0;

    virtual void OffsetAccelKeyInRecords() = 0;
};

template<PrimitiveGroupC PrimitiveGroupType>
class AcceleratorGroupOptiX final
    : public AcceleratorGroupT<AcceleratorGroupOptiX<PrimitiveGroupType>, PrimitiveGroupType, AcceleratorGroupOptixI>
{
    using Base = AcceleratorGroupT<AcceleratorGroupOptiX <PrimitiveGroupType>,
                                   PrimitiveGroupType, AcceleratorGroupOptixI>;
    using HitRecordVector = std::vector<GenericHitRecord<>>;
    public:
    static std::string_view TypeName();
    using PrimitiveGroup    = PrimitiveGroupType;
    using DataSoA           = EmptyType;
    using PGSoA             = typename PrimitiveGroup::DataSoA;

    template<class TG = TransformGroupIdentity>
    using Accelerator = OptiXAccelDetail::AcceleratorOptiX<PrimitiveGroup, TG>;

    static constexpr auto TransformLogic = PrimitiveGroup::TransformLogic;
    static constexpr auto IsTriangle = TrianglePrimGroupC<PrimitiveGroup>;

    private:
    OptixDeviceContext      contextOptiX = nullptr;
    DeviceMemory            memory;
    Span<Byte>              dAllAccelerators;
    Span<PrimitiveKey>      dAllLeafs;
    Span<TransformKey>      dTransformKeys;
    Span<PGSoA>             dPrimGroupSoA;
    Span<Byte>              dTransformGroupSoAList;
    // Host side
    std::vector<uint32_t>   hConcreteHitRecordCounts;
    std::vector<Vector2ui>  hConcreteHitRecordPrimRanges;
    std::vector<size_t>     hTransformSoAOffsets;
    HitRecordVector         hHitRecords;
    // Instance Related
    std::vector<OptixTraversableHandle> hInstanceAccelHandles;
    std::vector<uint32_t>               hInstanceHitRecordCounts;
    std::vector<uint32_t>               hInstanceCommonFlags;

    std::vector<OptixTraversableHandle>
    MultiBuildTriangleCLT(const PreprocessResult& ppResult,
                          const GPUQueue& queue);
    std::vector<OptixTraversableHandle>
    MultiBuildTrianglePPT(const PreprocessResult& ppResult,
                          const GPUQueue& queue);
    std::vector<OptixTraversableHandle>
    MultiBuildGenericCLT(const PreprocessResult& ppResult,
                         const GPUQueue& queue);
    std::vector<OptixTraversableHandle>
    MultiBuildGenericPPT(const PreprocessResult& ppResult,
                         const GPUQueue& queue);

    public:
    // Constructors & Destructor
    AcceleratorGroupOptiX(uint32_t accelGroupId,
                          BS::thread_pool&,
                          const GPUSystem&,
                          const GenericGroupPrimitiveT& pg,
                          const AccelWorkGenMap&);
    //
    void    PreConstruct(const BaseAcceleratorI*) override;
    void    Construct(AccelGroupConstructParams, const GPUQueue&) override;
    void    WriteInstanceKeysAndAABBs(Span<AABB3> dAABBWriteRegion,
                                      Span<AcceleratorKey> dKeyWriteRegion,
                                      const GPUQueue&) const override;
    void    AcquireIASConstructionParams(Span<OptixTraversableHandle> dTraversableHandles,
                                         Span<Matrix4x4> dInstanceMatrices,
                                         Span<uint32_t> dSBTCounts,
                                         Span<uint32_t> dFlags,
                                         const GPUQueue& queue) const override;
    std::vector<OptiXAccelDetail::ShaderTypeNames>
            GetShaderTypeNames() const override;

    std::vector<GenericHitRecord<>>
            GetHitRecords() const override;
    std::vector<uint32_t>
            GetShaderOffsets() const override;
    void    OffsetAccelKeyInRecords() override;

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
    using ShaderNameMap = std::map<OptiXAccelDetail::ShaderTypeNames, std::vector<uint32_t>>;
    static std::string_view TypeName();
    private:
    ContextOptiX                contextOptiX;
    std::vector<ComputeCapabilityTypePackOptiX> optixTypesPerCC;
    // Memory related
    DeviceMemory                allMem;
    Span<ArgumentPackOpitX>     dLaunchArgPack;
    Span<Byte>                  dAccelMemory;
    Span<GenericHitRecord<>>    dHitRecords;
    Span<EmptyHitRecord>        dEmptyRecords;
    // State of the CC
    uint32_t currentCCIndex     = std::numeric_limits<uint32_t>::max();
    //

    // Host
    OptixShaderBindingTable commonSBT;
    OptixTraversableHandle  baseAccelerator;

    protected:
    AABB3           InternalConstruct(const std::vector<size_t>& instanceOffsets) override;
    void GenerateShaders(EmptyHitRecord& rgRecord, EmptyHitRecord& missRecord,
                         std::vector<GenericHitRecord<>>&, const ShaderNameMap&);
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

    void    AllocateForTraversal(size_t maxRayCount) override;
    size_t  GPUMemoryUsage() const override;
    // OptiX special

    OptixDeviceContext GetOptixDeviceHandle() const;
};

inline ContextOptiX::operator OptixDeviceContext() const
{
    return contextOptiX;
}

inline ContextOptiX::operator OptixDeviceContext()
{
    return contextOptiX;
}

#include "AcceleratorOptiX.hpp"