#include "AcceleratorOptiX.h"
#include "TransformC.h"

#include <cassert>
#include <map>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <fstream>

#include "Core/System.h"
#include "Core/Expected.h"
#include "Core/Filesystem.h"

#include "Device/GPUAlgScan.h"
#include "Device/GPUSystem.hpp"
#include "Device/GPUAlgReduce.h"
#include "Device/GPUAlgGeneric.h"

// Magic linking, these are populated via CUDA runtime?
#include <optix_host.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCCopyToOptixInstance(// Output
                           MRAY_GRID_CONSTANT const Span<OptixInstance> dInstances,
                           // Input
                           MRAY_GRID_CONSTANT const Span<const Matrix3x4> dMatrices,
                           MRAY_GRID_CONSTANT const Span<const uint32_t> dInstanceFlags,
                           MRAY_GRID_CONSTANT const Span<const OptixTraversableHandle> dHandles,
                           MRAY_GRID_CONSTANT const Span<const uint32_t> dGlobalSBTOffsets,
                           MRAY_GRID_CONSTANT const Span<const AccelInstanceMask> dInstanceMasks)
{
    uint32_t totalInstanceCount = static_cast<uint32_t>(dInstances.size());

    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < totalInstanceCount; i += kp.TotalSize())
    {
        OptixInstance out =
        {
            .transform =
            {
                dMatrices[i][0], dMatrices[i][1], dMatrices[i][ 2], dMatrices[i][ 3],
                dMatrices[i][4], dMatrices[i][5], dMatrices[i][ 6], dMatrices[i][ 7],
                dMatrices[i][8], dMatrices[i][9], dMatrices[i][10], dMatrices[i][11]
            },
            .instanceId = i,
            .sbtOffset = dGlobalSBTOffsets[i],
            .visibilityMask = static_cast<unsigned int>(dInstanceMasks[i].mask),
            .flags = dInstanceFlags[i],
            .traversableHandle = dHandles[i]
        };
        dInstances[i] = out;
    }
}


MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCCopyInstanceMaskInfo(// Output
                            MRAY_GRID_CONSTANT const Span<InstanceMaskOptiX> dInstanceMaskPacks,
                            // Input
                            MRAY_GRID_CONSTANT const Span<const uint32_t> dInstanceFlags,
                            MRAY_GRID_CONSTANT const Span<const AccelInstanceMask> dInstanceMasks)
{
    uint32_t totalInstanceCount = static_cast<uint32_t>(dInstanceMaskPacks.size());

    KernelCallParams kp;
    for(uint32_t i = kp.GlobalId(); i < totalInstanceCount; i += kp.TotalSize())
    {
        InstanceMaskOptiX mask =
        {
            .mask = dInstanceMasks[i],
            .enforceAnyHit = (dInstanceFlags[i] & OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT) > 0
        };

        dInstanceMaskPacks[i] = mask;
    }
}

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCCopyAllAccelKeysOptiX(// Output
                             Span<AcceleratorKey> dAccelKeys,
                             // Input
                             Span<const GenericHitRecord<>> dHitRecords,
                             Span<const uint32_t> dGlobalInstanceSBTOffsets)
{
    KernelCallParams kp;
    uint32_t keyCount = uint32_t(dAccelKeys.size());
    for(uint32_t i = kp.GlobalId(); i < keyCount; i += kp.TotalSize())
    {
        uint32_t offset = dGlobalInstanceSBTOffsets[i];
        const auto& dHR = dHitRecords[offset];
        dAccelKeys[i] = dHR.data.acceleratorKey;
    }
}

static constexpr auto OPTIX_LOGGER_NAME = "OptiXLogger";
static constexpr auto OPTIX_LOGGER_FILE_NAME = "optix_log";
static constexpr auto OPTIX_SHADERS_FOLDER = "OptiXShaders";

#ifdef MRAY_COMPILE_OPTIX_AS_PTX
    static constexpr auto OPTIX_SHADER_NAME = "OptiXPTX.ptx";
#else
    static constexpr auto OPTIX_SHADER_NAME = "OptiXPTX.optixir";
#endif

auto OptiXAccelDetail::ShaderTypeNames::operator<=>(const ShaderTypeNames& t) const
{
    return Tuple(primName, transformName) <=> Tuple(t.primName, t.transformName);
}

Expected<std::vector<char>>
DevourFile(const std::string& shaderName,
           const std::string& executablePath)
{
    std::string fullPath = Filesystem::RelativePathToAbsolute(shaderName,
                                                              executablePath);
    std::ifstream shaderFile = std::ifstream(fullPath, std::ios::binary);

    if(!shaderFile.is_open())
        return MRayError("Unable to open shader file \"{}\"",
                         fullPath);

    std::vector<char> data(std::istreambuf_iterator<char>(shaderFile), {});
    return data;
}

void OptiXAssert(OptixResult code, const char* file, int line)
{
    if(code != OPTIX_SUCCESS)
    {
        MRAY_ERROR_LOG("Optix Failure: {:s} {:s} {:d}",
                       optixGetErrorString(code), file, line);
        assert(false);
    }
}

void OptixLog(unsigned int level, const char* tag, const char* message, void*)
{
    auto logger = spdlog::get(OPTIX_LOGGER_NAME);
    switch(level)
    {
        case 1: logger->error("[FATAL]{:s}, {:s}", tag, message); break;
        case 2: logger->error("{:s}, {:s}", tag, message); break;
        case 3: logger->warn("{:s}, {:s}", tag, message); break;
        case 4: logger->info("{:s}, {:s}", tag, message); break;
    }
    logger->flush();
}

ComputeCapabilityTypePackOptiX::ComputeCapabilityTypePackOptiX(const std::string& s)
    : computeCapability(s)
{}

ComputeCapabilityTypePackOptiX::ComputeCapabilityTypePackOptiX(ComputeCapabilityTypePackOptiX&& other) noexcept
    : optixModule(std::exchange(other.optixModule, nullptr))
    , programGroups(std::exchange(other.programGroups, std::vector<OptixProgramGroup>{}))
    , pipeline(std::exchange(other.pipeline, nullptr))
{}

ComputeCapabilityTypePackOptiX&
ComputeCapabilityTypePackOptiX::operator=(ComputeCapabilityTypePackOptiX&& other) noexcept
{
    assert(this != &other);

    if(pipeline) OPTIX_CHECK(optixPipelineDestroy(pipeline));
    for(auto pg : programGroups)
    {
        if(pg) OPTIX_CHECK(optixProgramGroupDestroy(pg));
    }
    programGroups.clear();
    if(optixModule) OPTIX_CHECK(optixModuleDestroy(optixModule));

    optixModule = std::exchange(other.optixModule, nullptr);
    programGroups = std::exchange(other.programGroups, std::vector<OptixProgramGroup>{});
    pipeline = std::exchange(other.pipeline, nullptr);
    return *this;
}

ComputeCapabilityTypePackOptiX::~ComputeCapabilityTypePackOptiX()
{

    if(pipeline) OPTIX_CHECK(optixPipelineDestroy(pipeline));
    for(auto pg : programGroups)
    {
        if(pg) OPTIX_CHECK(optixProgramGroupDestroy(pg));
    }
    programGroups.clear();
    if(optixModule) OPTIX_CHECK(optixModuleDestroy(optixModule));
}

auto ComputeCapabilityTypePackOptiX::operator<=>(const ComputeCapabilityTypePackOptiX& right) const
{
    return computeCapability <=> right.computeCapability;
}

MRayOptiXContext::MRayOptiXContext()
{
    // We assume every device is otpix capable (this is more or less true)
    // except for Maxwell? (CC_50 is maxwell? I forgot)
    // (TODO: Check and throw if fails)
    CUcontext mainCUDAContext;
    CUDA_DRIVER_CHECK(cuCtxGetCurrent(&mainCUDAContext));

    // Init optix functions
    // TODO: No de-init??
    OPTIX_CHECK(optixInit());
    try
    {
        spdlog::basic_logger_mt(OPTIX_LOGGER_NAME,
                                OPTIX_LOGGER_FILE_NAME, true);
    }
    catch(const spdlog::spdlog_ex& ex)
    {
        throw MRayError("OptiX log init failed: {:s}", ex.what());
    }

    //
    const OptixDeviceContextOptions opts =
    {
        .logCallbackFunction = OptixLog,
        .logCallbackData = nullptr,
        .logCallbackLevel = (MRAY_IS_DEBUG)
                ? 4u : 1u,
        .validationMode = (MRAY_IS_DEBUG)
                ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
                : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF
    };
    OPTIX_CHECK(optixDeviceContextCreate(mainCUDAContext, &opts, &contextOptiX));
}

MRayOptiXContext::~MRayOptiXContext()
{
    OPTIX_CHECK(optixDeviceContextDestroy(contextOptiX));

    auto logger = spdlog::get(OPTIX_LOGGER_NAME);
    logger->flush();
    spdlog::drop(OPTIX_LOGGER_NAME);
}

std::string_view BaseAcceleratorOptiX::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Hardware"sv;
    return BaseAccelTypeName<Name>;
}

BaseAcceleratorOptiX::BaseAcceleratorOptiX(ThreadPool& tp, const GPUSystem& sys,
                                           const AccelGroupGenMap& genMap,
                                           const AccelWorkGenMap& workGenMap)
    : BaseAcceleratorT<BaseAcceleratorOptiX>(tp, sys, genMap, workGenMap)
    , allMem(sys.AllGPUs(), 32_MiB, 256_MiB, true)
{
    std::vector<std::string> ccList;
    ccList.reserve(gpuSystem.AllGPUs().size());
    for(const auto& gpu : gpuSystem.AllGPUs())
        ccList.push_back(gpu->ComputeCapability());
    // Only get unique CC's (no need to create
    // pipeline for each GPU)
    // TODO: This is probably wrong, check some multi-gpu
    // optix examples. We may need one context per gpu maybe
    std::sort(ccList.begin(), ccList.end());
    auto end = std::unique(ccList.begin(), ccList.end());
    ccList.erase(end, ccList.end());

    for(const auto& ccName : ccList)
    {
        optixTypesPerCC.emplace_back(ccName);
        auto& optixTypeThisCC = optixTypesPerCC.back();

        std::string shaderPath = MRAY_FORMAT("{}/{}/CC_{}",
                                             GetProcessPath(),
                                             OPTIX_SHADERS_FOLDER,
                                             optixTypeThisCC.computeCapability);
        auto shader = DevourFile(OPTIX_SHADER_NAME, shaderPath);
        if(shader.has_error()) throw shader.error();

        const char* shaderData = shader.value().data();
        size_t shaderSize = shader.value().size();
        OPTIX_CHECK(optixModuleCreate(contextOptiX,
                                      &OptiXAccelDetail::MODULE_OPTIONS_OPTIX,
                                      &OptiXAccelDetail::PIPELINE_OPTIONS_OPTIX,
                                      shaderData, shaderSize, nullptr, nullptr,
                                      &optixTypeThisCC.optixModule));
    }
}

AABB3 BaseAcceleratorOptiX::InternalConstruct(const std::vector<size_t>& instanceOffsets)
{
    static_assert((sizeof(OptixAabb) == sizeof(AABB3)) &&
                  (alignof(OptixAabb) <= alignof(AABB3)),
                  "Optix and MRay AABBs do not match!");
    size_t totalInstanceCount = instanceOffsets.back();

    // First, create the traversable
    // Temporary allocate AABB, AcceleratorKey,
    // "OptixInstance", Matrix4x4... for each instance.
    // Device stuff
    DeviceMemory tempMem({gpuSystem.AllGPUs()}, 32_MiB, 128_MiB);
    Span<AABB3> dLeafAABBs;
    Span<OptixTraversableHandle> dTraversableHandles;
    Span<Matrix3x4> dInstanceMatrices;
    Span<uint32_t> dSBTCounts;
    Span<uint32_t> dFlags;
    Span<OptixInstance> dInstanceBuildData;
    Span<uint32_t> dSBTOffsets;
    Span<Byte> dScanOrReduceTempMem;
    Span<AABB3> dReducedAABB;
    Span<AccelInstanceMask> dInstanceMasks;

    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    size_t algoTempMemSize = std::max(DeviceAlgorithms::ExclusiveScanTMSize<uint32_t>(totalInstanceCount + 1, queue),
                                      DeviceAlgorithms::ReduceTMSize<AABB3>(totalInstanceCount, queue));
    MemAlloc::AllocateMultiData(Tie(dLeafAABBs, dTraversableHandles,
                                    dInstanceMatrices, dSBTCounts, dFlags,
                                    dInstanceMasks, dInstanceBuildData,
                                    dSBTOffsets, dScanOrReduceTempMem,
                                    dReducedAABB),
                                tempMem,
                                {totalInstanceCount, totalInstanceCount, totalInstanceCount,
                                 totalInstanceCount, totalInstanceCount, totalInstanceCount,
                                 totalInstanceCount, totalInstanceCount + 1, algoTempMemSize,
                                 1});
    // Again CUDA Init check warns about this, but this should be set?
    // via "AcquireIASConstructionParams". maybe we alloc too much?
    // Just memset everything (TODO: wasteful due to barrier wait)
    queue.MemsetAsync(Span(static_cast<Byte*>(tempMem), tempMem.Size()), 0x00);
    queue.Barrier().Wait();

    // Write all required data to buffers
    hBatchStartOffsets.reserve(accelInstances.size());
    //
    uint32_t totalAcceleratorWorkCount = 0;
    size_t accelI = 0;
    GPUQueueIteratorRoundRobin qIt(gpuSystem);
    for(const auto& accGroup : generatedAccels)
    {
        using BaseOptiX = AcceleratorGroupOptixI;
        using Base = AcceleratorGroupI;
        Base* aGroup = accGroup.second.get();
        BaseOptiX* aGroupOptiX = dynamic_cast<BaseOptiX*>(aGroup);
        size_t localCount = instanceOffsets[accelI + 1] - instanceOffsets[accelI];
        size_t offset = instanceOffsets[accelI];
        // Get required parameters for IAS construction
        auto dHandleRegion = dTraversableHandles.subspan(offset, localCount);
        auto dSBTCountRegion = dSBTCounts.subspan(offset, localCount);
        auto dMatrixRegion = dInstanceMatrices.subspan(offset, localCount);
        auto dFlagRegion = dFlags.subspan(offset, localCount);
        auto dMaskRegion = dInstanceMasks.subspan(offset, localCount);
        aGroupOptiX->AcquireIASConstructionParams(dHandleRegion, dMatrixRegion,
                                                  dSBTCountRegion, dFlagRegion,
                                                  dMaskRegion, qIt.Queue());
        aGroupOptiX->OffsetAccelKeyInRecords();
        totalAcceleratorWorkCount += aGroup->InstanceTypeCount();

        //
        hBatchStartOffsets.insert(hBatchStartOffsets.cend(),
                                  size_t(aGroup->InstanceTypeCount()),
                                  uint32_t(instanceOffsets[accelI]));

        accelI++;
        qIt.Next();
    }
    // Wait all queues
    gpuSystem.SyncAll();

    // Calculate global offsets
    DeviceAlgorithms::ExclusiveScan
    (
        dSBTOffsets, dScanOrReduceTempMem,
        ToConstSpan(dSBTCounts), 0u, queue,
        std::plus{}
    );

    // Write these to OptixInstance struct
    queue.IssueWorkKernel<KCCopyToOptixInstance>
    (
        "KCCopyToOptixInstance",
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(totalInstanceCount)},
        // Output
        dInstanceBuildData,
        // Input
        dInstanceMatrices,
        dFlags,
        dTraversableHandles,
        dSBTOffsets,
        dInstanceMasks
    );

    // Actually Construct to a temp mem to find out the compact size
    // Again double dip
    OptixBuildInput buildInput
    {
        .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
        .instanceArray = OptixBuildInputInstanceArray
        {
            .instances = std::bit_cast<CUdeviceptr>(dInstanceBuildData.data()),
            .numInstances = static_cast<uint32_t>(totalInstanceCount),
            .instanceStride = 0
        }
    };

    OptixAccelBufferSizes bufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(contextOptiX, &OptiXAccelDetail::BUILD_OPTIONS_OPTIX,
                                             &buildInput, 1u, &bufferSizes));

    size_t totalTempAccelMemSize = (Math::NextMultiple(bufferSizes.outputSizeInBytes,
                                                       MemAlloc::DefaultSystemAlignment()) +
                                    Math::NextMultiple(bufferSizes.tempSizeInBytes,
                                                       MemAlloc::DefaultSystemAlignment()));

    OptixTraversableHandle phonyHandle;
    Span<uint64_t> dCompactSize;
    Span<Byte> dNonCompactAccelMem;
    Span<Byte> dNonCompactAccelTempMem;
    DeviceMemory accelTempMem({queue.Device()}, totalTempAccelMemSize, totalTempAccelMemSize << 1);
    MemAlloc::AllocateMultiData(Tie(dNonCompactAccelMem,
                                    dNonCompactAccelTempMem,
                                    dCompactSize),
                                accelTempMem,
                                {bufferSizes.outputSizeInBytes,
                                 bufferSizes.tempSizeInBytes, 1});
    // I think there is a bug on CUDA Init check,
    // It does not track optix functions??
    // Memset anyway
    queue.MemsetAsync(Span(static_cast<Byte*>(accelTempMem), accelTempMem.Size()), 0x00);

    Array<OptixAccelEmitDesc, 2> emitProps =
    {
        OptixAccelEmitDesc
        {
            .result = std::bit_cast<CUdeviceptr>(dCompactSize.data()),
            .type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE
        },
        OptixAccelEmitDesc
        {
            .result = std::bit_cast<CUdeviceptr>(dLeafAABBs.data()),
            .type = OPTIX_PROPERTY_TYPE_AABBS
        }
    };
    using mray::cuda::ToHandleCUDA;
    OPTIX_CHECK(optixAccelBuild(contextOptiX, ToHandleCUDA(queue),
                                &OptiXAccelDetail::BUILD_OPTIONS_OPTIX,
                                &buildInput, 1u,
                                std::bit_cast<CUdeviceptr>(dNonCompactAccelTempMem.data()),
                                dNonCompactAccelTempMem.size(),
                                std::bit_cast<CUdeviceptr>(dNonCompactAccelMem.data()),
                                dNonCompactAccelMem.size(), &phonyHandle,
                                emitProps.data(), 2u));

    // Reduce the AABB for host
    DeviceAlgorithms::Reduce(Span<AABB3, 1>(dReducedAABB),
                             dScanOrReduceTempMem,
                             ToConstSpan(dLeafAABBs),
                             AABB3::Negative(),
                             queue, UnionAABB3Functor());

    uint64_t compactedSize = 0;
    uint32_t totalRecordCount = 0;
    queue.MemcpyAsync(Span(&compactedSize, 1), ToConstSpan(dCompactSize));
    queue.MemcpyAsync(Span(&totalRecordCount, 1),
                      ToConstSpan(Span(&dSBTOffsets.back(), 1)));
    queue.MemcpyAsync(Span(&sceneAABB, 1),
                      ToConstSpan(Span(dReducedAABB.data(), 1)));
    queue.Barrier().Wait();

    // Finally do the persistent allocation
    MemAlloc::AllocateMultiData(Tie(dLaunchArgPack, dAccelMemory,
                                    dHitRecords, dEmptyRecords,
                                    dGlobalInstanceInvTransforms,
                                    dGlobalInstanceTraversableHandles,
                                    dGlobalInstanceMasks,
                                    dGlobalInstanceSBTOffsets),
                                allMem,
                                {1, compactedSize, totalRecordCount,
                                 3 + totalAcceleratorWorkCount * MAX_GAS_IN_AN_INSTANCE,
                                 instanceOffsets.back(), instanceOffsets.back(),
                                 instanceOffsets.back(), instanceOffsets.back() + 1});

    OPTIX_CHECK(optixAccelCompact(contextOptiX, ToHandleCUDA(queue), phonyHandle,
                                  std::bit_cast<CUdeviceptr>(dAccelMemory.data()),
                                  dAccelMemory.size(), &baseAccelerator));

    // Copy the traversable handles etc. to the persistent buffer.
    // Invert the transforms
    DeviceAlgorithms::Transform(dGlobalInstanceInvTransforms, ToConstSpan(dInstanceMatrices),
                                queue, KCInvertTransforms());
    // This is required for local ray casting
    queue.MemcpyAsync(dGlobalInstanceTraversableHandles, ToConstSpan(dTraversableHandles));
    queue.MemcpyAsync(dGlobalInstanceSBTOffsets, ToConstSpan(Span(dSBTOffsets)));
    queue.IssueWorkKernel<KCCopyInstanceMaskInfo>
    (
        "KCCopyInstanceMaskInfo",
        DeviceWorkIssueParams{.workCount = static_cast<uint32_t>(totalInstanceCount)},
        // Output
        dGlobalInstanceMasks,
        // Input
        dFlags,
        dInstanceMasks
    );

    // Easy part is done
    // now compile the shaders and attach those on the records
    //
    // -- Local Ray Casting Related --
    // To Support local ray casting for all the instances of the scene
    // we emulate the SBT lookup by hand. Thus we create empty hit records
    // for all accelerator work types (aka. Primitive/Transform pairs).
    // Since MRay at most supports "TracerConstants::MaxPrimBatchPerSurface"
    // ; which is MAX_GAS_IN_AN_INSTANCE, we pre allocate that amount of records
    //
    // For each accelerator work type, a single kernel will be launched its SBT
    // will consists of these empty records. Global hit record array that IAS
    // uses will be indexed via AcceleratorKey + some offset that is fed to OptiX
    // launch.
    //
    // +3 is for Miss / Common RayGeneration / Local RayGeneration shaders.
    uint32_t emptyHRCount = 3 + totalAcceleratorWorkCount * MAX_GAS_IN_AN_INSTANCE;
    std::vector<EmptyHitRecord> hEmptyRecords(emptyHRCount, EmptyHitRecord{});
    //
    ShaderNameMap shaderNames;
    std::vector<GenericHitRecord<>> hAllHitRecords;
    hAllHitRecords.reserve(totalInstanceCount);
    // Write all required data to buffers
    uint32_t accelWorkId = 0;
    for(const auto& accGroup : generatedAccels)
    {
        using Base = AcceleratorGroupOptixI;
        const Base* aGroup = dynamic_cast<const Base*>(accGroup.second.get());
        uint32_t recordStartOffset = static_cast<uint32_t>(hAllHitRecords.size());
        // Get the shader names
        const auto& hitRecords = aGroup->GetHitRecords();
        hAllHitRecords.insert(hAllHitRecords.end(), hitRecords.cbegin(), hitRecords.cend());

        //
        auto localTypeNames = aGroup->GetShaderTypeNames();
        auto recordOffsets = aGroup->GetShaderOffsets();
        assert(localTypeNames.size() == (recordOffsets.size() - 1));
        for(size_t j = 0; j < recordOffsets.size() - 1; j++)
        {

            uint32_t start = recordOffsets[j];
            uint32_t end = recordOffsets[j + 1];
            uint32_t count = end - start;
            const auto& typeName = localTypeNames[j];
            auto& shaderOffsetList = shaderNames[typeName];
            shaderOffsetList.localRayCastHROffset = accelWorkId * MAX_GAS_IN_AN_INSTANCE;
            auto& indexList = shaderOffsetList.globalRayCastHROffsets;
            auto endLoc = indexList.insert(indexList.end(), count, 0);
            std::iota(endLoc, endLoc + count, recordStartOffset + start);
            accelWorkId++;
        }
    }

    // Now we have all the things we need to generate shaders.
    GenerateShaders(hEmptyRecords, hAllHitRecords, shaderNames);

    //
    queue.MemcpyAsync(dHitRecords, ToConstSpan(Span(hAllHitRecords)));
    queue.MemcpyAsync(dEmptyRecords, ToConstSpan(Span(hEmptyRecords)));
    // Finally set the table and GG
    commonCastSBT = {};
    // RG
    commonCastSBT.raygenRecord = std::bit_cast<CUdeviceptr>(dEmptyRecords.data() + RG_COMMON_RECORD);
    // MISS
    commonCastSBT.missRecordBase = std::bit_cast<CUdeviceptr>(dEmptyRecords.data() + MISS_RECORD);
    commonCastSBT.missRecordStrideInBytes = sizeof(EmptyHitRecord);
    commonCastSBT.missRecordCount = 1u;
    // HITS
    commonCastSBT.hitgroupRecordBase = std::bit_cast<CUdeviceptr>(dHitRecords.data());
    commonCastSBT.hitgroupRecordStrideInBytes = sizeof(GenericHitRecord<>);
    commonCastSBT.hitgroupRecordCount = static_cast<uint32_t>(dHitRecords.size());
    //
    localCastSBTList.reserve(totalAcceleratorWorkCount);
    for(uint32_t i = 0; i < totalAcceleratorWorkCount; i++)
    {
        OptixShaderBindingTable localCastSBT = {};
        localCastSBT.missRecordBase = std::bit_cast<CUdeviceptr>(dEmptyRecords.data() + MISS_RECORD);
        localCastSBT.missRecordStrideInBytes = sizeof(EmptyHitRecord);
        localCastSBT.missRecordCount = 1u;
        //
        localCastSBT.hitgroupRecordBase = std::bit_cast<CUdeviceptr>(dEmptyRecords.data() + 3 +
                                                                     i * MAX_GAS_IN_AN_INSTANCE);
        localCastSBT.hitgroupRecordStrideInBytes = sizeof(EmptyHitRecord);
        localCastSBT.hitgroupRecordCount = MAX_GAS_IN_AN_INSTANCE;
        //
        localCastSBT.raygenRecord = std::bit_cast<CUdeviceptr>(dEmptyRecords.data() + RG_LOCAL_RECORD);
        //
        localCastSBTList.push_back(localCastSBT);
    }

    // Wait all queues
    gpuSystem.SyncAll();
    // All Done!
    return sceneAABB;
}

void BaseAcceleratorOptiX::GenerateShaders(std::vector<EmptyHitRecord>& emptyRecords,
                                           std::vector<GenericHitRecord<>>& records,
                                           const ShaderNameMap& shaderNames)
{
    static constexpr auto RaygenCommonName  = "__raygen__OptiX";
    static constexpr auto RaygenLocalName   = "__raygen__LocalOptiX";
    static constexpr auto MissName  = "__miss__OptiX";
    static constexpr auto CH_PREFIX = "__closesthit__";
    static constexpr auto AH_PREFIX = "__anyhit__";
    static constexpr auto IS_PREFIX = "__intersection__";

    static constexpr auto CH_INDEX = 0;
    static constexpr auto AH_INDEX = 1;
    static constexpr auto IS_INDEX = 2;
    // These are runtime, we need persistence
    // to generate multiple data
    std::vector<std::array<std::string, 3>> typeHitPackNames;
    typeHitPackNames.reserve(shaderNames.size());

    // First, generate the names (little bit of string
    // processing is required)
    for(const auto& [shaderPack, _] : shaderNames)
    {
        using TracerConstants::PRIM_PREFIX;
        using TracerConstants::TRANSFORM_PREFIX;
        assert(shaderPack.primName.starts_with(PRIM_PREFIX));
        assert(shaderPack.transformName.starts_with(TRANSFORM_PREFIX));
        //
        std::string_view primStripped = shaderPack.primName.substr(PRIM_PREFIX.size());
        std::string_view transStripped = shaderPack.transformName.substr(TRANSFORM_PREFIX.size());
        typeHitPackNames.push_back
        ({
            std::string(CH_PREFIX) + std::string(primStripped),
            std::string(AH_PREFIX) + std::string(primStripped),
            std::string{}
         });
        if(!shaderPack.isTriangle)
        {
            typeHitPackNames.back()[IS_INDEX] =
                (std::string(IS_PREFIX) + std::string(primStripped)
                 + "_" + std::string(transStripped));
        }
    }

    for(auto& optixTypePack : optixTypesPerCC)
    {
        OptixProgramGroupOptions options = {};
        std::vector<OptixProgramGroupDesc> pgDesc;

        pgDesc.push_back
        (
            OptixProgramGroupDesc
            {
                .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                .raygen = OptixProgramGroupSingleModule
                {
                    .module = optixTypePack.optixModule,
                    .entryFunctionName = RaygenCommonName
                }
            }
        );
        pgDesc.push_back
        (
            OptixProgramGroupDesc
            {
                .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                .raygen = OptixProgramGroupSingleModule
                {
                    .module = optixTypePack.optixModule,
                    .entryFunctionName = RaygenLocalName
                }
            }
        );
        pgDesc.push_back
        (
            OptixProgramGroupDesc
            {
                .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
                .raygen = OptixProgramGroupSingleModule
                {
                    .module = optixTypePack.optixModule,
                    .entryFunctionName = MissName
                }
            }
        );

        uint32_t i = 0;
        for(const auto& shaderPack : shaderNames)
        {
            pgDesc.push_back
            (
                OptixProgramGroupDesc
                {
                    .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                    .hitgroup = OptixProgramGroupHitgroup
                    {
                        .moduleCH = optixTypePack.optixModule,
                        .entryFunctionNameCH = typeHitPackNames[i][CH_INDEX].c_str(),
                        .moduleAH = optixTypePack.optixModule,
                        .entryFunctionNameAH = typeHitPackNames[i][AH_INDEX].c_str(),
                        .moduleIS =  nullptr,
                        .entryFunctionNameIS = nullptr
                    }
                }
            );
            if(!shaderPack.first.isTriangle)
            {
                pgDesc.back().hitgroup.moduleIS = optixTypePack.optixModule;
                pgDesc.back().hitgroup.entryFunctionNameIS = typeHitPackNames[i][IS_INDEX].c_str();
            }
            i++;
        }
        // Finally gen group and pipeline
        optixTypePack.programGroups.resize(pgDesc.size(), nullptr);
        OPTIX_CHECK(optixProgramGroupCreate(contextOptiX, pgDesc.data(),
                                            static_cast<uint32_t>(pgDesc.size()),
                                            &options, nullptr, nullptr,
                                            optixTypePack.programGroups.data()));

        OptixPipelineLinkOptions linkOptions =
        {
            .maxTraceDepth = 1
        };
        OPTIX_CHECK(optixPipelineCreate(contextOptiX, &OptiXAccelDetail::PIPELINE_OPTIONS_OPTIX,
                                        &linkOptions, optixTypePack.programGroups.data(),
                                        static_cast<uint32_t>(optixTypePack.programGroups.size()),
                                        nullptr, nullptr, &optixTypePack.pipeline));

        // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
        // parameters to optixPipelineSetStackSize.
        OptixStackSizes stack_sizes = {};
        for(const auto& pg : optixTypePack.programGroups)
            OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, nullptr));

        uint32_t contStackSize;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes,
                                               linkOptions.maxTraceDepth,
                                               0, 0,
                                               nullptr, nullptr,
                                               &contStackSize));
        OPTIX_CHECK(optixPipelineSetStackSize(optixTypePack.pipeline,
                                              0, 0, contStackSize,
                                              2u));
    }

    // Now set the descriptiors
    currentCCIndex = 0;
    const auto& ccPack = optixTypesPerCC[currentCCIndex];
    uint32_t pgIndex = 0;
    for(const auto& [_, offsetPack] : shaderNames)
    {
        for(uint32_t index : offsetPack.globalRayCastHROffsets)
        {
            auto& record = records[index];
            OPTIX_CHECK(optixSbtRecordPackHeader(ccPack.programGroups[pgIndex + 3],
                                                 record.header));
        }
        // TODO: We probably can memcpy here, but dunno. Just let the OptiX
        // to write w/e it is needed by itself.
        for(uint32_t index = 0; index < MAX_GAS_IN_AN_INSTANCE; index++)
        {
            auto& record = emptyRecords[offsetPack.localRayCastHROffset + index + 3];
            OPTIX_CHECK(optixSbtRecordPackHeader(ccPack.programGroups[pgIndex + 3],
                                                 record.header));
        }
        pgIndex++;
    }
    // RG and Miss records and finish
    OPTIX_CHECK(optixSbtRecordPackHeader(ccPack.programGroups[RG_COMMON_RECORD],
                                         emptyRecords[RG_COMMON_RECORD].header));
    OPTIX_CHECK(optixSbtRecordPackHeader(ccPack.programGroups[RG_LOCAL_RECORD],
                                         emptyRecords[RG_LOCAL_RECORD].header));
    OPTIX_CHECK(optixSbtRecordPackHeader(ccPack.programGroups[MISS_RECORD],
                                         emptyRecords[MISS_RECORD].header));
}

void BaseAcceleratorOptiX::AllocateForTraversal(size_t)
{}

void BaseAcceleratorOptiX::CastRays(// Output
                                    Span<VolumeIndex> dVolumeIndices,
                                    Span<HitKeyPack> dHitIds,
                                    Span<MetaHit> dHitParams,
                                    // I-O
                                    Span<BackupRNGState> dRNGStates,
                                    Span<RayGMem> dRays,
                                    // Input
                                    Span<const RayIndex> dRayIndices,
                                    //
                                    RayCastOptions options,
                                    const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    static const auto annotation = gpuSystem.CreateAnnotation("Ray Casting"sv);
    const auto _ = annotation.AnnotateScope();

    // Scene may be empty, skip ray casting
    if(maxBitsUsedOnKey == Vector2ui::Zero())
        return;

    // TODO: Currently only works for single GPU
    assert(gpuSystem.AllGPUs().size() == 1);
    assert(dRayIndices.size() != 0);

    // This code is not generic, so we go in and take the stuff
    // from device interface specific stuff
    using mray::cuda::ToHandleCUDA;
    const ComputeCapabilityTypePackOptiX& deviceTypes = optixTypesPerCC[currentCCIndex];

    // Copy args
    ArgumentPackOptiX argPack =
    {
        .rayCastOptions = options,
        .mode           = RenderModeOptiX::NORMAL,
        .nParams        = NormalRayCastArgPackOptiX
        {
            .baseAccelerator = baseAccelerator,
            .dVolumeIndices  = dVolumeIndices,
            .dHitKeys        = dHitIds,
            .dHits           = dHitParams,
            .dRNGStates      = dRNGStates,
            .dRays           = dRays,
            .dRayIndices     = dRayIndices
        }
    };
    queue.MemcpyAsync(dLaunchArgPack, Span<const ArgumentPackOptiX>(&argPack, 1));
    CUdeviceptr argsPtr = std::bit_cast<CUdeviceptr>(dLaunchArgPack.data());

    // Launch!
    OPTIX_CHECK(optixLaunch(deviceTypes.pipeline, ToHandleCUDA(queue), argsPtr,
                            dLaunchArgPack.size_bytes(), &commonCastSBT,
                            static_cast<uint32_t>(dRayIndices.size()), 1u, 1u));
    OPTIX_LAUNCH_CHECK();
}

void BaseAcceleratorOptiX::CastVisibilityRays(Bitspan<uint32_t> dIsVisibleBuffer,
                                              // I-O
                                              Span<BackupRNGState> dRNGStates,
                                              // Input
                                              Span<const RayGMem> dRays,
                                              Span<const RayIndex> dRayIndices,
                                              RayCastOptions options,
                                              const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    static const auto annotation = gpuSystem.CreateAnnotation("Visibilty Casting"sv);
    const auto _ = annotation.AnnotateScope();

    // Scene may be empty, skip ray casting
    if(maxBitsUsedOnKey == Vector2ui::Zero())
        return;

    // TODO: Currently only works for single GPU
    assert(gpuSystem.AllGPUs().size() == 1);
    assert(dRayIndices.size() != 0);
    // This code is not generic, so we go in and take the stuff
    // from device interface specific stuff
    using mray::cuda::ToHandleCUDA;
    const ComputeCapabilityTypePackOptiX& deviceTypes = optixTypesPerCC[currentCCIndex];

    // Copy args
    ArgumentPackOptiX argPack =
    {
        .rayCastOptions = options,
        .mode           = RenderModeOptiX::VISIBILITY,
        .vParams        = VisibilityCastArgPackOptiX
        {
            .baseAccelerator    = baseAccelerator,
            .dIsVisibleBuffer   = dIsVisibleBuffer,
            .dRNGStates         = dRNGStates,
            .dRays              = dRays,
            .dRayIndices        = dRayIndices
        }
    };
    queue.MemcpyAsync(dLaunchArgPack, Span<const ArgumentPackOptiX>(&argPack, 1));
    CUdeviceptr argsPtr = std::bit_cast<CUdeviceptr>(dLaunchArgPack.data());

    // Launch!
    OPTIX_CHECK(optixLaunch(deviceTypes.pipeline, ToHandleCUDA(queue), argsPtr,
                            dLaunchArgPack.size_bytes(), &commonCastSBT,
                            static_cast<uint32_t>(dRayIndices.size()), 1u, 1u));
    OPTIX_LAUNCH_CHECK();
}

void BaseAcceleratorOptiX::CastLocalRays(// Output
                                         Span<VolumeIndex> dVolumeIndices,
                                         Span<HitKeyPack> dHitIds,
                                         Span<MetaHit> dHitParams,
                                         // I-O
                                         Span<BackupRNGState> dRNGStates,
                                         Span<RayGMem> dRays,
                                         // Input
                                         Span<const RayIndex> dRayIndices,
                                         Span<const AcceleratorKey> dAccelKeys,
                                         //
                                         CommonKey hAccelKeyBatchPortion,
                                         RayCastOptions options,
                                         const GPUQueue& queue)
{
    using namespace std::string_view_literals;
    static const auto annotation = gpuSystem.CreateAnnotation("Local Ray Casting"sv);
    const auto _ = annotation.AnnotateScope();

    // Scene may be empty, skip ray casting
    if(maxBitsUsedOnKey == Vector2ui::Zero())
        return;

    // TODO: Currently only works for single GPU
    assert(gpuSystem.AllGPUs().size() == 1);
    assert(dRayIndices.size() != 0);
    // This code is not generic, so we go in and take the stuff
    // from device interface specific stuff
    using mray::cuda::ToHandleCUDA;
    const ComputeCapabilityTypePackOptiX& deviceTypes = optixTypesPerCC[currentCCIndex];

    // Find the offset
    uint32_t batchStartOffset = uint32_t(hBatchStartOffsets[hAccelKeyBatchPortion]);
    const auto* localCastSBT = &localCastSBTList[hAccelKeyBatchPortion];

    // Copy args
    ArgumentPackOptiX argPack =
    {
        .rayCastOptions = options,
        .mode           = RenderModeOptiX::LOCAL,
        .lParams        = LocalRayCastArgPackOptiX
        {
            .dVolumeIndices      = dVolumeIndices,
            .dHitKeys            = dHitIds,
            .dHits               = dHitParams,
            .dRNGStates          = dRNGStates,
            .dRays               = dRays,
            .dRayIndices         = dRayIndices,
            .dAcceleratorKeys    = dAccelKeys,
            .dGlobalInstanceTraversables  = dGlobalInstanceTraversableHandles,
            .dGlobalInstanceInvTransforms = dGlobalInstanceInvTransforms,
            .dGlobalInstanceSBTOffsets    = dGlobalInstanceSBTOffsets,
            .dGlobalInstanceMasks         = dGlobalInstanceMasks,
            .dGlobalHitRecordList         = dHitRecords,
            .batchStartOffset    = batchStartOffset
        }
    };
    queue.MemcpyAsync(dLaunchArgPack, Span<const ArgumentPackOptiX>(&argPack, 1));
    CUdeviceptr argsPtr = std::bit_cast<CUdeviceptr>(dLaunchArgPack.data());

    // Launch!
    OPTIX_CHECK(optixLaunch(deviceTypes.pipeline, ToHandleCUDA(queue), argsPtr,
                            dLaunchArgPack.size_bytes(), localCastSBT,
                            static_cast<uint32_t>(dRayIndices.size()), 1u, 1u));
    OPTIX_LAUNCH_CHECK();
}

void BaseAcceleratorOptiX::WriteAllAcceleratorKeys(Span<AcceleratorKey> dAccelKeys,
                                                   const GPUQueue& queue) const
{
    assert(dAccelKeys.size() == dGlobalInstanceSBTOffsets.size() - 1);

    queue.IssueWorkKernel<KCCopyAllAccelKeysOptiX>
    (
        "KCCopyAllAccelKeysOptiX",
        DeviceWorkIssueParams{.workCount = uint32_t(TotalInstanceCount())},
        // Output
        dAccelKeys,
        // Input
        dHitRecords,
        dGlobalInstanceSBTOffsets
    );
}

size_t BaseAcceleratorOptiX::GPUMemoryUsage() const
{
    size_t totalSize = allMem.Size();
    for(const auto& [_, accelGroup] : this->generatedAccels)
    {
        totalSize += accelGroup->GPUMemoryUsage();
    }
    return totalSize;
}

OptixDeviceContext BaseAcceleratorOptiX::GetOptixDeviceHandle() const
{
    return contextOptiX;
}