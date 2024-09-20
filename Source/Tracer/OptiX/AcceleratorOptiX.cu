#include "AcceleratorOptiX.h"

#include <cassert>
#include <map>
#include <optix_host.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <fstream>

#include "Core/System.h"
#include "Core/Expected.h"
#include "Core/Filesystem.h"
#include "Core/Error.hpp"

#include "Device/GPUAlgScan.h"
#include "Device/GPUSystem.hpp"

// Magic linking, these are populated via CUDA runtime?
#include <optix_function_table_definition.h>

MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCCopyToOptixInstance(// Output
                           MRAY_GRID_CONSTANT const Span<OptixInstance> dInstances,
                           // Input
                           MRAY_GRID_CONSTANT const Span<const Matrix4x4> dMatrices,
                           MRAY_GRID_CONSTANT const Span<const uint32_t> dInstanceFlags,
                           MRAY_GRID_CONSTANT const Span<const OptixTraversableHandle> dHandles,
                           MRAY_GRID_CONSTANT const Span<const uint32_t> dGlobalSBTOffsets)
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
            .visibilityMask = 0xFF,
            .flags = dInstanceFlags[i],
            .traversableHandle = dHandles[i]
        };
        dInstances[i] = out;
    }
}

static constexpr auto OPTIX_LOGGER_NAME = "OptiXLogger";
static constexpr auto OPTIX_LOGGER_FILE_NAME = "optix_log";
static constexpr auto OPTIX_SHADERS_FOLDER = "OptiXShaders";
static constexpr auto OPTIX_SHADER_NAME = "OptiXPTX.optixir";

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
    std::streamoff size = std::ifstream(fullPath,
                                        std::ifstream::ate |
                                        std::ifstream::binary).tellg();
    std::vector<Byte> source(static_cast<size_t>(size), Byte(0));
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
    , programGroup(std::exchange(other.programGroup, nullptr))
    , pipeline(std::exchange(other.pipeline, nullptr))
{}

ComputeCapabilityTypePackOptiX&
ComputeCapabilityTypePackOptiX::operator=(ComputeCapabilityTypePackOptiX&& other) noexcept
{
    assert(this != &other);

    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(programGroup));
    OPTIX_CHECK(optixModuleDestroy(optixModule));

    optixModule = std::exchange(other.optixModule, nullptr);
    programGroup = std::exchange(other.programGroup, nullptr);
    pipeline = std::exchange(other.pipeline, nullptr);
    return *this;
}

ComputeCapabilityTypePackOptiX::~ComputeCapabilityTypePackOptiX()
{
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(programGroup));
    OPTIX_CHECK(optixModuleDestroy(optixModule));
}

auto ComputeCapabilityTypePackOptiX::operator<=>(const ComputeCapabilityTypePackOptiX& right) const
{
    return computeCapability <=> right.computeCapability;
}

ContextOptiX::ContextOptiX()
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
        auto logger = spdlog::basic_logger_mt(OPTIX_LOGGER_NAME,
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

ContextOptiX::~ContextOptiX()
{
    OPTIX_CHECK(optixDeviceContextDestroy(contextOptiX));
}

std::string_view BaseAcceleratorOptiX::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Hardware"sv;
    return BaseAccelTypeName<Name>;
}

BaseAcceleratorOptiX::BaseAcceleratorOptiX(BS::thread_pool& tp, const GPUSystem& sys,
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

struct PrimShaderNames
{
    std::string_view PGName;
    std::string_view TGName;
};

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
    Span<Matrix4x4> dInstanceMatrices;
    Span<uint32_t> dSBTCounts;
    Span<uint32_t> dFlags;
    Span<OptixInstance> dInstanceBuildData;
    Span<uint32_t> dSBTOffsets;
    Span<Byte> dScanOrReduceTempMem;
    Span<AABB3> dReducedAABB;

    size_t algoTempMemSize = std::max(DeviceAlgorithms::ExclusiveScanTMSize<uint32_t>(totalInstanceCount + 1),
                                      DeviceAlgorithms::ReduceTMSize<AABB3>(totalInstanceCount));
    MemAlloc::AllocateMultiData(std::tie(dLeafAABBs,  dTraversableHandles,
                                         dInstanceMatrices, dSBTCounts, dFlags,
                                         dInstanceBuildData, dSBTOffsets,
                                         dScanOrReduceTempMem, dReducedAABB),
                                tempMem,
                                {totalInstanceCount, totalInstanceCount,
                                 totalInstanceCount, totalInstanceCount, totalInstanceCount,
                                 totalInstanceCount, totalInstanceCount + 1, algoTempMemSize,
                                 1});

    // Write all required data to buffers
    size_t i = 0;
    GPUQueueIteratorRoundRobin qIt(gpuSystem);
    for(const auto& accGroup : generatedAccels)
    {
        using Base = AcceleratorGroupOptixI;
        Base* aGroup = static_cast<Base*>(accGroup.second.get());
        size_t localCount = instanceOffsets[i + 1] - instanceOffsets[i];
        size_t offset = instanceOffsets[i];
        // Get required parameters for IAS construction
        auto dHandleRegion = dTraversableHandles.subspan(offset, localCount);
        auto dSBTCountRegion = dSBTCounts.subspan(offset, localCount);
        auto dMatrixRegion = dInstanceMatrices.subspan(offset, localCount);
        auto dFlagRegion = dFlags.subspan(offset, localCount);
        aGroup->AcquireIASConstructionParams(dHandleRegion, dMatrixRegion,
                                             dSBTCountRegion, dFlagRegion,
                                             qIt.Queue());
        aGroup->OffsetAccelKeyInRecords();
        i++;
        qIt.Next();
    }
    // Wait all queues
    gpuSystem.SyncAll();

    // Calculate global offsets
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    DeviceAlgorithms::ExclusiveScan
    (
        dSBTOffsets, dScanOrReduceTempMem,
        ToConstSpan(dSBTCounts), 0u, queue,
        std::plus{}
    );

    // Write these to OptixInstance struct
    queue.IssueSaturatingKernel<KCCopyToOptixInstance>
    (
        "KCCopyToOptixInstance",
        KernelIssueParams{.workCount = static_cast<uint32_t>(totalInstanceCount)},
        // Output
        dInstanceBuildData,
        // Input
        dInstanceMatrices,
        dFlags,
        dTraversableHandles,
        dSBTOffsets
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
    MemAlloc::AllocateMultiData(std::tie(dNonCompactAccelMem,
                                         dNonCompactAccelTempMem,
                                         dCompactSize),
                                accelTempMem,
                                {bufferSizes.outputSizeInBytes,
                                 bufferSizes.tempSizeInBytes, 1});

    std::array<OptixAccelEmitDesc, 2> emitProps =
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
    MemAlloc::AllocateMultiData(std::tie(dLaunchArgPack, dAccelMemory, dHitRecords),
                                allMem, {1, compactedSize, totalRecordCount});

    OPTIX_CHECK(optixAccelCompact(contextOptiX, ToHandleCUDA(queue), phonyHandle,
                                  std::bit_cast<CUdeviceptr>(dAccelMemory.data()),
                                  dAccelMemory.size(), &baseAccelerator));

    // Easy part is done
    // now compile the shaders and attach those on the records
    ShaderNameMap shaderNames;
    std::vector<GenericHitRecord<>> hAllHitRecords;
    hAllHitRecords.reserve(totalInstanceCount);
    // Write all required data to buffers
    for(const auto& accGroup : generatedAccels)
    {
        using Base = AcceleratorGroupOptixI;
        const Base* aGroup = static_cast<const Base*>(accGroup.second.get());
        uint32_t recordStartOffset = static_cast<uint32_t>(hAllHitRecords.size());
        // Get the shader names
        auto hitRecords = aGroup->GetHitRecords();
        hAllHitRecords.insert(hAllHitRecords.end(), hitRecords.cbegin(), hitRecords.cend());
        //
        auto localTypeNames = aGroup->GetShaderTypeNames();
        auto recordOffsets = aGroup->GetShaderOffsets();
        assert(localTypeNames.size() == recordOffsets.size());
        for(size_t i = 0; i < recordOffsets.size() - 1; i++)
        {
            uint32_t start = recordStartOffset + recordOffsets[i];
            uint32_t end = recordStartOffset + recordOffsets[i + 1];

            const auto& typeName = localTypeNames[i];
            auto& indexList = shaderNames[typeName];
            auto endLoc = indexList.insert(indexList.end(), end - start, 0);
            std::iota(endLoc - (end - start), endLoc, recordStartOffset);
        }
    }

    // Now we have all the things we need to generate shaders.
    GenerateShaders(hAllHitRecords, shaderNames);

    // Wait all queues
    gpuSystem.SyncAll();
    return sceneAABB;
}

void BaseAcceleratorOptiX::GenerateShaders(std::vector<GenericHitRecord<>>& records,
                                           const ShaderNameMap& shaderNames)
{

}

void BaseAcceleratorOptiX::AllocateForTraversal(size_t)
{}

void BaseAcceleratorOptiX::CastRays(// Output
                                   Span<HitKeyPack> dHitIds,
                                   Span<MetaHit> dHitParams,
                                   // I-O
                                   Span<BackupRNGState> dRNGStates,
                                   Span<RayGMem> dRays,
                                   // Input
                                   Span<const RayIndex> dRayIndices,
                                   const GPUQueue& queue)
{
    // This code is not generic, so we go in and take the stuff
    // from device interface specific stuff
    using mray::cuda::ToHandleCUDA;
    const ComputeCapabilityTypePackOptiX& deviceTypes = optixTypesPerCC[0];

    // Copy args
    ArgumentPackOpitX argPack =
    {
        .baseAccelerator    = baseAccelerator,
        .dHitKeys           = dHitIds,
        .dHits              = dHitParams,
        .dRNGStates         = dRNGStates,
        .dRays              = dRays,
        .dRayIndices        = dRayIndices
    };
    queue.MemcpyAsync(dLaunchArgPack, Span<const ArgumentPackOpitX>(&argPack, 1));
    CUdeviceptr argsPtr = std::bit_cast<CUdeviceptr>(dLaunchArgPack.data());

    // Launch!
    OPTIX_CHECK(optixLaunch(deviceTypes.pipeline, ToHandleCUDA(queue), argsPtr,
                            dLaunchArgPack.size_bytes(), &commonSBT,
                            static_cast<uint32_t>(dRayIndices.size()), 1u, 1u));
    OPTIX_LAUNCH_CHECK();
}

void BaseAcceleratorOptiX::CastShadowRays(// Output
                                          Bitspan<uint32_t> dIsVisibleBuffer,
                                          Bitspan<uint32_t> dFoundMediumInterface,
                                          // I-O
                                          Span<BackupRNGState> dRNGStates,
                                          // Input
                                          Span<const RayIndex> dRayIndices,
                                          Span<const RayGMem> dShadowRays,
                                          const GPUQueue& queue)
{}

void BaseAcceleratorOptiX::CastLocalRays(// Output
                                         Span<HitKeyPack> dHitIds,
                                         Span<MetaHit> dHitParams,
                                         // I-O
                                         Span<BackupRNGState> dRNGStates,
                                         // Input
                                         Span<const RayGMem> dRays,
                                         Span<const RayIndex> dRayIndices,
                                         Span<const AcceleratorKey> dAccelIdPacks,
                                         const GPUQueue& queue)
{

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