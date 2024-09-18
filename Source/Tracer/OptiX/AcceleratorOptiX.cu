#include "AcceleratorOptiX.h"

#include <cassert>
#include <optix_host.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <fstream>
// Magic linking, these are populated via CUDA runtime?
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include "Core/System.h"
#include "Core/Expected.h"
#include "Core/Filesystem.h"
#include "Core/Error.hpp"

static constexpr auto OPTIX_LOGGER_NAME = "OptiXLogger";
static constexpr auto OPTIX_LOGGER_FILE_NAME = "optix_log";
static constexpr auto OPTIX_SHADERS_FOLDER = "OptiXShaders";
static constexpr auto OPTIX_SHADER_NAME = "OptiXPTX.optixir";

Expected<std::vector<Byte>>
DevourFile(const std::string& shaderName,
           const std::string& executablePath)
{
    std::string fullPath = Filesystem::RelativePathToAbsolute(shaderName,
                                                              executablePath);
    std::streamoff size = std::ifstream(fullPath,
                                        std::ifstream::ate |
                                        std::ifstream::binary).tellg();
    assert(size == Math::NextMultiple(size, std::streamoff(4)));
    std::vector<Byte> source(static_cast<size_t>(size), Byte(0));
    std::ifstream shaderFile = std::ifstream(fullPath, std::ios::binary);

    if(!shaderFile.is_open())
        return MRayError("Unable to open shader file \"{}\"",
                         fullPath);
    shaderFile.read(reinterpret_cast<char*>(source.data()),
                    static_cast<std::streamsize>(source.size()));
    return source;
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

void OptixLog(unsigned int level, const char* tag, const char* message, void* cbdata)
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
    , pipeline(std::exchange(other.pipeline, nullptr))
{}

ComputeCapabilityTypePackOptiX&
ComputeCapabilityTypePackOptiX::operator=(ComputeCapabilityTypePackOptiX&& other) noexcept
{
    assert(this != &other);

    OPTIX_CHECK(optixModuleDestroy(optixModule));
    OPTIX_CHECK(optixPipelineDestroy(pipeline));

    optixModule = std::exchange(other.optixModule, nullptr);
    pipeline = std::exchange(other.pipeline, nullptr);
    return *this;
}

ComputeCapabilityTypePackOptiX::~ComputeCapabilityTypePackOptiX()
{
    OPTIX_CHECK(optixModuleDestroy(optixModule));
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
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
    OPTIX_CHECK(optixInit());
    try
    {
        auto logger = spdlog::basic_logger_mt(OPTIX_LOGGER_NAME,
                                              OPTIX_LOGGER_FILE_NAME);
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
    OPTIX_CHECK(optixDeviceContextCreate(mainCUDAContext, &opts, &optixContext));
}

ContextOptiX::~ContextOptiX()
{
    OPTIX_CHECK(optixDeviceContextDestroy(optixContext));
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
    , accelMem(sys.AllGPUs(), 32_MiB, 256_MiB, true)
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

        std::string shaderPath = MRAY_FORMAT("{}/{}/{}",
                                             GetProcessPath(),
                                             OPTIX_SHADERS_FOLDER,
                                             optixTypeThisCC.computeCapability);
        auto shader = DevourFile(OPTIX_SHADER_NAME, OPTIX_SHADER_NAME);
        if(shader.has_error()) throw shader.error();

        const char* shaderData = reinterpret_cast<char*>(shader.value().data());
        size_t shaderSize = shader.value().size();
        OPTIX_CHECK(optixModuleCreate(optixContext,
                                      &OptiXAccelDetail::MODULE_OPTIONS_OPTIX,
                                      &OptiXAccelDetail::PIPELINE_OPTIONS_OPTIX,
                                      shaderData, shaderSize, nullptr, nullptr,
                                      &optixTypeThisCC.optixModule));
    }
}

AABB3 BaseAcceleratorOptiX::InternalConstruct(const std::vector<size_t>& instanceOffsets)
{
    return AABB3::Negative();
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
    const GPUDevice* device = queue.Device();
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

void BaseAcceleratorOptiX::Construct(BaseAccelConstructParams p)
{


    //OptixModule moduleOut;
    ////


    //constexpr OptixPipelineLinkOptions linkOpts =
    //{
    //    .maxTraceDepth = 1,
    //};

    //std::array<OptixProgramGroupDesc, 3> g =
    //{
    //    OptixProgramGroupDesc
    //    {
    //        .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
    //        .flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE,
    //        .raygen = OptixProgramGroupSingleModule
    //        {
    //            .module = nullptr,
    //            .entryFunctionName = "__raygen__OptiX",
    //        },
    //    },
    //    OptixProgramGroupDesc
    //    {
    //        .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
    //        .flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE,
    //        .miss = OptixProgramGroupSingleModule
    //        {
    //            .module = nullptr,
    //            .entryFunctionName = "__miss__OptiX",
    //        },
    //    },
    //    OptixProgramGroupDesc
    //    {
    //        .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
    //        .flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE,
    //        .hitgroup = OptixProgramGroupHitgroup
    //        {
    //            .moduleCH = nullptr,
    //            .entryFunctionNameCH = "__closesthit__",
    //            .moduleAH = nullptr,
    //            .entryFunctionNameAH = "__anyhit__",
    //            .moduleIS = nullptr,
    //            .entryFunctionNameIS = "__intersection__"
    //        }
    //    }
    //};

    //std::array<OptixProgramGroup, 3> programGroups;

    //const OptixProgramGroupOptions programGroupOpts =
    //{
    //    nullptr,
    //};

    //OPTIX_CHECK(optixProgramGroupCreate(optixContext, g.data(), g.size(),
    //                                    &programGroupOpts,
    //                                    nullptr, nullptr, programGroups.data()));



    //OptixPipeline pipeline;
    //OPTIX_CHECK(optixPipelineCreate(optixContext, &PIPELINE_OPTIONS_OPTIX,
    //                                &linkOpts, programGroups.data(),
    //                                programGroups.size(), nullptr, nullptr,
    //                                &pipeline));



    //optixTypes.reserve(gpuSystem.AllGPUs().size());
    //for(const GPUDevice* device : gpuSystem.AllGPUs())
    //{
    //    DeviceTypesOptiX& typePack = optixTypes.emplace_back();



    //}



    //
    //BaseAcceleratorT<BaseAcceleratorOptiX>::Construct(p);
}

size_t BaseAcceleratorOptiX::GPUMemoryUsage() const
{
    size_t totalSize = accelMem.Size();
    for(const auto& [_, accelGroup] : this->generatedAccels)
    {
        totalSize += accelGroup->GPUMemoryUsage();
    }
    return totalSize;
}