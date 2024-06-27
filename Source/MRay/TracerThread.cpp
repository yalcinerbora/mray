#include "TracerThread.h"

#include <fstream>
#include <filesystem>

#include <BS/BS_thread_pool.hpp>
#include <nlohmann/json.hpp>

#include "Core/Timer.h"
#include "Core/TypeNameGenerators.h"
#include "Core/Error.h"
#include "Core/Error.hpp"

#include "Common/JsonCommon.h"

static constexpr RendererId INVALID_RENDERER_ID = RendererId(std::numeric_limits<uint32_t>::max());

enum class FilterType
{
    BOX,
    TENT,
    GAUSSIAN,
    MITCHELL_NETRAVALI
};

struct BoxParams        { Float radius; };
struct TentParams       { Float radius; };
struct GaussParams      { Float radius; Float alpha; };
struct MitchellParams   { Float radius; Float b; Float c; };

class FilterParameters : public Variant<BoxParams, TentParams,
                                        GaussParams, MitchellParams>
{
    using Base = Variant<BoxParams, TentParams,
                         GaussParams, MitchellParams>;

    public:
    enum E
    {
        BOX,
        TENT,
        GAUSSIAN,
        MITCHELL_NETRAVALI,
        END
    };

    static constexpr auto TYPE_NAME = "type";
    static constexpr auto RADIUS_NAME = "radius";
    static constexpr auto GAUSS_ALPHA_NAME = "alpha";
    static constexpr auto MN_B_NAME = "b";
    static constexpr auto MN_C_NAME = "b";

    private:
    static constexpr std::array<std::string_view, static_cast<size_t>(END)> Names =
    {
        "Box",
        "Tent",
        "Gaussian",
        "Mitchell-Netravali"
    };


    public:
    using Base::Base;

    static constexpr std::string_view   ToString(E e);
    static constexpr E                  FromString(std::string_view e);
};

constexpr std::string_view FilterParameters::ToString(E e)
{
    return Names[static_cast<uint32_t>(e)];
}

constexpr typename FilterParameters::E
FilterParameters::FromString(std::string_view sv)
{
    using IntType = std::underlying_type_t<E>;
    IntType i = 0;
    for(const std::string_view& checkSV : Names)
    {
        if(checkSV == sv) return E(i);
        i++;
    }
    return END;
}

struct TracerConfig
{
    std::string dllName;
    std::string dllCreateFuncName = "ConstructTracer";
    std::string dllDeleteFuncName = "DestroyTracer";

    TracerParameters params;
};

void from_json(const nlohmann::json& node, AcceleratorType& t)
{
    using namespace std::string_view_literals;
    static constexpr std::array<std::string_view, 3> TYPES =
    {
        "Linear"sv,
        "BVH"sv,
        "Hardware"
    };
    auto loc = std::find(TYPES.cbegin(), TYPES.cend(),
                         node.get<std::string_view>());
    size_t result = std::distance(TYPES.cbegin(), loc);

    if(result == TYPES.size())
        throw MRayError("Unknown sample type name {}",
                        node.get<std::string_view>());
    t = static_cast<AcceleratorType>(result);
}

void from_json(const nlohmann::json& node, SamplerType& t)
{
    using namespace std::string_view_literals;
    static constexpr std::array<std::string_view, 1> TYPES =
    {
        "Independent"sv
    };
    auto loc = std::find(TYPES.cbegin(), TYPES.cend(),
                         node.get<std::string_view>());
    size_t result = std::distance(TYPES.cbegin(), loc);

    if(result == TYPES.size())
        throw MRayError("Unknown sample type name {}",
                        node.get<std::string_view>());
    t = static_cast<SamplerType>(result);
}

Expected<TracerConfig> LoadTracerConfig(const std::string& configJsonPath)
{
    using namespace std::literals;

    // Object keys
    static constexpr auto DLL_NAME = "TracerDLL"sv;
    static constexpr auto PARAMETERS_NAME = "Parameters"sv;
    // DLL entries
    static constexpr auto DLL_FILE_NAME = "name"sv;
    static constexpr auto DLL_CONSTRUCT_NAME = "construct"sv;
    static constexpr auto DLL_DESTRUCT_NAME = "destruct"sv;
    // Params
    static constexpr auto SEED_NAME = "seed"sv;
    static constexpr auto ACCEL_TYPE_NAME = "acceleratorType"sv;
    static constexpr auto PAR_HINT_NAME = "parallelHint"sv;
    static constexpr auto SAMPLER_TYPE_NAME = "samplerType"sv;
    static constexpr auto CLAMP_TEX_RES_NAME = "clampTexRes"sv;
    static constexpr auto PARTITION_LOGIC_NAME = "partitionLogic"sv;

    nlohmann::json configJson;
    auto OptionalFetch = [](auto& outEntry, std::string_view NAME,
                            const nlohmann::json& inJson) -> void
    {
        const auto it = inJson.find(NAME);
        if(it != inJson.cend())
            outEntry = *it;
    };

    try
    {
        std::ifstream file(configJsonPath);
        if(!file.is_open())
            return MRayError("Tracer config file \"{}\" is not found",
                             configJsonPath);
        configJson = nlohmann::json::parse(file, nullptr, true, true);

        // DLL portion
        TracerConfig config;
        const auto& dllJson = configJson[DLL_NAME];
        config.dllName = dllJson[DLL_FILE_NAME];
        OptionalFetch(config.dllCreateFuncName, DLL_CONSTRUCT_NAME, dllJson);
        OptionalFetch(config.dllDeleteFuncName, DLL_DESTRUCT_NAME, dllJson);

        // TODO: Add option config reading
        nlohmann::json paramsJson;
        OptionalFetch(paramsJson, PARAMETERS_NAME, configJson);
        if(paramsJson.empty()) return config;

        OptionalFetch(config.params.seed, SEED_NAME, paramsJson);
        OptionalFetch(config.params.accelMode, ACCEL_TYPE_NAME, paramsJson);
        OptionalFetch(config.params.parallelizationHint, PAR_HINT_NAME, paramsJson);
        OptionalFetch(config.params.samplerType, SAMPLER_TYPE_NAME, paramsJson);
        OptionalFetch(config.params.clampedTexRes, CLAMP_TEX_RES_NAME, paramsJson);
        return config;
    }
    catch(const MRayError& e)
    {
        return e;
    }
    catch(const nlohmann::json::exception& e)
    {
        return MRayError("Json Error ({})", std::string(e.what()));
    }
}

MRayError TracerThread::CreateRendererFromConfig(const std::string& configJsonPath)
{
    using namespace TypeNameGen::Runtime;
    using namespace std::literals;
    static constexpr auto INITIAL_NAME = "initialName"sv;
    static constexpr auto RENDERER_LIST_NAME = "Renderers"sv;

    TracerConfig config;
    nlohmann::json configJson;
    try
    {
        std::ifstream file(configJsonPath);
        if(!file.is_open())
            return MRayError("Renderer config file \"{}\" is not found",
                             configJsonPath);
        configJson = nlohmann::json::parse(file, nullptr, true, true);


        const nlohmann::json& renderers = configJson.at(RENDERER_LIST_NAME);
        const nlohmann::json& rendererName = configJson.at(INITIAL_NAME);
        std::string rName = AddRendererPrefix(rendererName.get<std::string_view>());
        currentRendererName = rName;

        const nlohmann::json& rendererNode = renderers.at(rendererName);

        assert(currentRenderer == INVALID_RENDERER_ID);
        currentRenderer = tracer->CreateRenderer(rName);
        RendererAttributeInfoList attributes;

        uint32_t attribIndex = 0;
        for(const auto& attrib : attributes)
        {
            using enum GenericAttributeInfo::E;
            using enum AttributeIsArray;
            if(std::get<IS_ARRAY_INDEX>(attrib) == IS_ARRAY)
                return MRayError("Config read \"{}\": Array renderer attributes "
                                 "are not supported yet",
                                 configJsonPath);

            AttributeOptionality optionality = std::get<OPTIONALITY_INDEX>(attrib);
            std::string_view name = std::get<LOGIC_INDEX>(attrib);
            MRayDataTypeRT type = std::get<LAYOUT_INDEX>(attrib);

            MRayError e = std::visit([&, this](auto&& t) -> MRayError
            {
                using enum AttributeOptionality;
                using T = std::remove_cvref_t<decltype(t)>::Type;
                auto loc = rendererNode.find(name);

                if(optionality != MR_OPTIONAL && loc == rendererNode.end())
                    return MRayError("Config read \"{}\": Mandatory variable \"{}\" "
                                     "for \"{}\" is not found in config file",
                                     configJsonPath, name, rName);
                if(loc == rendererNode.end()) return MRayError::OK;

                T in = loc->get<T>();
                TransientData data(std::in_place_type_t<T>{}, 1);
                data.Push(Span<const T>(&in, 1));
                tracer->PushRendererAttribute(currentRenderer, attribIndex,
                                              std::move(data));
                return MRayError::OK;
            }, type);

            if(e) return e;
            attribIndex++;
        }
    }
    catch(const MRayError& e)
    {
        return e;
    }
    catch(const nlohmann::json::exception& e)
    {
        return MRayError("Json Error ({})", std::string(e.what()));
    }
    return MRayError::OK;
}

void TracerThread::RestartRenderer()
{
    // Nothing to restart
    if(currentRenderer == INVALID_RENDERER_ID)
        return;

    tracer->StopRender();

    RenderImageParams rp =
    {
        .resolution = resolution,
        .regionMin = regionMin,
        .regionMax = regionMax,
        .semaphore = currentSem,
        .initialSemCounter = 0
    };
    RenderBufferInfo rbi = tracer->StartRender(currentRenderer,
                                               sceneIds.camSurfaces[currentCamIndex],
                                               rp,
                                               currentRenderLogic0,
                                               currentRenderLogic1,
                                               currentCamTransform);

    transferQueue.Enqueue(TracerResponse
    (
        std::in_place_index<TracerResponse::RENDER_BUFFER_INFO>,
        rbi
    ));
    // Clear the image section as well
    transferQueue.Enqueue(TracerResponse
    (
        std::in_place_index<TracerResponse::CLEAR_IMAGE_SECTION>,
        true
    ));
}

void TracerThread::HandleRendering()
{
    RendererOutput renderOut = tracer->DoRenderWork();
    if(renderOut.analytics)
    {
        transferQueue.Enqueue(TracerResponse
        (
            std::in_place_index<TracerResponse::RENDERER_ANALYTICS>,
            renderOut.analytics.value()
        ));
    }
    if(renderOut.imageOut)
    {
        transferQueue.Enqueue(TracerResponse
        (
            std::in_place_index<TracerResponse::IMAGE_SECTION>,
            renderOut.imageOut.value()
        ));
    }
}

void TracerThread::HandleStartStop(bool newStartStopSignal)
{
    MRAY_LOG("[Tracer]: Start/Stop {}", newStartStopSignal);

    bool isStart = newStartStopSignal;
    isInSleepMode = !isStart;
    isRendering = isStart;
    // If we are previously paused
    // Just continue
    if(isPaused && isStart)
    {
        isPaused = false;
    }
    // If not paused, but start is issued
    // Restart the renderer
    else if(isStart)
    {
        RestartRenderer();
    }
    // We are stopped stop rendering
    else
    {
        isPaused = false;
        tracer->StopRender();
    }

}

void TracerThread::HandlePause()
{
    MRAY_LOG("[Tracer]: Pause");
    isInSleepMode = true;
    isRendering = false;
    isPaused = true;
}

void TracerThread::HandleSceneChange(const std::string& newScene)
{
    MRAY_LOG("[Tracer]: NewScene {}", newScene);
    tracer->ClearAll();
    currentRenderer = INVALID_RENDERER_ID;

    // TODO: Single scene loading, change this later maybe
    // for tracer supporting multiple scenes
    using namespace std::filesystem;
    if(currentScene) currentScene->ClearScene();
    std::string fileExt = path(newScene).extension().string();
    fileExt = fileExt.substr(1);
    SceneLoaderI* loader = sceneLoaders.at(fileExt).get();
    currentScene = loader;
    Expected<TracerIdPack> result = currentScene->LoadScene(*tracer, newScene);
    if(result.has_error())
    {
        MRAY_ERROR_LOG("[Tracer]: Failed to Load Scene\n    {}",
                       result.error().GetError());
        isTerminated = true;
        return;
    }
    else
    {
        sceneIds = std::move(result.value());
        MRAY_LOG("[Tracer]: Scene \"{}\" loaded in {}ms",
                 newScene, sceneIds.loadTimeMS);
    }

    // Commit the surfaces
    MRAY_LOG("[Tracer]: Committing Surfaces...");
    Timer timer; timer.Start();
    //
    auto [sceneAABB, instanceCount,
        accelCount] = tracer->CommitSurfaces();
    currentSceneAABB = sceneAABB;
    //
    timer.Split();
    MRAY_LOG("[Tracer]: Surfaces committed in {}ms\n"
             "    AABB      : {}\n"
             "    Instances : {}\n"
             "    Accels    : {}",
             timer.Elapsed<Millisecond>(),
             currentSceneAABB,
             instanceCount, accelCount);

    currentCamIndex = 0;
    CamSurfaceId camSurf = sceneIds.camSurfaces[currentCamIndex];
    currentCamTransform = tracer->GetCamTransform(camSurf);

    // When new scene is loaded, send the
    // Initial cam transform
    transferQueue.Enqueue(TracerResponse
    (
        std::in_place_index<TracerResponse::CAMERA_INIT_TRANSFORM>,
        currentCamTransform
    ));

    // Scene Analytic Data
    transferQueue.Enqueue(TracerResponse
    (
        std::in_place_index<TracerResponse::SCENE_ANALYTICS>,
        SceneAnalyticData
        {
            .sceneName = newScene,
            .sceneLoadTimeS = sceneIds.loadTimeMS,
            .mediumCount = static_cast<uint32_t>(sceneIds.mediums.size()),
            .primCount = static_cast<uint32_t>(sceneIds.prims.size()),
            .textureCount = static_cast<uint32_t>(sceneIds.textures.size()),
            .surfaceCount = static_cast<uint32_t>(sceneIds.surfaces.size()),
            .lightCount = static_cast<uint32_t>(sceneIds.lightSurfaces.size()),
            .cameraCount = static_cast<uint32_t>(sceneIds.camSurfaces.size()),
            .sceneExtent = currentSceneAABB
        }
    ));
    // Send Used Memory
    transferQueue.Enqueue(TracerResponse
    (
        std::in_place_index<TracerResponse::MEMORY_USAGE>,
        tracer->UsedDeviceMemory()
    ));

    // Recreate the renderer
    if(!currentRendererName.empty())
        currentRenderer = tracer->CreateRenderer(currentRendererName);

    // Restart the renderer
    RestartRenderer();
}

void TracerThread::HandleRendererChange(const std::string& rendererName)
{
    MRAY_LOG("[Tracer]: NewRenderer {}", rendererName);
    currentRendererName = rendererName;
    currentRenderLogic0 = 0;
    currentRenderLogic1 = 0;
    if(currentRenderer != INVALID_RENDERER_ID)
        tracer->DestroyRenderer(currentRenderer);
    currentRenderer = tracer->CreateRenderer(currentRendererName);

    RestartRenderer();
}

void TracerThread::LoopWork()
{
    Optional<CameraTransform>       transform;
    Optional<std::string>           rendererName;
    Optional<uint32_t>              renderLogic0;
    Optional<uint32_t>              renderLogic1;
    Optional<uint32_t>              cameraIndex;
    Optional<std::string>           scenePath;
    Optional<TimelineSemaphore*>    syncSem;
    Optional<Float>                 time;
    Optional<bool>                  pauseContinue;
    Optional<bool>                  startStop;
    bool                            hdrSaveDemand = false;
    bool                            sdrSaveDemand = false;
    Optional<std::string>           initialRenderConfig;

    auto ProcessCommand = [&](VisorAction command)
    {
        bool stopConsuming = false;
        using ActionType = typename VisorAction::Type;
        ActionType tp = static_cast<ActionType>(command.index());
        switch(tp)
        {
            using enum ActionType;
            case CHANGE_CAMERA: cameraIndex = std::get<CHANGE_CAMERA>(command); break;
            case CHANGE_CAM_TRANSFORM: transform = std::get<CHANGE_CAM_TRANSFORM>(command); break;
            case CHANGE_RENDERER: rendererName = std::get<CHANGE_RENDERER>(command); break;
            case CHANGE_RENDER_LOGIC0: renderLogic0 = std::get<CHANGE_RENDER_LOGIC0>(command); break;
            case CHANGE_RENDER_LOGIC1: renderLogic1 = std::get<CHANGE_RENDER_LOGIC1>(command); break;
            case CHANGE_TIME: time = std::get<CHANGE_TIME>(command); break;
            case LOAD_SCENE: scenePath = std::get<LOAD_SCENE>(command); break;
            case SEND_SYNC_SEMAPHORE: syncSem = std::get<SEND_SYNC_SEMAPHORE>(command); break;
            case DEMAND_HDR_SAVE: hdrSaveDemand = true; break;
            case DEMAND_SDR_SAVE: sdrSaveDemand = true; break;
            case KICKSTART_RENDER:
            {
                initialRenderConfig = std::get<KICKSTART_RENDER>(command);
                stopConsuming = true;
                break;
            }
            case PAUSE_RENDER:
            {
                pauseContinue = std::get<PAUSE_RENDER>(command);
                stopConsuming = true;
                break;
            }
            case START_STOP_RENDER:
            {
                startStop = std::get<START_STOP_RENDER>(command);
                stopConsuming = true;
                break;
            }
            default:
            {
                MRAY_WARNING_LOG("[Tracer] Unkown visor action is ignored!");
                break;
            }
        }
        return stopConsuming;
    };

    auto CheckQueueAndExit = [this]()
    {
        if(transferQueue.IsTerminated())
        {
            isTerminated = true;
            return true;
        }
        return false;
    };
    try
    {
        // If we are on sleep mode (not rendering probably)
        // Do blocking wait
        VisorAction command;
        if(isInSleepMode)
        {
            transferQueue.Dequeue(command);
            if(CheckQueueAndExit()) return;
            ProcessCommand(command);
        }
        // On every "frame", we will do the latest common commands
        // Low latency commands should be transform commands probably
        else while(transferQueue.TryDequeue(command))
        {
            // Technically this loop may not terminate,
            // if stuff comes too fast. But we just setting some data so
            // it should not be possible
            bool stopConsuming = ProcessCommand(command);
            if(stopConsuming) break;
        }

        if(hdrSaveDemand)
        {
            MRAY_LOG("[Tracer]: Delegate HDR save");
            transferQueue.Enqueue(TracerResponse
            (
                std::in_place_index<TracerResponse::SAVE_AS_HDR>,
                RenderImageSaveInfo
                {
                    .prefix = "exr",
                    .time = 3.0f,
                    .sample = 13
                }
            ));
        }

        if(sdrSaveDemand)
        {
            MRAY_LOG("[Tracer]: Delegate SDR save");
            transferQueue.Enqueue(TracerResponse
            (
                std::in_place_index<TracerResponse::SAVE_AS_SDR>,
                RenderImageSaveInfo
                {
                    .prefix = "png",
                    .time = 3.0f,
                    .sample = 13
                }
            ));
        }



        // Initial render config
        if(initialRenderConfig)
        {
            MRAY_LOG("[Tracer]: Initial render config {}", initialRenderConfig.value());
            if(MRayError e = CreateRendererFromConfig(initialRenderConfig.value()))
            {
                MRAY_ERROR_LOG("[Tracer]: Failed to Load Render Config\n"
                               "    {}", e.GetError());
                isTerminated = true;
                return;
            }
        }
        // New camera!
        if(cameraIndex)
        {
            MRAY_LOG("[Tracer]: NewCamera {}", cameraIndex.value());
            currentCamIndex = cameraIndex.value();

            CamSurfaceId camSurf = sceneIds.camSurfaces[currentCamIndex];
            currentCamTransform = tracer->GetCamTransform(camSurf);

            // When cam is changed send the initial transform
            transferQueue.Enqueue(TracerResponse
            (
                std::in_place_index<TracerResponse::CAMERA_INIT_TRANSFORM>,
                currentCamTransform
            ));

            RestartRenderer();
        }
        // New transform
        if(transform)
        {
            currentCamTransform = transform.value();

            MRAY_LOG("[Tracer]: NewTransform G{}, P{}, U{}",
                     currentCamTransform.gazePoint,
                     currentCamTransform.position,
                     currentCamTransform.up);

            RestartRenderer();
        }
        // New renderer
        if(rendererName) HandleRendererChange(rendererName.value());
        // New semaphore
        if(syncSem)
        {
            MRAY_LOG("[Tracer]: NewSem {:p}", static_cast<void*>(syncSem.value()));
            currentSem = syncSem.value();
        }
        // New scene
        if(scenePath) HandleSceneChange(scenePath.value());
        // Start/Stop
        if(startStop) HandleStartStop(startStop.value());
        // Pause/Continue
        if(pauseContinue) HandlePause();
        // If we are rendering continue...
        if(isRendering) HandleRendering();
        // TODO: Support scene time change
        if(time) MRAY_WARNING_LOG("[Tracer]: Scene time change is not supported!");
        // Render logic changes
        if(renderLogic0)
        {
            MRAY_LOG("[Tracer]: NewRenderLogic0 {}", renderLogic0.value());
            currentRenderLogic0 = renderLogic0.value();
            RestartRenderer();

        }
        if(renderLogic1)
        {
            MRAY_LOG("[Tracer]: NewRenderLogic1 {}", renderLogic1.value());
            currentRenderLogic1 = renderLogic1.value();
            RestartRenderer();
        }
        // Here check the queue's situation
        // and do not iterate again if queues are closed
        if(CheckQueueAndExit()) return;
    }
    catch(const MRayError& e)
    {
        MRAY_ERROR_LOG("[Tracer]: Fatal Error! \"{}\"", e.GetError());
        isTerminated = true;
    }
    catch(const std::exception& e)
    {
        MRAY_ERROR_LOG("[Tracer]: Unkown Error! \"{}\"", e.what());
        isTerminated = true;
    }
}

void TracerThread::InitialWork()
{
    // Initialize device context (i.e. CUDA context)
    auto InitDeviceContext = tracer->GetThreadInitFunction();
    InitDeviceContext();

    // Vector of string_view to string conversion
    auto Convert = [](auto&& svVec)
    {
        std::vector<std::string> result(svVec.size());
        for(size_t i = 0; i < svVec.size(); i++)
            result[i] = svVec[i];
        return result;
    };

    // Send all types
    transferQueue.Enqueue(TracerResponse
    (
        std::in_place_index<TracerResponse::TRACER_ANALYTICS>,
        TracerAnalyticData
        {
            .camTypes = Convert(tracer->CameraGroups()),
            .lightTypes = Convert(tracer->LightGroups()),
            .primTypes = Convert(tracer->PrimitiveGroups()),
            .mediumTypes = Convert(tracer->MediumGroups()),
            .materialTypes = Convert(tracer->MaterialGroups()),
            .rendererTypes = Convert(tracer->Renderers()),
            .tracerColorSpace = tracer->Parameters().globalTextureColorSpace,
            .totalGPUMemoryBytes = tracer->TotalDeviceMemory()
        }
    ));
}

void TracerThread::FinalWork()
{
    // Terminate the queue manually if we are crashing
    if(isTerminated)
    {
        transferQueue.Terminate();
    }
}

TracerThread::TracerThread(TransferQueue& queue,
                           BS::thread_pool& tp)
    : dllFile{nullptr}
    , tracer{nullptr, nullptr}
    , transferQueue(queue.GetTracerView())
    , threadPool(tp)
    , currentRenderer(INVALID_RENDERER_ID)
{}

MRayError TracerThread::MTInitialize(const std::string& tracerConfigFile)
{
    using namespace std::string_literals;
    using namespace std::string_view_literals;
    // TODO: Get this from CMake later
    static const auto MRAY_SCENE_LOADER_LIB_NAME = "SceneLoaderMRay"s;
    static const auto MRAY_SCENE_LOADER_LIB_C_NAME = "ConstructSceneLoaderMRay"s;
    static const auto MRAY_SCENE_LOADER_LIB_D_NAME = "DestroySceneLoaderMRay"s;
    static constexpr auto MRAY_SCENE_LOADER_EXT_NAME = "json"sv;

    // Emplace dll's
    // TODO: Maybe load on demand later
    sceneLoaderDLLs.emplace(MRAY_SCENE_LOADER_LIB_NAME, MRAY_SCENE_LOADER_LIB_NAME);

    // Create loaders
    auto& sceneLoaderDLL = sceneLoaderDLLs.at(MRAY_SCENE_LOADER_LIB_NAME);
    sceneLoaders.emplace(MRAY_SCENE_LOADER_EXT_NAME, SceneLoaderPtr{nullptr, nullptr});
    MRayError e = sceneLoaderDLL.GenerateObjectWithArgs<SceneLoaderConstructorArgs>
    (
        sceneLoaders.at(MRAY_SCENE_LOADER_EXT_NAME),
        SharedLibArgs
        {
            MRAY_SCENE_LOADER_LIB_C_NAME,
            MRAY_SCENE_LOADER_LIB_C_NAME
        },
        threadPool
    );
    if(e) return e;

    Expected<TracerConfig> tConfE = LoadTracerConfig(tracerConfigFile);
    if(tConfE.has_error()) return tConfE.error();
    const TracerConfig& tracerConfig = tConfE.value();

    SharedLibArgs args =
    {
        tracerConfig.dllCreateFuncName,
        tracerConfig.dllDeleteFuncName
    };
    dllFile = std::make_unique<SharedLibrary>(tracerConfig.dllName);
    MRayError err = dllFile->GenerateObjectWithArgs<TracerConstructorArgs, TracerI>(tracer, args,
                                                                                    tracerConfig.params);
    if(err) return err;

    tracer->SetThreadPool(threadPool);
    return MRayError::OK;
}

bool TracerThread::InternallyTerminated() const
{
    return isTerminated;
}

GPUThreadInitFunction TracerThread::GetThreadInitFunction() const
{
    return tracer->GetThreadInitFunction();
}

void TracerThread::SetInitialResolution(const Vector2ui& res,
                                        const Vector2ui& rMin,
                                        const Vector2ui& rMax)
{
    resolution = res;
    regionMin = rMin;
    regionMax = rMax;
}