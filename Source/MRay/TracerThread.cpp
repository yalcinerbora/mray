#include "TracerThread.h"
#include <BS/BS_thread_pool.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

#include <Core/Timer.h>

std::string GetTypeNameFromRenderConfig(const std::string&)
{
    //....
    return std::string();
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
    static constexpr auto DLL_NAME              = "TracerDLL"sv;
    static constexpr auto PARAMETERS_NAME       = "Parameters"sv;
    // DLL entries
    static constexpr auto DLL_FILE_NAME         = "name"sv;
    static constexpr auto DLL_CONSTRUCT_NAME    = "construct"sv;
    static constexpr auto DLL_DESTRUCT_NAME     = "destruct"sv;
    // Params
    static constexpr auto SEED_NAME             = "seed"sv;
    static constexpr auto ACCEL_TYPE_NAME       = "acceleratorType"sv;
    static constexpr auto ITEM_POOL_NAME        = "itemPoolSize"sv;
    static constexpr auto SAMPLER_TYPE_NAME     = "samplerType"sv;
    static constexpr auto CLAMP_TEX_RES_NAME    = "clampTexRes"sv;
    static constexpr auto PARTITION_LOGIC_NAME  = "partitionLogic"sv;

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
        OptionalFetch(config.params.itemPoolSize, ITEM_POOL_NAME, paramsJson);
        OptionalFetch(config.params.samplerType, SAMPLER_TYPE_NAME, paramsJson);
        OptionalFetch(config.params.clampedTexRes, CLAMP_TEX_RES_NAME, paramsJson);
        //OptionalFetch(config.params., PARTITION_LOGIC_NAME, paramsJson);


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

void TracerThread::SetRendererParams(const std::string&)
{

}

void TracerThread::RestartRenderer()
{

}

void TracerThread::LoopWork()
{
    Optional<CameraTransform>       transform;
    Optional<std::string>           rendererName;
    Optional<uint32_t>              renderLogic0;
    Optional<uint32_t>              renderLogic1;
    Optional<uint32_t>              cameraIndex;
    Optional<std::string>           scenePath;
    Optional<SystemSemaphoreHandle> syncSem;
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

        if(initialRenderConfig)
        {
            if(currentRenderer != std::numeric_limits<RendererId>::max())
                tracer->DestroyRenderer(currentRenderer);

            MRAY_LOG("[Tracer]: Initial render config {}", initialRenderConfig.value());
            //std::string rName = GetTypeNameFromRenderConfig(initialRenderConfig.value());
            //currentRenderer = tracer->CreateRenderer(rName);
            //SetRendererParams(initialRenderConfig.value());
            //tracer->CommitRendererReservations(currentRenderer);
        }

        // New scene!
        if(scenePath)
        {
            MRAY_LOG("[Tracer]: NewScene {}", scenePath.value());
            tracer->ClearAll();

            // TODO: Single scene loading, change this later maybe
            // for tracer supporting multiple scenes
            using namespace std::filesystem;
            if(currentScene) currentScene->ClearScene();
            std::string fileExt = path(scenePath.value()).extension().string();
            fileExt = fileExt.substr(1);
            SceneLoaderI* loader = sceneLoaders.at(fileExt).get();
            currentScene = loader;
            Expected<TracerIdPack> result = currentScene->LoadScene(*tracer, scenePath.value());
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
                         scenePath.value(),
                         sceneIds.loadTimeMS);
            }

            // Commit the surfaces
            MRAY_LOG("[Tracer]: Committing Surfaces...");
            Timer timer; timer.Start();
            //
            auto [currentSceneAABB, instanceCount,
                  accelCount] = tracer->CommitSurfaces();
            //
            timer.Split();
            MRAY_LOG("[Tracer]: Surfaces committed in {}ms\n"
                     "    AABB      : {}\n"
                     "    Instances : {}\n"
                     "    Accels    : {}",
                     timer.Elapsed<Millisecond>(),
                     currentSceneAABB,
                     instanceCount, accelCount);
            MRAY_LOG("DevMem {}MiB", tracer->UsedDeviceMemory() >> 20);

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
                    .sceneName = scenePath.value(),
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

            // Send Tracer Analytics
            transferQueue.Enqueue(TracerResponse
            (
                std::in_place_index<TracerResponse::TRACER_ANALYTICS>,
                TracerAnalyticData
                {
                    .camTypes = {{"A", 1}},
                    .lightTypes = {{"B", 1}},
                    .primTypes = {{"C", 1}},
                    .mediumTypes = {{"D", 1}},
                    .materialTypes = {{"E", 1}},
                    .rendererTypes =
                    {
                        "TexDisplay",
                        "DirectTracer",
                        "PathTracer",
                        "AOTracer",
                        "PhotonMapper"
                    },
                    .tracerColorSpace = MRayColorSpaceEnum::MR_ACES_CG,
                    .totalGPUMemoryBytes = tracer->TotalDeviceMemory(),
                    .usedGPUMemoryBytes = tracer->UsedDeviceMemory()
                }
            ));


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

        // New transform!
        if(transform)
        {
            currentCamTransform = transform.value();

            MRAY_LOG("[Tracer]: NewTransform G{}, P{}, U{}",
                     currentCamTransform.gazePoint,
                     currentCamTransform.position,
                     currentCamTransform.up);

            RestartRenderer();
        }

        // New renderer!
        if(rendererName)
        {
            MRAY_LOG("[Tracer]: NewRenderer {}", rendererName.value());

            if(currentRenderer != std::numeric_limits<RendererId>::max())
                tracer->DestroyRenderer(currentRenderer);
            currentRenderer = tracer->CreateRenderer(rendererName.value());
            tracer->CommitRendererReservations(currentRenderer);

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

        if(syncSem)
        {
            MRAY_LOG("[Tracer]: NewSem {}", syncSem.value());
            currentSem = syncSem.value();
        }

        // New time!
        if(time)
        {
            MRAY_LOG("[Tracer]: NewTime {}", time.value());
            // TODO: Update scene etc...
        }

        // Pause/Continue!
        if(pauseContinue)
        {
            isInSleepMode = pauseContinue.value();
            MRAY_LOG("[Tracer]: Pause/Cont {}", pauseContinue.value());
            isRendering = false;
        }


        if(renderLogic0)
        {
            MRAY_LOG("[Tracer]: NewRenderLogic0 {}", renderLogic0.value());
        }

        if(renderLogic1)
        {
            MRAY_LOG("[Tracer]: NewRenderLogic1 {}", renderLogic1.value());
        }

        // Start/Stop!
        if(startStop)
        {
            bool isStart = startStop.value();
            isInSleepMode = !isStart;
            isRendering = isStart;
            MRAY_LOG("[Tracer]: Start/Stop {}", isStart);

            if(isStart)
            {
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
                                                           rp, currentCamTransform);

                transferQueue.Enqueue(TracerResponse
                (
                    std::in_place_index<TracerResponse::RENDER_BUFFER_INFO>,
                    rbi
                ));
            }
            else
            {
                tracer->StopRender();
            }
        }


        // If we are rendering continue...
        if(isRendering)
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

            //using namespace std::chrono_literals;
            //std::this_thread::sleep_for(3ms);
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

    // Send the renderers initially
    // Send the initial tracer state
    auto rendererTypesSV = tracer->Renderers();
    std::vector<std::string> rendererTypes(rendererTypesSV.size());
    for(size_t i = 0; i < rendererTypes.size(); i++)
    {
        rendererTypes[i] = rendererTypesSV[i];
    }

    transferQueue.Enqueue(TracerResponse
    (
        std::in_place_index<TracerResponse::TRACER_ANALYTICS>,
        TracerAnalyticData
        {
            .camTypes = {},
            .lightTypes = {},
            .primTypes = {},
            .mediumTypes = {},
            .materialTypes = {},
            .rendererTypes = rendererTypes,
            .tracerColorSpace = MRayColorSpaceEnum::MR_ACES_CG,
            .totalGPUMemoryBytes = tracer->TotalDeviceMemory(),
            .usedGPUMemoryBytes = tracer->UsedDeviceMemory()
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
    , currentRenderer(std::numeric_limits<RendererId>::max())
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