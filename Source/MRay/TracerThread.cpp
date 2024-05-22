#include "TracerThread.h"
#include <BS/BS_thread_pool.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

struct TracerConfig
{
    std::string dllName;
    std::string dllCreateFuncName = "ConstructTracer";
    std::string dllDeleteFuncName = "DestroyTracer";
};


Expected<TracerConfig> LoadTracerConfig(const std::string& configJsonPath)
{
    using namespace std::literals;

    // Object keys
    static constexpr auto DLL_NAME          = "TracerDLL"sv;
    static constexpr auto OPTIONS_NAME      = "TracerOptions"sv;
    // DLL entries
    static constexpr auto DLL_FILE_NAME         = "name"sv;
    static constexpr auto DLL_CONSTRUCT_NAME    = "construct"sv;
    static constexpr auto DLL_DESTRUCT_NAME     = "destruct"sv;
    // Options

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
        // ...

        return config;
    }
    catch(const nlohmann::json::exception& e)
    {
        return MRayError("Json Error ({})", std::string(e.what()));
    }
}

void TracerThread::LoopWork()
{
    Optional<CameraTransform>       transform;
    Optional<uint32_t>              rendererIndex;
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
            case CHANGE_RENDERER: rendererIndex = std::get<CHANGE_RENDERER>(command); break;
            case CHANGE_RENDER_LOGIC0: renderLogic0 = std::get<CHANGE_RENDER_LOGIC0>(command); break;
            case CHANGE_RENDER_LOGIC1: renderLogic1 = std::get<CHANGE_RENDER_LOGIC1>(command); break;
            case CHANGE_TIME: time = std::get<CHANGE_TIME>(command); break;
            case LOAD_SCENE: scenePath = std::get<LOAD_SCENE>(command); break;
            case SEND_SYNC_SEMAPHORE: syncSem = std::get<SEND_SYNC_SEMAPHORE>(command); break;
            case DEMAND_HDR_SAVE: hdrSaveDemand = true; break;
            case DEMAND_SDR_SAVE: sdrSaveDemand = true; break;
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

    // New scene!
    if(scenePath)
    {
        MRAY_LOG("[Tracer]: NewScene {}", scenePath.value());

        // When new scene is loaded, send the
        // Initial cam transform
        transferQueue.Enqueue(TracerResponse
        (
            std::in_place_index<TracerResponse::CAMERA_INIT_TRANSFORM>,
            CameraTransform
            {
                .position = Vector3::Zero(),
                .gazePoint = Vector3(0, 0, -1),
                .up = Vector3(0, 1, 0)
            }
        ));

        // Scene Analytic Data
        transferQueue.Enqueue(TracerResponse
        (
            std::in_place_index<TracerResponse::SCENE_ANALYTICS>,
            SceneAnalyticData
            {
                .sceneName = scenePath.value(),
                .sceneLoadTimeS = 10.0,
                .mediumCount = 3,
                .primCount = 5,
                .textureCount = 7,
                .surfaceCount = 9,
                .cameraCount = 11,
                .sceneExtent = AABB3(Vector3(0.0, 0.0, 0.0),
                                     Vector3(10.0, 10.0, 10.0)),
                .timeRange = Vector2(1, 10)
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
                .totalGPUMemoryMiB = 8000.0,
                .usedGPUMemoryMiB = 100.0
            }
        ));

        //using namespace std::filesystem;
        //tracer->ClearAll();
        //// TODO: Single scene loading, change this later maybe
        //// for tracer supporting multiple scenes
        //if(currentScene) currentScene->ClearScene();
        //std::string fileExt = path(scenePath.value()).extension().string();
        //SceneLoaderI* loader = sceneLoaders.at(fileExt).get();
        //currentScene = loader;
        //currentScene->LoadScene(*tracer, scenePath.value());
    }

    // New camera!
    if(cameraIndex)
    {
        MRAY_LOG("[Tracer]: NewCamera {}", cameraIndex.value());

        // When cam is changed send the initial transform
        // Initial cam transform
        transferQueue.Enqueue(TracerResponse
        (
            std::in_place_index<TracerResponse::CAMERA_INIT_TRANSFORM>,
            CameraTransform
            {
                .position = Vector3::Zero(),
                .gazePoint = Vector3(0, 0, -1),
                .up = Vector3(0, 1, 0)
            }
        ));

        //// New Camera
        //// Stop rendering
        //tracer->StopRender();
        ////transferQueue.Enqueue(...);

        ////tracer->...
        //// Change camera
        //// send the initial transform
        //transferQueue.Enqueue(CameraTransform
        //{
        //    .position = Vector3::Zero(),
        //    .gazePoint = Vector3::Zero(),
        //    .up = Vector3::Zero()
        //});
    }

    // New transform!
    if(transform)
    {
        const auto& t = transform.value();
        MRAY_LOG("[Tracer]: NewTransform G{}, P{}, U{}",
                 t.gazePoint,
                 t.position,
                 t.up);

        //tracer->StopRender();
        //tracer->StartRender(..., ..., ...);
    }

    // New renderer!
    if(rendererIndex)
    {
        MRAY_LOG("[Tracer]: NewRenderer {}", rendererIndex.value());
        //// Initial cam transform
        //transferQueue.Enqueue(TracerResponse
        //(
        //    std::in_place_index<TracerResponse::CAMERA_INIT_TRANSFORM>,
        //    CameraTransform
        //    {
        //        .position = Vector3(1, 2, 3),
        //        .gazePoint = Vector3(4, 5, 6),
        //        .up = Vector3(7, 8, 9)
        //    }
        //));
    }

    if(renderLogic0)
    {
        MRAY_LOG("[Tracer]: NewRenderLogic0 {}", renderLogic0.value());
    }

    if(renderLogic1)
    {
        MRAY_LOG("[Tracer]: NewRenderLogic1 {}", renderLogic1.value());
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

    // New time!
    if(time)
    {
        MRAY_LOG("[Tracer]: NewTime {}", time.value());
        // Update stuff
        //currentScene->UpdateScene();
    }

    // Pause/Continue!
    if(pauseContinue)
    {
        isInSleepMode = pauseContinue.value();
        MRAY_LOG("[Tracer]: Pause/Cont {}", pauseContinue.value());
        isRendering = false;
    }

    // Start/Stop!
    if(startStop)
    {
        isInSleepMode = !startStop.value();
        isRendering = startStop.value();
        MRAY_LOG("[Tracer]: Start/Stop {}", startStop.value());
    }

    if(syncSem)
    {
        MRAY_LOG("[Tracer]: NewSem {}", syncSem.value());
    }

    //static uint64_t i = 0;
    //MRAY_LOG("[Tracer]: Loop {}", i);
    //i++;

    // If we are rendering continue...
    if(isRendering)
    {
        RendererOutput renderOut; // = tracer->DoRenderWork();
        // if(renderOut.analytics)
        // if(renderOut.imageOut)

        transferQueue.Enqueue(TracerResponse
        (
            std::in_place_index<TracerResponse::RENDERER_ANALYTICS>,
            RendererAnalyticData
            {
                .throughput = 3.32,
                .throughputSuffix = "M rays/sec",
                .workPerPixel = 512,
                .workPerPixelSuffix = "spp",
                .iterationTimeMS = 20,
                .renderResolution = Vector2ui(1920, 1080),
                .outputColorSpace = MRayColorSpaceEnum::MR_ACES_CG,
                .customLogicSize0 = 3,
                .customLogicSize1 = 10
            }
        ));

        using namespace std::chrono_literals;
        std::this_thread::sleep_for(3ms);
    }


    // Here check the queue's situation
    // and do not iterate again if queues are closed
    if(CheckQueueAndExit()) return;
}

void TracerThread::InitialWork()
{
    // Do a handshake
    // Send the initial tracer state

    //tracer->
    //transferQueue.Enqueue(TracerResponse
    //(
    //    std::in_place_index<TracerResponse::TRACER_ANALYTICS>,
    //    TracerAnalyticData
    //    {
    //        .camTypes = {{"A", 1}},
    //        .lightTypes = {{"B", 1}},
    //        .primTypes = {{"C", 1}},
    //        .mediumTypes = {{"D", 1}},
    //        .materialTypes = {{"E", 1}},
    //        .rendererTypes =
    //        {
    //            "TexDisplay",
    //            "DirectTracer",
    //            "PathTracer",
    //            "AOTracer",
    //            "PhotonMapper"
    //        },
    //        .tracerColorSpace = MRayColorSpaceEnum::MR_ACES_CG,
    //        .totalGPUMemoryMiB = 8000.0,
    //        .usedGPUMemoryMiB = 100.0
    //    }
    //));

}

void TracerThread::FinalWork()
{
    //if(fatalErrorOccured)

}

TracerThread::TracerThread(TransferQueue& queue, BS::thread_pool& tp)
    : dllFile{nullptr}
    , tracer{nullptr, nullptr}
    , transferQueue(queue.GetTracerView())
    , threadPool(tp)
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
    if(e) throw e;
}

MRayError TracerThread::MTInitialize(const std::string& tracerConfigFile)
{
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
                                                                                    threadPool);
    if(err) return err;
    return MRayError::OK;
}

bool TracerThread::InternallyTerminated() const
{
    return isTerminated;
}