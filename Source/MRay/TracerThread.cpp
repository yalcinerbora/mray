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
    Optional<uint32_t>              cameraIndex;
    Optional<std::string>           scenePath;
    Optional<Float>                 time;
    Optional<bool>                  pauseContinue;
    Optional<bool>                  startStop;
    Optional<SystemSemaphoreHandle> syncSem;

    // On every "frame", we will do the latest common commands
    VisorAction command;
    while(transferQueue.TryDequeue(command))
    {
        bool stopConsuming = false;
        // Technically this loop may not terminate,
        // if stuff comes too fast. But we just setting some data so
        // it should not be possible
        using ActionType = typename VisorAction::Type;
        ActionType tp = static_cast<ActionType>(command.index());
        switch(tp)
        {
            using enum ActionType;
            case CHANGE_CAMERA: cameraIndex = std::get<CHANGE_CAMERA>(command); break;
            case CHANGE_CAM_TRANSFORM: transform = std::get<CHANGE_CAM_TRANSFORM>(command); break;
            case CHANGE_RENDERER: rendererIndex = std::get<CHANGE_RENDERER>(command); break;
            case CHANGE_TIME: time = std::get<CHANGE_TIME>(command); break;
            case LOAD_SCENE: scenePath = std::get<LOAD_SCENE>(command); break;
            case SEND_SYNC_SEMAPHORE: syncSem = std::get<SEND_SYNC_SEMAPHORE>(command); break;
            case PAUSE_RENDER:
            {
                pauseContinue = std::get<PAUSE_RENDER>(command);
                sleepMode = !pauseContinue.value();
                stopConsuming = true;
                break;
            }
            case START_STOP_RENDER:
            {
                startStop = std::get<START_STOP_RENDER>(command);
                sleepMode = startStop.value();
                stopConsuming = true;
                break;
            }
            default:
            {
                MRAY_ERROR_LOG("Unknown command Id!");
                fatalErrorOccured = true;
            }
        }
        //
        if(stopConsuming) break;
    }

    // New scene!
    if(scenePath)
    {
        using namespace std::filesystem;
        tracer->ClearAll();
        // TODO: Single scene loading, change this later maybe
        // for tracer supporting multiple scenes
        if(currentScene) currentScene->ClearScene();
        std::string fileExt = path(scenePath.value()).extension().string();
        SceneLoaderI* loader = sceneLoaders.at(fileExt).get();
        currentScene = loader;
        currentScene->LoadScene(*tracer, scenePath.value());
    }

    // New camera!
    if(cameraIndex)
    {
        // New Camera
        // Stop rendering
        tracer->StopRender();
        //transferQueue.Enqueue(...);

        //tracer->...
        // Change camera
        // send the initial transform
        transferQueue.Enqueue(CameraTransform
        {
            .position = Vector3::Zero(),
            .gazePoint = Vector3::Zero(),
            .up = Vector3::Zero()
        });
    }

    // New transform!
    if(transform)
    {
        tracer->StopRender();
        //tracer->StartRender(..., ..., ...);
    }

    // New renderer!
    if(rendererIndex)
    {

    }

    // New time!
    if(time)
    {
        // Update stuff
        //currentScene->UpdateScene();
    }

    // Pause/Continue!
    if(pauseContinue)
    {

    }

    // Start/Stop!
    if(startStop)
    {

    }

    // If we are rendering continue...
    if(isRendering)
    {
        RendererOutput renderOut = tracer->DoRenderWork();
        if(renderOut.analytics) transferQueue.Enqueue(renderOut.analytics.value());
        if(renderOut.imageOut) transferQueue.Enqueue(renderOut.imageOut.value());
    }

}

void TracerThread::InitialWork()
{}

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
    return fatalErrorOccured;
}