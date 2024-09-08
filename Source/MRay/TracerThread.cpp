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

struct TracerConfig
{
    std::string dllName;
    std::string dllCreateFuncName = "ConstructTracer";
    std::string dllDeleteFuncName = "DestroyTracer";

    TracerParameters params;
};

void from_json(const nlohmann::json& node, FilterType& t)
{
    auto name = node.at(FilterType::TYPE_NAME).get<std::string_view>();
    auto radius = node.at(FilterType::RADIUS_NAME).get<Float>();

    FilterType::E type = FilterType::FromString(name);
    if(type == FilterType::END)
        throw MRayError("Unknown filter type name \"{}\"",
                        name);
    t = FilterType{type, radius};
}

void from_json(const nlohmann::json& node, AcceleratorType& t)
{
    auto name = node.get<std::string_view>();
    AcceleratorType::E type = AcceleratorType::FromString(name);
    if(type == AcceleratorType::END)
        throw MRayError("Unknown accelerator type name \"{}\"",
                        name);
    t = AcceleratorType{type};
}

void from_json(const nlohmann::json& node, SamplerType& t)
{
    auto name = node.get<std::string_view>();
    SamplerType::E type = SamplerType::FromString(name);
    if(type == SamplerType::END)
        throw MRayError("Unknown sampler type name \"{}\"",
                        name);
    t = SamplerType{type};
}

void from_json(const nlohmann::json& node, MRayColorSpaceEnum& t)
{
    auto name = node.get<std::string_view>();
    MRayColorSpaceEnum e = MRayColorSpaceStringifier::FromString(name);
    if(e == MRayColorSpaceEnum::MR_END)
        throw MRayError("Unknown color space \"{}\"", name);
    t = e;
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
    static constexpr auto PAR_HINT_NAME         = "parallelHint"sv;
    static constexpr auto SAMPLER_TYPE_NAME     = "samplerType"sv;
    static constexpr auto CLAMP_TEX_RES_NAME    = "clampTexRes"sv;
    static constexpr auto GEN_MIP_NAME          = "genMipmaps"sv;
    static constexpr auto TEX_COLOR_SPACE_NAME  = "globalTexColorSpace"sv;
    static constexpr auto MIP_GEN_FILTER_NAME   = "mipGenFilter"sv;
    static constexpr auto FILM_FILTER_NAME      = "filmFilter"sv;
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

        // TODO: Add optional config reading
        nlohmann::json paramsJson;
        OptionalFetch(paramsJson, PARAMETERS_NAME, configJson);
        if(paramsJson.empty()) return config;

        OptionalFetch(config.params.seed, SEED_NAME, paramsJson);
        OptionalFetch(config.params.accelMode, ACCEL_TYPE_NAME, paramsJson);
        OptionalFetch(config.params.parallelizationHint, PAR_HINT_NAME, paramsJson);
        OptionalFetch(config.params.samplerType, SAMPLER_TYPE_NAME, paramsJson);
        OptionalFetch(config.params.clampedTexRes, CLAMP_TEX_RES_NAME, paramsJson);
        OptionalFetch(config.params.genMips, GEN_MIP_NAME, paramsJson);
        OptionalFetch(config.params.globalTextureColorSpace, TEX_COLOR_SPACE_NAME, paramsJson);
        OptionalFetch(config.params.mipGenFilter, MIP_GEN_FILTER_NAME, paramsJson);
        OptionalFetch(config.params.filmFilter, FILM_FILTER_NAME, paramsJson);
        // TODO: Add this later
        //OptionalFetch(config.params.partitionLogic, PARTITION_LOGIC_NAME, paramsJson);
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
        RendererAttributeInfoList attributes = tracer->AttributeInfo(currentRenderer);

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
                using MRDataType = std::remove_cvref_t<decltype(t)>;
                using T = MRDataType::Type;
                auto loc = rendererNode.find(name);
                if(optionality != MR_OPTIONAL && loc == rendererNode.end())
                    return MRayError("Config read \"{}\": Mandatory variable \"{}\" "
                                     "for \"{}\" is not found in config file",
                                     configJsonPath, name, rName);
                if(loc == rendererNode.end()) return MRayError::OK;

                T in = loc->get<T>();
                if constexpr(MRDataType::Name == MRayDataEnum::MR_STRING)
                {
                    TransientData data(std::in_place_type_t<T>{}, in.size());
                    data.ReserveAll();
                    Span<char> out = data.AccessAsString();
                    std::copy(in.cbegin(), in.cend(), out.begin());
                    tracer->PushRendererAttribute(currentRenderer, attribIndex,
                                                  std::move(data));
                }
                else
                {
                    TransientData data(std::in_place_type_t<T>{}, 1);
                    data.Push(Span<const T>(&in, 1));
                    tracer->PushRendererAttribute(currentRenderer, attribIndex,
                                                  std::move(data));
                }
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

    // Flush the GPU before freeing memory
    tracer->Flush();
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
        throw MRayError("Failed to Load Scene\n    {}",
                        result.error().GetError());
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
    // We need to flush the Tracer to time.
    // Commit surfaces may be hybrid process (GPU/CPU combo)
    // so we can't directly measure from GPU.
    tracer->Flush();
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
    Optional<SemaphoreInfo>         syncSem;
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
        bool semaphoreDropped = (currentSem.semaphore &&
                                 currentSem.semaphore->IsInvalidated());
        bool queueDropped = transferQueue.IsTerminated();
        isTerminated = (semaphoreDropped || queueDropped);

        if(semaphoreDropped) transferQueue.Terminate();
        if(queueDropped && currentSem.semaphore) currentSem.semaphore->Invalidate();

        if(isTerminated) MRAY_LOG("[Tracer]: Terminating!");
        return isTerminated;
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
            MRAY_LOG("[Tracer]: NewSem {:p} - {:d}",
                     static_cast<void*>(syncSem.value().semaphore),
                     syncSem.value().importMemAlignment);
            currentSem = syncSem.value();
            tracer->SetupRenderEnv(currentSem.semaphore,
                                   currentSem.importMemAlignment, 0);
        }
        // New scene
        if(scenePath) HandleSceneChange(scenePath.value());
        // Start/Stop
        if(startStop) HandleStartStop(startStop.value());
        // Pause/Continue
        if(pauseContinue) HandlePause();
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

        // After handling other things do rendering
        // and continue
        // If we are rendering continue...
        if(isRendering) HandleRendering();

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
    MRayError err = MRayError::OK;
    // GPU System may fail to construct which thorws an exception
    try
    {
        err = dllFile->GenerateObjectWithArgs<TracerConstructorArgs, TracerI>
        (
            tracer, args,
            tracerConfig.params
        );
    }
    catch(const MRayError& e)
    {
        err = e;
    }
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