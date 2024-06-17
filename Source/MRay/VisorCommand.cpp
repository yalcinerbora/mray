#include "VisorCommand.h"

#include <CLI/CLI.hpp>
#include <string_view>
#include <BS/BS_thread_pool.hpp>

#include "Visor/EntryPoint.h"

#include "Core/SharedLibrary.h"
#include "Core/System.h"
#include "Core/Error.h"
#include "Core/Error.hpp"

#include "TracerThread.h"

#include <nlohmann/json.hpp>

// TODO: Code duplication!
// Vector type read is also on scene loader json dll
template<ArrayLikeC T>
void from_json(const nlohmann::json& n, T& out)
{
    using IT = typename T::InnerType;
    using S = Span<IT, T::Dims>;
    std::array<IT, T::Dims> a = n;
    out = T(ToConstSpan(S(a)));
}

Expected<VisorConfig> LoadVisorConfig(const std::string& configJsonPath)
{
    using namespace std::literals;

    // Object keys
    static constexpr auto DLL_NAME          = "VisorDLL"sv;
    static constexpr auto OPTIONS_NAME      = "VisorOptions"sv;
    static constexpr auto INTERVAL_OUT_NAME = "IntervalOutput"sv;
    // DLL entries
    static constexpr auto DLL_FILE_NAME         = "name"sv;
    static constexpr auto DLL_CONSTRUCT_NAME    = "construct"sv;
    static constexpr auto DLL_DESTRUCT_NAME     = "destruct"sv;
    // Options
    static constexpr auto OPT_CBUFFER_NAME      = "commandBufferSize"sv;
    static constexpr auto OPT_RBUFFER_NAME      = "responseBufferSize"sv;
    static constexpr auto OPT_ENFORCE_IGPU_NAME = "enforceIGPU"sv;
    static constexpr auto OPT_DISPLAY_HDR_NAME  = "displayHDR"sv;
    static constexpr auto OPT_WINDOW_SIZE_NAME  = "windowSize"sv;
    static constexpr auto OPT_REAL_TIME_NAME    = "realTime"sv;

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
            return MRayError("Visor config file \"{}\" is not found",
                             configJsonPath);
        configJson = nlohmann::json::parse(file, nullptr, true, true);

        // DLL portion
        VisorConfig config;
        const auto& dllJson = configJson[DLL_NAME];
        config.dllName = dllJson[DLL_FILE_NAME];
        OptionalFetch(config.dllCreateFuncName, DLL_CONSTRUCT_NAME, dllJson);
        OptionalFetch(config.dllDeleteFuncName, DLL_DESTRUCT_NAME, dllJson);

        //
        const auto& optJson = configJson[OPTIONS_NAME];
        OptionalFetch(config.commandBufferSize, OPT_CBUFFER_NAME, optJson);
        OptionalFetch(config.responseBufferSize, OPT_RBUFFER_NAME, optJson);
        OptionalFetch(config.enforceIGPU, OPT_ENFORCE_IGPU_NAME, optJson);
        OptionalFetch(config.displayHDR, OPT_DISPLAY_HDR_NAME, optJson);
        OptionalFetch(config.realTime, OPT_REAL_TIME_NAME, optJson);
        OptionalFetch(config.wSize, OPT_WINDOW_SIZE_NAME, optJson);

        // TODO: Add interval output config reading
        // ...

        return config;
    }
    catch(const nlohmann::json::exception& e)
    {
        return MRayError("Json Error ({})", std::string(e.what()));
    }
}

namespace MRayCLI::VisorNames
{
    using namespace std::literals;
    static constexpr auto Name = "visor"sv;
    static constexpr auto Description = "enables the visor system for interactive "
                                        "rendering"sv;
};

MRayError VisorCommand::Invoke()
{
    // Load the config here, instead of delegating to the
    // implemented interface (since we need the dll file anyway)
    Expected<VisorConfig> vConfE = LoadVisorConfig(visorConfigFile);
    if(vConfE.has_error()) return vConfE.error();
    const VisorConfig& visorConfig = vConfE.value();

    SharedLibrary lib(visorConfig.dllName);
    SharedLibPtr<VisorI> visorSystem = {nullptr, nullptr};
    SharedLibArgs VisorDLLArgs =
    {
        .mangledConstructorName = visorConfig.dllCreateFuncName,
        .mangledDestructorName = visorConfig.dllDeleteFuncName
    };
    MRayError e = lib.GenerateObjectWithArgs<Tuple<>>(visorSystem,
                                                      VisorDLLArgs);
    if(e) return e;

    // Transfer queue, responsible for communication between main thread
    // (window render thread) and tracer thread
    TransferQueue transferQueue(visorConfig.commandBufferSize,
                                visorConfig.responseBufferSize,
    [&visorSystem]()
    {
        // Trigge event (on glfw) calls "glfwPostEmptyEvent()"
        // so that when tracer isses a response,
        // glfw can process it
        visorSystem->TriggerEvent();
    });

    // Thread pool, many things will need this
    // TODO: Change this to HW concurrency,
    // this is for debugging
    uint32_t threadCount = std::thread::hardware_concurrency();
    //uint32_t threadCount = 1;
    BS::thread_pool threadPool(threadCount);

    // Get the tracer dll
    TracerThread tracerThread(transferQueue, threadPool);
    e = tracerThread.MTInitialize(tracerConfigFile);
    if(e) return e;

    // Reset the thread pool and initialize the threads with GPU specific
    // initialization routine, also change the name of the threads.
    // We need to do this somewhere here, if we do it on tracer side
    // due to passing between dll boundaries, it crash on destruction.
    threadPool.reset(threadCount, [&tracerThread]()
    {
        auto GPUInit = tracerThread.GetThreadInitFunction();
        GPUInit();
    });
    std::vector<std::thread::native_handle_type> handles;
    handles = threadPool.get_native_handles();
    for(size_t i = 0; i < handles.size(); i++)
    {
        using namespace std::string_literals;
        std::string name = "WorkerThread_"s + std::to_string(i);
        RenameThread(handles[i], name);
    }

    // Actual initialization, process path should be called from here
    // In dll it may return .dll's path (on windows, I think)
    std::string processPath = GetProcessPath();
    e = visorSystem->MTInitialize(transferQueue,
                                  &threadPool,
                                  visorConfig,
                                  processPath);
    if(e) return e;


    if(imgRes)
    {
        Vector2ui resolution(imgRes.value()[0], imgRes.value()[1]);
        tracerThread.SetInitialResolution(resolution,
                                          Vector2ui::Zero(),
                                          resolution);
    }

    // Start the tracer thread
    tracerThread.Start("TracerThread");

    // Initially send the scene to tracer
    if(sceneFile)
    {
        using enum VisorAction::Type;
        VisorAction va(std::in_place_index<LOAD_SCENE>, sceneFile.value());
        transferQueue.GetVisorView().Enqueue(std::move(va));
    }

    // Initially send the renderer to tracer
    if(renderConfigFile)
    {
        visorSystem->MTInitiallyStartRender(renderConfigFile.value());
    }

    // ====================== //
    //     Real-time Loop     //
    // ====================== //
    e = MRayError::OK;
    try
    {
        while(!visorSystem->MTIsTerminated() &&
              !transferQueue.GetVisorView().IsTerminated())
        {
            visorSystem->MTWaitForInputs();
            visorSystem->MTRender();
        }
    }
    catch(const MRayError& err)
    {
        e = err;
    }
    catch(const std::exception& err)
    {
        e = MRayError("Unkown Error: {}", err.what());
    }

    // Order is important here
    // First wait the thread pool
    threadPool.wait();
    // Destroy the transfer queue
    // So that the tracer can drop from queue wait
    transferQueue.Terminate();
    // First stop the tracer, since tracer commands
    // submit glfw "empty event" to trigger visor rendering
    tracerThread.Stop();
    // Now we can destory the tracer
    visorSystem->MTDestroy();
    // All Done!
    return e;
}

CLI::App* VisorCommand::GenApp(CLI::App& mainApp)
{
    using namespace MRayCLI::VisorNames;
    CLI::App* visorApp = mainApp.add_subcommand(std::string(Name), std::string(Description));
        //->alias(std::string(Name));

    // Input
    visorApp->add_option("--tracerConf, --tConf"s, tracerConfigFile,
                         "Tracer configuration file, mainly specifies the "
                         "tracer dll name to be loaded."s)
        ->check(CLI::ExistingFile)
        ->required()
        ->expected(1);

    visorApp->add_option("--visorConf, --vConf"s, visorConfigFile,
                         "Visor configuration file."s)
        ->check(CLI::ExistingFile)
        ->required()
        ->expected(1);

    CLI::Option* sceneOption = visorApp->add_option("--scene, -s"s, sceneFile,
                                                    "Initial scene file (optional)."s)
        ->check(CLI::ExistingFile)
        ->expected(1)
        ->required();

    CLI::Option* rendOption = visorApp->add_option("--renderConf, --rConf"s, renderConfigFile,
                                                   "Initial renderer to be launched. "
                                                   "Requires a scene to be set (optional)."s)
        ->check(CLI::ExistingFile)
        ->needs(sceneOption)
        ->expected(1)
        ->required();

    // TODO: Change this to be a region maybe?
    visorApp->add_option("--resolution, -r"s, imgRes,
                         "Initial renderer's resolution. "
                         "Requires a renderer to be set (optional)."s)
        ->check(CLI::Number)
        ->expected(1)
        ->delimiter('x')
        ->needs(rendOption)
        ->required();

    return visorApp;
}

CommandI& VisorCommand::Instance()
{
    static VisorCommand c = {};
    return c;
}