#include "VisorCommand.h"

#include <CLI/CLI.hpp>
#include <string_view>
#include <BS/BS_thread_pool.hpp>

#include "Visor/EntryPoint.h"

#include "Core/SharedLibrary.h"
#include "Core/System.h"

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
                                        "rendering (default)"sv;
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
    TransferQueue transferQueue(1, 1, [&visorSystem]()
    {
        // Trigge event (on glfw) calls "glfwPostEmptyEvent()"
        // so that when tracer isses a response,
        // glfw can process it
        visorSystem->TriggerEvent();
    });

    // Thread pool, many things will need this
    // TODO: Change this to HW concurrency,
    // this is for debugging
    BS::thread_pool threadPool(1);

    // Get the tracer dll
    TracerThread tracerThread(transferQueue, threadPool);
    //e = tracerThread.MTInitialize(tracerConfigFile);
    //if(e) return e;


    // Actual initialization, process path should be called from here
    // In dll it may return .dll's path (on windows, I think)
    std::string processPath = GetProcessPath();
    e = visorSystem->MTInitialize(transferQueue,
                                  &threadPool,
                                  visorConfig,
                                  processPath);
    if(e) return e;

    // Start the tracer thread
    //tracerThread.Start();

    // ====================== //
    //     Real-time Loop     //
    // ====================== //
    while(!visorSystem->MTIsTerminated())
    {
        visorSystem->MTWaitForInputs();
        visorSystem->MTRender();
    }

    // GG!
    threadPool.wait();
    visorSystem->MTDestroy();
    //tracerThread.Stop();
    return MRayError::OK;
}

CLI::App* VisorCommand::GenApp(CLI::App& mainApp)
{
    using namespace MRayCLI::VisorNames;
    CLI::App* visorApp = mainApp.add_subcommand("", std::string(Description))
        ->alias(std::string(Name));

    //// Input
    //visorApp->add_option("--tracerConf, -t"s, tracerConfigFile,
    //                     "Tracer configuration file, mainly specifies the "
    //                     "tracer dll name to be loaded."s)
    //    ->check(CLI::ExistingFile)
    //    ->required();
    //
    visorApp->add_option("--visorConf, --vConf"s, visorConfigFile,
                         "Visor configuration file."s)
        ->check(CLI::ExistingFile)
        ->required();
    //
    //CLI::Option* sceneOption = visorApp->add_option("--scene, -s"s, sceneFile,
    //                                                "Initial scene file (optional)."s)
    //    ->check(CLI::ExistingFile);

    //CLI::Option* rendOption = visorApp->add_option("--renderConf, -r"s, renderConfig,
    //                                               "Initial renderer to be launched. "
    //                                               "Requires a scene to be set (optional)."s)
    //    ->check(CLI::ExistingFile)
    //    ->needs(sceneOption);

    //// TODO: Change this to be a region maybe?
    //visorApp->add_option("--resolution, -r"s, imgRes,
    //                     "Initial renderer to be launched. "
    //                     "Requires a scene to be set (optional)."s)
    //    ->check(CLI::ExistingFile)
    //    ->needs(rendOption);

    return visorApp;
}

CommandI& VisorCommand::Instance()
{
    static VisorCommand c = {};
    return c;
}