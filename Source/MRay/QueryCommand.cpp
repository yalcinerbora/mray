#include "QueryCommand.h"

#include "Core/Log.h"
#include "Core/Error.h"
#include "Core/ThreadPool.h"

#include "TracerThread.h"

#include <CLI/CLI.hpp>
#include <string_view>
#include <thread>

namespace MRayCLI::QueryNames
{
    using namespace std::literals;
    static constexpr auto Name = "query"sv;
    static constexpr auto Description = "Queries the supported types of "
                                        "the given Tracer shared library"sv;
};

QueryCommand::QueryCommand()
{}

MRayError QueryCommand::Invoke()
{
    // Thread pool, many things will need this
    ThreadPool threadPool;
    TransferQueue transferQueue(1, 1, [](){});

    // Get the tracer dll
    TracerThread tracerThread(transferQueue, threadPool);
    MRayError e = tracerThread.MTInitialize(tracerConfigFile);
    if(e) return e;

    if(queryTypeName.empty())
        tracerThread.DisplayTypes();
    else
        tracerThread.DisplayTypeAttributes(queryTypeName);

    return MRayError::OK;
}

CLI::App* QueryCommand::GenApp(CLI::App& mainApp)
{
    using namespace MRayCLI::QueryNames;

    CLI::App* queryApp = mainApp.add_subcommand(std::string(Name),
                                                std::string(Description));
    // Input
    queryApp->add_option("--tracerConf, --tConf"s, tracerConfigFile,
                          "Tracer configuration file, mainly specifies the "
                          "tracer dll name to be loaded."s)
        ->check(CLI::ExistingFile)
        ->required()
        ->expected(1);

    queryApp->add_option("--attributes, -a"s, queryTypeName,
                         "display attributes of the given type"s);

    return queryApp;
}

CommandI& QueryCommand::Instance()
{
    static QueryCommand c = {};
    return c;
}