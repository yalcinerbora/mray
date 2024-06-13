
#include "Core/Types.h"
#include "Core/MRayDescriptions.h"
#include "Core/DataStructures.h"

#include <CLI/CLI.hpp>
#include <string_view>
#include "CommandI.h"
#include "ConvertCommand.h"
#include "VisorCommand.h"

const StaticVector<CommandI*, MAX_SUBCOMMAND_COUNT>&
InitializeCommands() noexcept
{
    static const StaticVector<CommandI*, MAX_SUBCOMMAND_COUNT> CommandList =
    {
        &VisorCommand::Instance(),
        &ConvertCommand::Instance()
    };
    return CommandList;
};

#include "Core/Matrix.h"

int main(int argc, const char* const argv[])
{
    StaticVector<TypedCommand, MAX_SUBCOMMAND_COUNT> appList;

    CLI::App app{std::string(MRay::Description),
                 std::string(MRay::Name)};
    app.footer(std::string(MRay::Authors) + "\n" +
               std::string(MRay::License));
    app.get_formatter()->column_width(40);
    app.option_defaults()->always_capture_default();

    app.get_formatter()->label("SUBCOMMAND", "Command");
    app.get_formatter()->label("SUBCOMMANDS", "Commands");
    app.get_formatter()->label("OPTIONS", "Options");
    app.get_formatter()->label("TEXT", "Text");
    app.get_formatter()->label("REQUIRED", "Required");

    // Version information
    using namespace std::string_literals;
    app.set_version_flag("--version, -v"s, []() -> std::string
    {
        using namespace MRay;
        return MRAY_FORMAT("{}: {}\n"
                           "-----\n"
                           "Version    : {}\n"
                           "Platform   : {}\n"
                           "C++        : {}\n"
                           "Device C++ : {}\n"
                           "-----\n"
                           "{}",
                           Name, Description,
                           VersionString,
                           PlatformName,
                           CompilerName,
                           DeviceCompilerName,
                           License);
    });
    // Create Subcommands
    const auto& commandList = InitializeCommands();
    for(CommandI* command : commandList)
        appList.emplace_back(command, command->GenApp(app));

    // Actual parsing
    try { app.parse(argc, argv); }
    catch(const CLI::ParseError& e) { return app.exit(e); }

    // Find the first parsed subcommand
    // This is fine, since we do not utilize multiple subcommands at the same time
    auto appIt = std::find_if(appList.cbegin(), appList.cend(),
                              [](const TypedCommand& sc)
    {
        return sc.second->parsed();
    });
    assert(appIt != appList.cend());

    MRayError e;
    try
    {
        e = appIt->first->Invoke();

    }
    catch(const MRayError& err)
    {
        e = err;
    }
    catch(const std::exception& err)
    {
        e = MRayError("{}", err.what());
    }

    if(e)
    {
        MRAY_ERROR_LOG("[Visor]: {}", e.GetError());
        return 1;
    }
    return 0;
}