
#include "Core/Types.h"
#include "Core/MRayDescriptions.h"
#include "Core/DataStructures.h"

#include <CLI/CLI.hpp>
#include <string_view>
#include "CommandI.h"
#include "ConvertCommand.h"

const StaticVector<CommandI*, MAX_SUBCOMMAND_COUNT>&
InitializeCommands() noexcept
{
    static const StaticVector<CommandI*, MAX_SUBCOMMAND_COUNT> CommandList =
    {
        &ConvertCommand::Instance()
    };
    return CommandList;
};

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
    //CLI11_PARSE(app, argc, argv);
    try
    {
        (app).parse((argc), (argv));                                                                                   \
    }
    catch(const CLI::ParseError& e)
    {
        return (app).exit(e);                                                                                          \
    }

    // Find the first parsed subcommand
    // This is fine, since we do not utilize multiple subcommands at the same time
    auto appIt = std::find_if(appList.cbegin(), appList.cend(),
                              [](const TypedCommand& sc)
    {
        return sc.second->parsed();
    });



    if(appIt == appList.cend())
    {
        // openup GUI etc.
        MRAY_ERROR_LOG("GUI mode is not yet implemented!");
        return 1;
    }

    MRayError e = appIt->first->Invoke();
    if(e)
    {
        MRAY_ERROR_LOG("{}", e.GetError());
        return 1;
    }
    return 0;
}