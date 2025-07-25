#include "Core/MRayDescriptions.h"
#include "Core/DataStructures.h"
#include "Core/Error.h"
#include "Core/System.h"
#include "Core/Timer.h"
#include "Core/Profiling.h"

#include <CLI/CLI.hpp>
#include "CommandI.h"
#include "ConvertCommand.h"
#include "VisorCommand.h"
#include "QueryCommand.h"
#include "RunCommand.h"

class ProcessTimer
{
    private:
    Timer       t;
    bool        display = false;

    public:
    // Constructors & Destructor
                    ProcessTimer();
                    ProcessTimer(const ProcessTimer&) = delete;
                    ProcessTimer(ProcessTimer&&) = delete;
    ProcessTimer&   operator=(const ProcessTimer&) = delete;
    ProcessTimer&   operator=(ProcessTimer&&) = delete;
                    ~ProcessTimer();

    void            Display();
};

ProcessTimer::ProcessTimer()
{
    t.Start();
}

ProcessTimer::~ProcessTimer()
{
    if(!display) return;

    t.Split();
    std::string time = FormatTimeDynamic(t);
    MRAY_LOG("\"int main(...)\" time: {:s}", time);
}

void ProcessTimer::Display()
{
    display = true;
}

const StaticVector<CommandI*, MAX_SUBCOMMAND_COUNT>&
InitializeCommands() noexcept
{
    static StaticVector<CommandI*, MAX_SUBCOMMAND_COUNT> CommandList =
    {
        &ConvertCommand::Instance(),
        &RunCommand::Instance(),
        &QueryCommand::Instance()
    };

    if constexpr(MRay::IsVisorBuilt)
        CommandList.push_back(&VisorCommand::Instance());

    return CommandList;
};

int main(int argc, const char* const argv[])
{
    ProcessTimer processTimer;

    if(!EnableVTMode())
    {
        MRAY_WARNING_LOG("Unable to set virtual terminal processing. "
                         "Log outputs may have invalid characters!");
    }

    StaticVector<TypedCommand, MAX_SUBCOMMAND_COUNT> appList;

    CLI::App app{std::string(MRay::Description),
                 std::string(MRay::Name)};
    app.footer(std::string(MRay::Authors) + "\n" +
               std::string(MRay::License));
    app.option_defaults()->always_capture_default();
    app.get_formatter()->label("SUBCOMMAND", "Command");
    app.get_formatter()->label("SUBCOMMANDS", "Commands");
    app.get_formatter()->label("OPTIONS", "Options");
    app.get_formatter()->label("TEXT", "Text");
    app.get_formatter()->label("REQUIRED", "Required");
    app.set_help_all_flag("--help-all", "All args of sub-commands");
    app.require_subcommand();
    // Profiler (can be set via main app)
    Optional<bool> enableProfiling;
    app.add_flag("--profile, -p", enableProfiling, "Enable Tracy Profiling");

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

    ProfilerDLL profilerDLL(enableProfiling);

    processTimer.Display();
    MRayError e = appIt->first->Invoke();
    if(e)
    {
        MRAY_ERROR_LOG("[Main]: {}", e.GetError());
        return 1;
    }
    return 0;
}