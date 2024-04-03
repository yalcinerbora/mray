#include "RunCommand.h"
#include "GFGConverter/GFGConverter.h"

#include <CLI/CLI.hpp>
#include <string_view>

namespace MRayCLI::RunNames
{
    using namespace std::literals;
    static constexpr auto Name = "run"sv;
    static constexpr auto Description = "Directly runs a given scene file without GUI"sv;
};

MRayError RunCommand::Invoke()
{
    return MRayError::OK;
}

CLI::App* RunCommand::GenApp(CLI::App& mainApp)
{
    using namespace MRayCLI::RunNames;
    CLI::App* converter = mainApp.add_subcommand(std::string(Name),
                                                 std::string(Description));

    return converter;
}

CommandI& RunCommand::Instance()
{
    static RunCommand c = {};
    return c;
}