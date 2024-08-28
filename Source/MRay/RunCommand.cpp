#include "RunCommand.h"
#include "GFGConverter/GFGConverter.h"
#include "Core/Timer.h"

#include <CLI/CLI.hpp>
#include <string_view>

using DurationMS = std::chrono::milliseconds;
using LegolasLookupElem = Pair<std::string_view, DurationMS>;

// Instead of pulling std::chorno_literals to global space (it is a single
// translation unit but w/e), using constructors
static constexpr auto AnimDurationLong = DurationMS(850);
static constexpr auto AnimDurationShort = DurationMS(475);
static constexpr std::array LegolasAnimSheet =
{
    LegolasLookupElem{"< 0 >\r", AnimDurationLong},
    LegolasLookupElem{"<  0>\r", AnimDurationLong},
    LegolasLookupElem{"< 0 >\r", AnimDurationLong},
    LegolasLookupElem{"< _ >\r", AnimDurationShort},
    LegolasLookupElem{"< 0 >\r", AnimDurationLong},
    LegolasLookupElem{"<0  >\r", AnimDurationLong},
    LegolasLookupElem{"< 0 >\r", AnimDurationLong},
    LegolasLookupElem{"< _ >\r", AnimDurationShort}
};

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