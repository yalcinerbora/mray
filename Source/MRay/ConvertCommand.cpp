#include "ConvertCommand.h"
#include <CLI/CLI.hpp>
#include <string_view>

namespace MRayDefs::SceneConvert
{
    using namespace std::literals;
    static constexpr auto Name = "convert"sv;
    static constexpr auto Description = "Converts various scene types to MRay "
                                        "readable form"sv;
};

MRayError ConvertCommand::Invoke()
{
    return MRayError("Not implemented!");
}

CLI::App* ConvertCommand::GenApp(CLI::App& mainApp)
{
    using namespace MRayDefs::SceneConvert;

    CLI::App* converter = mainApp.add_subcommand(std::string(Name),
                                                 std::string(Description));
    // Input
    converter->add_option("file"s, inFileName,
                          "File to be converted (type is automatically "
                          "inferred by the file extension)."s)
        ->check(CLI::ExistingFile)
        ->required();

    // Output
    converter->add_option("--output, -o"s, outFileName,
                          "Output file (If not set, input file's "
                          "path/stem will be used)."s);

    // To GFG
    CLI::Option* gfgOpt;
    gfgOpt = converter->add_flag("--gfg, -g"s, useGFG,
                                 "Convert mesh files to GFG (GFG is a simple, binary, "
                                 "load-optimized file type)."s);
    //
    converter->add_flag("--compaq, -c"s, compaqMesh,
                        "Creates a single GFG mesh file for entire scene."s)
        ->needs(gfgOpt);
    //
    converter->add_flag("--smart, -s"s, removeDefs,
                        "Only convert the used mesh files to GFG."s)
        ->needs(gfgOpt);
    return converter;
}

CommandI* ConvertCommand::Instance()
{
    static ConvertCommand c = {};
    return &c;
}