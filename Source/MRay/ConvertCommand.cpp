#include "ConvertCommand.h"

#include "GFGConverter/GFGConverter.h"
#include "Core/Log.h"

#include <CLI/CLI.hpp>
#include <string_view>
#include <thread>

namespace MRayCLI::ConvertNames
{
    using namespace std::literals;
    static constexpr auto Name = "convert"sv;
    static constexpr auto Description = "Converts various scene types to MRay "
                                        "readable form"sv;
};

ConvertCommand::ConvertCommand()
    : threadCount(std::max(1u, std::thread::hardware_concurrency()))
{}

MRayError ConvertCommand::Invoke()
{

    if(useGFG)
    {
        using namespace MRayConvert;
        using enum ConvFlagEnum;
        ConversionFlags flags;
        if(compaqMesh) flags |= PACK_GFG;
        if(convQuats) flags |= NORMAL_AS_QUATERNION;
        if(!overwrite) flags |= FAIL_ON_OVERWRITE;

        namespace fs = std::filesystem;
        using namespace std::string_literals;

        if(outFileName.empty())
        {
            auto inPath = fs::path(inFileName);
            auto newName = inPath.stem().string() + "_gfg"s + inPath.extension().string();
            inPath.replace_filename(fs::path(newName));
            outFileName = inPath.string();
        }

        Expected<double> result = MRayConvert::ConvertMeshesToGFG(outFileName,
                                                                  inFileName,
                                                                  threadCount,
                                                                  flags);

        if(result.has_error())
            return result.error();

        MRAY_LOG("Conversion took {}s.", result.value());
        return MRayError::OK;
    }
    else
    {
        return MRayError("Not yet implemented!");
    }
}

CLI::App* ConvertCommand::GenApp(CLI::App& mainApp)
{
    using namespace MRayCLI::ConvertNames;

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

    converter->add_flag("--overwrite, -w"s, overwrite,
                        "Overwrite output files."s);
    converter->add_option("--threads, -t"s, threadCount,
                          "Thread pool's thread count."s)
        ->expected(1);

    // To GFG Options (TODO: remove "required" when other conversions are implemented)
    CLI::Option* gfgOpt;
    gfgOpt = converter->add_flag("--gfg, -g"s, useGFG,
                                 "Convert mesh files to GFG (GFG is a simple, binary, "
                                 "load-optimized file type)."s)
        ->required();
    //
    converter->add_flag("--pack, -p"s, compaqMesh,
                        "Creates a single GFG mesh file for entire scene."s)
        ->needs(gfgOpt);
    //
    converter->add_flag("--quatnorm, -q"s, convQuats,
                        "Convert normals to tangent space quaternions."s)
        ->needs(gfgOpt);
    return converter;
}

CommandI& ConvertCommand::Instance()
{
    static ConvertCommand c = {};
    return c;
}