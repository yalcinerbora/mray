#include "VisorCommand.h"

#include <CLI/CLI.hpp>
#include <string_view>

#include "Visor/EntryPoint.h"

#include "Core/SharedLibrary.h"
#include "Core/System.h"

namespace MRayCLI::VisorNames
{
    using namespace std::literals;
    static constexpr auto Name = "visor"sv;
    static constexpr auto Description = "enables the visor system for interactive "
                                        "rendering (default)"sv;
};

MRayError VisorCommand::Invoke()
{
    using namespace std::literals;
    static const std::string VisorDLLName = "Visor"s;
    static const SharedLibArgs VisorDLLArgs =
    {
        .mangledConstructorName = "ConstructVisor"s,
        .mangledDestructorName  = "DestroyVisor"s
    };

    SharedLibrary lib(VisorDLLName);
    SharedLibPtr<VisorI> visorSystem = {nullptr, nullptr};
    MRayError e = lib.GenerateObjectWithArgs<Tuple<>>(visorSystem,
                                                      VisorDLLArgs);
    if(e) return e;

    // TODO: Acq visor config from file
    VisorConfig vConf = {};
    vConf.enforceIGPU = false;
    vConf.displayHDR = true;
    vConf.wSize = Vector2i(1280, 720);

    // Actual initialization, process path should be called from here
    // In dll it may return .dll's path (on windows, I think)
    std::string processPath = GetProcessPath();
    e = visorSystem->MTInitialize(vConf, processPath);
    if(e) return e;

    // ====================== //
    //     Real-time Loop     //
    // ====================== //
    while(!visorSystem->MTIsTerminated())
    {
        visorSystem->MTWaitForInputs();
        visorSystem->MTRender();
    }

    // GG!
    visorSystem->MTDestroy();
    return MRayError::OK;
}

CLI::App* VisorCommand::GenApp(CLI::App& mainApp)
{
    using namespace MRayCLI::VisorNames;
    CLI::App* visorApp = mainApp.add_subcommand("", std::string(Description))
        ->alias(std::string(Name));
    return visorApp;

    //// Input
    //converter->add_option("file"s, inFileName,
    //                      "File to be converted (type is automatically "
    //                      "inferred by the file extension)."s)
    //    ->check(CLI::ExistingFile)
    //    ->required();

    //// Output
    //converter->add_option("--output, -o"s, outFileName,
    //                      "Output file (If not set, input file's "
    //                      "path/stem will be used)."s);

    //converter->add_flag("--overwrite, -w"s, overwrite,
    //                    "Overwrite output files."s);

    //// To GFG Options (TODO: remove "required" when other conversions are implemented)
    //CLI::Option* gfgOpt;
    //gfgOpt = converter->add_flag("--gfg, -g"s, useGFG,
    //                             "Convert mesh files to GFG (GFG is a simple, binary, "
    //                             "load-optimized file type)."s)
    //    ->required();
    ////
    //converter->add_flag("--pack, -p"s, compaqMesh,
    //                    "Creates a single GFG mesh file for entire scene."s)
    //    ->needs(gfgOpt);
    ////
    //converter->add_flag("--quatnorm, -q"s, convQuats,
    //                    "Convert normals to tangent space quaternions."s)
    //    ->needs(gfgOpt);
    //return converter;
}

CommandI& VisorCommand::Instance()
{
    static VisorCommand c = {};
    return c;
}