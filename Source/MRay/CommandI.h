#pragma once

#include <Core/DataStructures.h>

struct MRayError;

namespace CLI { class App; }

enum class CommandType
{
    EMPTY,      // A subcommand is not provided. MRay will open either cmd gui
                // or visor gui (if config is provided), then renders the scene
                // interactively.
    RENDER,     // Render subcommand, it renders the given scene to a file
    SCENE_CONV  // Scene conversion subcommond, given a scene it will convert to another
                // format
};

class CommandI;

using TypedCommand = std::pair<CommandI*, CLI::App*>;

class CommandI
{
    public:
    virtual                 ~CommandI() = default;

    virtual MRayError       Invoke() = 0;
    virtual CLI::App*       GenApp(CLI::App& mainApp) = 0;
};

static constexpr size_t MAX_SUBCOMMAND_COUNT = 32;