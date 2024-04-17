#pragma once

#include "CommandI.h"
#include "Core/Error.h"

class VisorCommand : public CommandI
{
    private:
    std::string         tracerConfigFile    = "";
    std::string         visorConfigFile     = "";
    std::string         sceneFile           = "";
    std::string         renderConfigFile    = "";
    Vector2i            imgRes              = Vector2i::Zero();

    private:            VisorCommand() = default;
    public:
    static CommandI&    Instance();

    MRayError           Invoke() override;
    CLI::App*           GenApp(CLI::App& mainApp) override;
};
