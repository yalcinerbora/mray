#pragma once

#include "CommandI.h"
#include "Core/Error.h"

class VisorCommand : public CommandI
{
    private:
    std::string         tracerConfig    = "";
    std::string         visorConfig     = "";
    std::string         sceneFile       = "";

    private:            VisorCommand() = default;
    public:
    static CommandI&    Instance();

    MRayError           Invoke() override;
    CLI::App*           GenApp(CLI::App& mainApp) override;
};
