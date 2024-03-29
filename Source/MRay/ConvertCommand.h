#pragma once

#include "CommandI.h"
#include "Core/Error.h"

class ConvertCommand : public CommandI
{
    private:
    bool            removeDefs      = false;
    bool            useGFG          = true;
    bool            compaqMesh      = true;
    std::string     inFileName      = "";
    std::string     outFileName     = "";

    private:            ConvertCommand() = default;
    public:
    static CommandI*    Instance();

    MRayError           Invoke() override;
    CLI::App*           GenApp(CLI::App& mainApp) override;
};
