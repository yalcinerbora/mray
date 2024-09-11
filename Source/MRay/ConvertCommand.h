#pragma once

#include "CommandI.h"

#include <string>

class ConvertCommand : public CommandI
{
    private:
    bool            convQuats       = true;
    bool            useGFG          = true;
    bool            compaqMesh      = true;
    bool            overwrite       = false;
    std::string     inFileName      = "";
    std::string     outFileName     = "";
    uint32_t        threadCount;

    private:            ConvertCommand();
    public:
    static CommandI&    Instance();

    MRayError           Invoke() override;
    CLI::App*           GenApp(CLI::App& mainApp) override;
};
