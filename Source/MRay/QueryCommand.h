#pragma once

#include "CommandI.h"

#include <string>

class QueryCommand : public CommandI
{
    private:
    std::string         tracerConfigFile;
    std::string         queryTypeName = "";

    private:            QueryCommand();
    public:
    static CommandI&    Instance();

    MRayError           Invoke() override;
    CLI::App*           GenApp(CLI::App& mainApp) override;
};
