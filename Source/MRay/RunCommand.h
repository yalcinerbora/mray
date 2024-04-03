#pragma once

#include "CommandI.h"
#include "Core/Error.h"

class RunCommand : public CommandI
{
    private:
    private:            RunCommand() = default;
    public:
    static CommandI&    Instance();

    MRayError           Invoke() override;
    CLI::App*           GenApp(CLI::App& mainApp) override;
};
