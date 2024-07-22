#pragma once

#include "CommandI.h"

class VisorCommand : public CommandI
{
    using OptionalRes = Optional<std::array<uint32_t, 2>>;
    private:
    std::string             tracerConfigFile    = "";
    std::string             visorConfigFile     = "";
    Optional<std::string>   sceneFile           = std::nullopt;
    Optional<std::string>   renderConfigFile    = std::nullopt;
    OptionalRes             imgRes              = std::array<uint32_t, 2>{0, 0};
    uint32_t                threadCount;

    private:            VisorCommand();
    public:
    static CommandI&    Instance();

    MRayError           Invoke() override;
    CLI::App*           GenApp(CLI::App& mainApp) override;
};
