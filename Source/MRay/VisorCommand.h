#pragma once

#include "CommandI.h"

#include <string>

class VisorCommand : public CommandI
{
    using OptionalRes = std::optional<std::array<uint32_t, 2>>;
    using OptionalStr = std::optional<std::string>;
    private:
    std::string   tracerConfigFile    = "";
    std::string   visorConfigFile     = "";
    OptionalStr   sceneFile           = std::nullopt;
    OptionalStr   renderConfigFile    = std::nullopt;
    OptionalRes   imgRes              = std::array<uint32_t, 2>{0, 0};
    uint32_t      threadCount;

                        VisorCommand();
    public:
    static CommandI&    Instance();

    MRayError           Invoke() override;
    CLI::App*           GenApp(CLI::App& mainApp) override;
};
