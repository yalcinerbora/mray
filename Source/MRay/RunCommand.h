#pragma once

#include "CommandI.h"

#include "Core/MemAlloc.h"
#include "Core/Timer.h"

class RunCommand : public CommandI
{
    using Resolution = std::array<uint32_t, 2>;
    private:
    std::string visorConfString;
    // Input Arg related
    std::string tracerConfigFile;
    std::string sceneFile;
    std::string renderConfigFile;
    Resolution  imgRes = std::array<uint32_t, 2>{0, 0};
    uint32_t    threadCount;

    // Runtime related
    MemAlloc::VectorBackedMemory imageMem;
    Timer                        timer;
    //
    bool                EventLoop();
    // Constructors
                        RunCommand() = default;
    public:
    static CommandI&    Instance();

    MRayError           Invoke() override;
    CLI::App*           GenApp(CLI::App& mainApp) override;
};
