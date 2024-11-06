#pragma once

#include "CommandI.h"

#include "Core/MemAlloc.h"
#include "Core/Timer.h"
#include "Core/TimelineSemaphore.h"

#include "Common/AnalyticStructs.h"
#include "Common/RenderImageStructs.h"

#include <BS/BS_thread_pool.hpp>

class TransferQueue;

class RunCommand : public CommandI
{
    using Resolution = std::array<uint32_t, 2>;
    using ThroughputAverage = Math::MovingAverage<16>;
    using TimeAverage = Math::MovingAverage<8>;

    private:
    std::string visorConfString;
    // Input Arg related
    std::string tracerConfigFile;
    std::string sceneFile;
    std::string renderConfigFile;
    Resolution  imgRes = std::array<uint32_t, 2>{0, 0};
    uint32_t    threadCount;

    // Runtime related
    MemAlloc::AlignedMemory imageMem;
    Span<double>            imageRData;
    Span<double>            imageGData;
    Span<double>            imageBData;
    Span<double>            imageSData;
    //
    Timer                   renderTimer;
    Timer                   cmdTimer;
    ThroughputAverage       renderThroughputAverage;
    TimeAverage             iterationTimeAverage;
    SceneAnalyticData       sceneInfo;
    TracerAnalyticData      tracerInfo;
    RendererAnalyticData    rendererInfo;
    RenderBufferInfo        renderBufferInfo;
    uint64_t                memUsage;
    BS::multi_future<void>  accumulateFuture;
    bool                    startDisplayProgressBar = false;
    // The "timeline semaphore" (CPU emulation)
    // This will be used to synchronize between MRay and Run
    TimelineSemaphore       syncSemaphore = TimelineSemaphore(0);
    //
    bool                EventLoop(TransferQueue&, BS::thread_pool&);
    // Constructors
                        RunCommand();
    public:
    static CommandI&    Instance();

    MRayError           Invoke() override;
    CLI::App*           GenApp(CLI::App& mainApp) override;
};
