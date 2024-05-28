#pragma once

#include "Core/TracerI.h"
#include "Core/RealtimeThread.h"
#include "Core/SharedLibrary.h"
#include "Core/SceneLoaderI.h"

#include "Common/TransferQueue.h"

namespace BS { class thread_pool; }

class TracerThread final : public RealtimeThread
{
    using SceneLoaderPtr = SharedLibPtr<SceneLoaderI>;
    private:
    // Tracer Related
    std::unique_ptr<SharedLibrary>  dllFile;
    SharedLibPtr<TracerI>           tracer;
    TransferQueue::TracerView       transferQueue;
    BS::thread_pool&                threadPool;

    std::map<std::string_view, SharedLibrary>   sceneLoaderDLLs;
    std::map<std::string_view, SceneLoaderPtr>  sceneLoaders;
    SceneLoaderI*                               currentScene = nullptr;

    bool        isTerminated    = false;
    // Should we do polling or blocking fetch from the queue
    // During rendering, system goes to poll mode to render as fast as possible
    bool        isInSleepMode   = true;

    // Some states
    // TODO: I'm pretty sure this will get complicated really fast
    // maybe change this to a state machine later
    bool isRendering = false;

    //
    void        LoopWork() override;
    void        InitialWork() override;
    void        FinalWork() override;

    public:
    // Constructors & Destructor
                TracerThread(TransferQueue& queue,
                             BS::thread_pool& tp);
                ~TracerThread() = default;

    MRayError   MTInitialize(const std::string& tracerConfig);
    bool        InternallyTerminated() const override;
    // Misc.
    GPUThreadInitFunction GetThreadInitFunction() const;
};
