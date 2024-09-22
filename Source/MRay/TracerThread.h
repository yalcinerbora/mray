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

    //
    std::map<std::string_view, SharedLibrary>   sceneLoaderDLLs;
    std::map<std::string_view, SceneLoaderPtr>  sceneLoaders;
    SceneLoaderI*                               currentScene = nullptr;

    // Learned something new (Check the other compilers though only checked MSVC)
    // You can use std::numeric_limits on user defined integral types, nice.
    Vector2ui   resolution;
    Vector2ui   regionMin;
    Vector2ui   regionMax;
    // Current State
    std::string         currentRendererName;
    uint32_t            currentRenderLogic0     = 0;
    uint32_t            currentRenderLogic1     = 0;
    RendererId          currentRenderer         = TracerIdInvalid<RendererId>;
    size_t              currentCamIndex         = 0;
    CameraTransform     currentCamTransform;
    AABB3               currentSceneAABB        = AABB3::Zero();
    TracerIdPack        sceneIds;
    SemaphoreInfo       currentSem = {};
    // Flow states
    // TODO: I'm pretty sure this will get complicated really fast
    // maybe change this to a state machine later
    // Tracer is terminated due to a fatal error
    bool isTerminated   = false;
    // Should we do polling or blocking fetch from the queue
    // During rendering, system goes to poll mode to render as fast as possible
    bool isInSleepMode  = true;
    // Are we currently rendering
    bool isRendering    = false;
    // Are we paused
    bool isPaused       = false;

    //
    void        LoopWork() override;
    void        InitialWork() override;
    void        FinalWork() override;

    MRayError   CreateRendererFromConfig(const std::string& configJsonPath);
    void        RestartRenderer();

    void        HandleRendering();
    void        HandleStartStop(bool newStartStopSignal);
    void        HandlePause();
    void        HandleSceneChange(const std::string&);
    void        HandleRendererChange(const std::string&);

    public:
    // Constructors & Destructor
                TracerThread(TransferQueue& queue,
                             BS::thread_pool& tp);
                ~TracerThread() = default;

    MRayError   MTInitialize(const std::string& tracerConfig);
    bool        InternallyTerminated() const override;

    void        SetInitialResolution(const Vector2ui& resolution,
                                     const Vector2ui& regionMin,
                                     const Vector2ui& regionMax);

    // Misc.
    GPUThreadInitFunction GetThreadInitFunction() const;
};
