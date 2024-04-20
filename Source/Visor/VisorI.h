#pragma once

#include "Core/Vector.h"

using namespace std::string_literals;

class TransferQueue;
namespace BS { class thread_pool; }

struct VisorConfig
{
    // DLL Related
    std::string         dllName;
    std::string         dllCreateFuncName = "ConstructVisor"s;
    std::string         dllDeleteFuncName = "DestroyVisor"s;
    // Window Related
    bool                enforceIGPU = true;
    bool                displayHDR  = true;
    bool                realTime    = false;
    Vector2i            wSize       = Vector2i(1280, 720);
    // Technical
    uint32_t            commandBufferSize   = 8;
    uint32_t            responseBufferSize  = 8;
};

class VisorI
{
    public:
    virtual             ~VisorI() = default;
    //
    virtual MRayError   MTInitialize(TransferQueue& transferQueue,
                                     BS::thread_pool*,
                                     const VisorConfig&,
                                     const std::string& processPath) = 0;
    virtual bool        MTIsTerminated() = 0;
    virtual void        MTWaitForInputs() = 0;
    virtual void        MTRender()  = 0;
    virtual void        MTDestroy() = 0;
    virtual void        TriggerEvent() = 0;
};