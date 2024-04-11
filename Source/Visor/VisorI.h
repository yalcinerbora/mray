#pragma once

#include "Core/Vector.h"

struct VisorConfig
{
    // Window Related
    bool                enforceIGPU;
    bool                displayHDR;
    bool                realTime;

    bool                vSyncOn;
    Vector2i            wSize;
};


class VisorI
{
    public:
    virtual             ~VisorI() = default;
    //
    virtual MRayError   MTInitialize(VisorConfig,
                                     const std::string& processPath) = 0;
    virtual bool        MTIsTerminated() = 0;
    virtual void        MTWaitForInputs() = 0;
    virtual void        MTRender()  = 0;
    virtual void        MTDestroy() = 0;
};