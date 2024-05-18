#pragma once

#include "TracerBase.h"

class Tracer final : public TracerBase
{
    private:
    static TypeGeneratorPack LibGlobalTypeGenerators;

    public:
    Tracer(BS::thread_pool& tp);
};


// put extern template here
// actually generate there