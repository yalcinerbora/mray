#pragma once

#include <string>
#include "Error.h"
#include "Types.h"

class TracerI;

class SceneLoaderI
{
    public:
    virtual ~SceneLoaderI() = default;

    virtual Pair<MRayError, double> LoadScene(const std::string& filePath) = 0;
        //,const TracerI& tracer) = 0;
};