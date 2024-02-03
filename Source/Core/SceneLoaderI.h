#pragma once

#include <string>
#include "Error.h"
#include "Types.h"

class TracerI;

class SceneLoaderI
{
    public:
    virtual ~SceneLoaderI() = default;

    virtual Pair<MRayError, double> LoadScene(TracerI& tracer,
                                              const std::string& filePath) = 0;
};