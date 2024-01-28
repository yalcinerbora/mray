#pragma once

#include <string>
#include "Error.h"

class TracerI;

class SceneLoaderI
{
    public:
    virtual ~SceneLoaderI() = default;

    virtual MRayError LoadScene(const std::string& filePath,
                                const TracerI& tracer) = 0;
};