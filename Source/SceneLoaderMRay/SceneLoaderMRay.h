#pragma once

#include "Core/SceneLoaderI.h"

class TracerI;

class SceneLoaderMRay : public SceneLoaderI
{
    public:
    MRayError LoadScene(const std::string& filePath,
                        const TracerI& tracer) override;
};