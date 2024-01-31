#pragma once

#include "Core/SceneLoaderI.h"
#include "BS/BS_thread_pool.hpp"

#include <nlohmann/json.hpp>

class TracerI;

class SceneLoaderMRay : public SceneLoaderI
{
    private:
    nlohmann::json      sceneJson;
    BS::thread_pool&    threadPool;

    MRayError           LoadAll();
    MRayError           OpenFile(const std::string& filePath);

    public:
    SceneLoaderMRay(BS::thread_pool& pool);


    Pair<MRayError, double> LoadScene(const std::string& filePath
                        //,const TracerI& tracer
    )  override;
};