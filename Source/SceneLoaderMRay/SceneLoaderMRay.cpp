#include "SceneLoaderMRay.h"

#include "Core/TracerI.h"
#include "Core/Log.h"
#include "Core/Timer.h"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <memory>

MRayError SceneLoaderMRay::LoadAll()
{
    // First get "base" stuff (only the ids for lookup)
    // Second Load the surfaces on to lists (again only the ids)
    //       While loading these, get the used textures (alpha mapping)

    // (DNR = Does not refer to any other object)
    // Third Load the primitives from the surface list *DNR*
    //       Load the transforms *DNR*
    //       Load cameras *DNR*
    //       Load the materials *DNR*
    //       Load the mediums *DNR*
    //
    //       Load the lights *May rely on textures*
    //           While loading these, get the used textures
    //
    //       Load the textures
    //           Assign the textures to the material attribute channels
    //           Assign the alpha map textures to the surfaces

    // ...

    return MRayError::OK;
}

MRayError SceneLoaderMRay::OpenFile(const std::string& filePath)
{
    const auto path = std::filesystem::path(filePath);
    std::ifstream file(path);

    if(!file.is_open())
        return MRayError(MRAY_FORMAT("Scene file \"{}\" is not found",
                                     filePath));
    // Parse Json
    sceneJson = std::move(nlohmann::json::parse(file, nullptr, true, true));
    return MRayError::OK;
}

SceneLoaderMRay::SceneLoaderMRay(BS::thread_pool& pool)
    :threadPool(pool)
{}

Pair<MRayError, double> SceneLoaderMRay::LoadScene(const std::string& filePath)
                                        //,const TracerI&)
{
    Timer t; t.Start();

    MRayError e = MRayError::OK;
    try
    {
        if(e = OpenFile(filePath)) return {e, -0.0};
        if(e = LoadAll()) return {e, -0.0};
    }
    // Catch the Tracer Related Errors
    // These can not be returned since tracer can be on a different thread
    catch (MRayError const& e)
    {
        return {e, -0.0};
    }
    catch(const nlohmann::json::parse_error& e)
    {
        return {MRayError(e.what()), -0.0};
    }
    t.Split();

    //tracer->SyncAll();
    return {MRayError::OK, t.Elapsed<Second>()};

}