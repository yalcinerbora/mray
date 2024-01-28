#include "SceneLoaderMRay.h"
#include "Core/TracerI.h"

MRayError SceneLoaderMRay::LoadScene(const std::string&,
                                     const TracerI&)
{
    return MRayError::OK;
}
