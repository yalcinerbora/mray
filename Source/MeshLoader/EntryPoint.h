#pragma once


#include <string>
#include <memory>

#include "Core/Error.h"
#include "Core/MRayDataType.h"
#include "Core/TracerI.h"

#include "MRayInput/MRayInput.h"


#ifdef MRAY_MESHLOADER_SHARED_EXPORT
    #define MRAY_MESHLOADER_ENTRYPOINT MRAY_DLL_EXPORT
#else
    #define MRAY_MESHLOADER_ENTRYPOINT MRAY_DLL_IMPORT
#endif

using InnerIdList = const std::vector<uint32_t>;

class MeshFileI
{
    public:

    virtual             ~MeshFileI() = default;

    virtual AABB3       AABB(uint32_t innerId = 0) const = 0;
    virtual uint32_t    MeshPrimitiveCount(uint32_t innerId = 0) const = 0;
    virtual uint32_t    MeshAttributeCount(uint32_t innerId = 0) const = 0;
    virtual std::string Name() const = 0;

    // Entire Data Fetch
    virtual bool            HasAttribute(PrimitiveAttributeLogic, uint32_t innerId = 0) const = 0;
    virtual MRayInput       GetAttribute(PrimitiveAttributeLogic, uint32_t innerId = 0) const = 0;
    virtual MRayDataTypeRT  AttributeLayout(PrimitiveAttributeLogic, uint32_t innerId = 0) const = 0;
};

// This is per thread (that is why there is another abstraction (MeshFile))
class MeshLoaderI
{
    public:
    virtual                             ~MeshLoaderI() = default;

    virtual std::unique_ptr<MeshFileI>  OpenFile(std::string& filePath) = 0;
};

// This is loaded once for the process
class MeshLoaderPoolI
{
    public:
    virtual std::unique_ptr<MeshLoaderI>  AcquireALoader(const std::string& extension) const = 0;
};

MRAY_MESHLOADER_ENTRYPOINT std::unique_ptr<const MeshLoaderPoolI> GetMeshLoader();

using MeshLoaderPoolPtr = std::unique_ptr<const MeshLoaderPoolI>;