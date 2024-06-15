#pragma once


#include <string>
#include <memory>

#include "Core/MRayDataType.h"
#include "Core/TracerI.h"

#include "TransientPool/TransientPool.h"


#ifdef MRAY_MESHLOADER_SHARED_EXPORT
    #define MRAY_MESHLOADER_ENTRYPOINT MRAY_DLL_EXPORT
#else
    #define MRAY_MESHLOADER_ENTRYPOINT MRAY_DLL_IMPORT
#endif

struct MRayError;

using InnerIdList = const std::vector<uint32_t>;

class MeshFileI
{
    public:

    virtual             ~MeshFileI() = default;

    virtual AABB3       AABB() const = 0;
    virtual uint32_t    MeshPrimitiveCount() const = 0;
    virtual uint32_t    MeshAttributeCount() const = 0;
    virtual std::string Name() const = 0;

    // Entire Data Fetch
    virtual bool            HasAttribute(PrimitiveAttributeLogic) const = 0;
    virtual TransientData   GetAttribute(PrimitiveAttributeLogic) const = 0;
    virtual MRayDataTypeRT  AttributeLayout(PrimitiveAttributeLogic) const = 0;
};

// This is per thread (that is why there is another abstraction (MeshFile))
class MeshLoaderI
{
    public:
    virtual                             ~MeshLoaderI() = default;

    virtual std::unique_ptr<MeshFileI>  OpenFile(std::string& filePath,
                                                 uint32_t innerIndex = 0) = 0;
};

// This is loaded once for the process
class MeshLoaderPoolI
{
    public:
    virtual                                 ~MeshLoaderPoolI() = default;
    virtual std::unique_ptr<MeshLoaderI>    AcquireALoader(const std::string& tag) const = 0;
};

// C Interface (Used when dynamically loading the DLL)
namespace MeshLoaderDetail
{
    extern "C" MRAY_MESHLOADER_ENTRYPOINT
    MeshLoaderPoolI* ConstructMeshLoaderPool();

    extern "C" MRAY_MESHLOADER_ENTRYPOINT
    void DestroyMeshLoaderPool(MeshLoaderPoolI*);

}

// C++ Interface
inline
std::unique_ptr<MeshLoaderPoolI, decltype(&MeshLoaderDetail::DestroyMeshLoaderPool)>
CreateMeshLoaderPool()
{
    using namespace MeshLoaderDetail;
    using Ptr = std::unique_ptr<MeshLoaderPoolI, decltype(&DestroyMeshLoaderPool)>;
    return Ptr(ConstructMeshLoaderPool(), &DestroyMeshLoaderPool);
}