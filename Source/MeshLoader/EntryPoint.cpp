#include "EntryPoint.h"
#include "MeshLoaderAssimp.h"
#include "MeshLoaderGFG.h"

#include "Core/TypeGenFunction.h"

#include <map>
#include <memory>

#include <assimp/DefaultLogger.hpp>

using MeshLoaderPtr = std::unique_ptr<MeshLoaderI>;

class MeshLoaderPool : public MeshLoaderPoolI
{
    using GeneratorFunc = GeneratorFuncType<MeshLoaderI>;

    private:
    std::map<std::string, GeneratorFunc>  generators;
    public:
                            MeshLoaderPool();
    virtual                 ~MeshLoaderPool();
    virtual MeshLoaderPtr   AcquireALoader(const std::string& extension) const override;
};

MeshLoaderPool::MeshLoaderPool()
{
    Assimp::DefaultLogger::set(new MRayAssimpLogger());
    Assimp::DefaultLogger::get()->setLogSeverity(Assimp::Logger::NORMAL);

    generators.emplace(MeshLoaderAssimp::Tag,
                       &GenerateType<MeshLoaderI, MeshLoaderAssimp>);
    generators.emplace(MeshLoaderGFG::Tag,
                       &GenerateType<MeshLoaderI, MeshLoaderGFG>);
}

MeshLoaderPool::~MeshLoaderPool()
{
    Assimp::DefaultLogger::kill();
}

MeshLoaderPtr MeshLoaderPool::AcquireALoader(const std::string& tag) const
{
    auto it = generators.find(tag);
    if(it == generators.cend())
    {
        throw MRayError("Unable to find a mesh loader "
                        "with tag {}", tag);
    }
    return it->second();
}

extern "C" MRAY_MESHLOADER_ENTRYPOINT
MeshLoaderPoolI* MeshLoaderDetail::ConstructMeshLoaderPool()
{
    return new MeshLoaderPool();
}

extern "C" MRAY_MESHLOADER_ENTRYPOINT
void MeshLoaderDetail::DestroyMeshLoaderPool(MeshLoaderPoolI* ptr)
{
    delete ptr;
}