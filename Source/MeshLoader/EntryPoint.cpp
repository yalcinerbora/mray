#include "EntryPoint.h"
#include "MeshLoaderAssimp.h"
#include "MeshLoaderGFG.h"

#include <map>
#include <memory>

template<class BaseType, class...Args>
using GeneratorFuncType = std::unique_ptr<BaseType> (*)(Args&&... args);

template<class BaseType, class Type, class...Args>
std::unique_ptr<BaseType> GenerateType(Args&&... args)
{
    return std::make_unique<Type>(std::forward<Args>(args)...);
}

using MeshLoaderPtr = std::unique_ptr<MeshLoaderI>;

class MeshLoaderPool : public MeshLoaderPoolI
{
    using GeneratorFunc = GeneratorFuncType<MeshLoaderI>;

    private:
    std::map<std::string, GeneratorFunc>  generators;

    public:
                            MeshLoaderPool();
    virtual                 ~MeshLoaderPool() = default;
    virtual MeshLoaderPtr   AcquireALoader(const std::string& extension) const override;
};

MeshLoaderPool::MeshLoaderPool()
{
    generators.emplace(MeshLoaderAssimp::Tag,
                       &GenerateType<MeshLoaderI, MeshLoaderAssimp>);
    generators.emplace(MeshLoaderGFG::Tag,
                       &GenerateType<MeshLoaderI, MeshLoaderGFG>);
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