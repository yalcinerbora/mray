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
    return std::make_unique<BaseType>(new Type(std::forward<Args>(args...)));
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
    // Load a load b load c
}

MeshLoaderPtr MeshLoaderPool::AcquireALoader(const std::string& extension) const
{
    auto it = generators.find(extension);
    if(it == generators.cend())
    {
        throw MRayError("Unable to find a mesh loader "
                        "for extension *.{}", extension);
    }
    return it->second();
}

MRAY_MESHLOADER_ENTRYPOINT std::unique_ptr<const MeshLoaderPoolI> GetMeshLoader()
{
    return std::unique_ptr<const MeshLoaderPoolI>(new MeshLoaderPool());
}