#pragma once

#include "EntryPoint.h"

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/DefaultLogger.hpp>

class MeshFileAssimp;

class MeshViewAssimp : public MeshFileViewI
{
    friend class MeshFileAssimp;
    private:
    uint32_t                innerIndex;
    const MeshFileAssimp&   assimpFile;

    MeshViewAssimp(uint32_t innerIndex, const MeshFileAssimp&);

    public:
    AABB3           AABB() const override;
    uint32_t        MeshPrimitiveCount() const override;
    uint32_t        MeshAttributeCount() const override;
    std::string     Name() const override;
    uint32_t        InnerIndex() const override;
    // Entire Data Fetch
    bool            HasAttribute(PrimitiveAttributeLogic) const override;
    TransientData   GetAttribute(PrimitiveAttributeLogic) const override;
    MRayDataTypeRT  AttributeLayout(PrimitiveAttributeLogic) const override;
};

class MeshFileAssimp : public MeshFileI
{
    friend class MeshViewAssimp;

    private:
    std::string                     filePath;
    Assimp::Importer&               importer;
    std::unique_ptr<const aiScene>  scene;

    public:
    MeshFileAssimp(Assimp::Importer&,
                   const std::string& filePath);

    std::unique_ptr<MeshFileViewI>
    ViewMesh(uint32_t innerIndex ) override;
    //
    std::string Name() const override;
};

class MeshLoaderAssimp final : public MeshLoaderI
{
    public:
    static constexpr std::string_view   Tag = "assimp";
    static constexpr std::string_view   AssimpLogFileName = "log_assimp";

    Assimp::Importer                importer;

    public:
                                    MeshLoaderAssimp();
    std::unique_ptr<MeshFileI>      OpenFile(std::string& filePath) override;
};

// Sanity checks
static_assert(sizeof(aiVector3D) == sizeof(Vector3));
static_assert(sizeof(aiVector2D) == sizeof(Vector2));