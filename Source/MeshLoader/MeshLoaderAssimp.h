#pragma once

#include "EntryPoint.h"

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/DefaultLogger.hpp>

class MeshFileAssimp : public MeshFileI
{
    private:
    std::string                     filePath;
    Assimp::Importer&               importer;
    std::unique_ptr<const aiScene>  scene;
    uint32_t                        innerIndex;

    public:
                    MeshFileAssimp(Assimp::Importer&,
                                   const std::string& filePath,
                                   uint32_t innerIndex = 0);

    AABB3           AABB() const override;
    uint32_t        MeshPrimitiveCount() const override;
    uint32_t        MeshAttributeCount() const override;
    std::string     Name() const override;
    // Entire Data Fetch
    bool            HasAttribute(PrimitiveAttributeLogic) const override;
    TransientData   GetAttribute(PrimitiveAttributeLogic) const override;
    MRayDataTypeRT  AttributeLayout(PrimitiveAttributeLogic) const override;
};

class MeshLoaderAssimp final : public MeshLoaderI
{
    public:
    static constexpr std::string_view   Tag = "assimp";
    static constexpr std::string_view   AssimpLogFileName = "log_assimp";

    Assimp::Importer                importer;

    public:
                                    MeshLoaderAssimp();
    std::unique_ptr<MeshFileI>      OpenFile(std::string& filePath,
                                             uint32_t innerIndex = 0) override;
};

// Sanity checks
static_assert(sizeof(aiVector3D) == sizeof(Vector3));
static_assert(sizeof(aiVector2D) == sizeof(Vector2));