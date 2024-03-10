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

    public:
                    MeshFileAssimp(Assimp::Importer&,
                                   const std::string& filePath);

    AABB3           AABB(uint32_t innerId = 0) const override;
    uint32_t        MeshPrimitiveCount(uint32_t innerId = 0) const override;
    uint32_t        MeshAttributeCount(uint32_t innerId = 0) const override;
    std::string     Name() const override;
    // Entire Data Fetch
    bool            HasAttribute(PrimitiveAttributeLogic, uint32_t innerId = 0) const override;
    TransientData   GetAttribute(PrimitiveAttributeLogic, uint32_t innerId = 0) const override;
    MRayDataTypeRT  AttributeLayout(PrimitiveAttributeLogic, uint32_t innerId = 0) const override;
};

class MeshLoaderAssimp final : public MeshLoaderI
{
    public:
    static constexpr std::string_view   Tag = "assimp";
    private:
    static constexpr std::string_view   AssimpLogFileName = "log_assimp";

    Assimp::Importer                importer;

    public:
                                    MeshLoaderAssimp();
    std::unique_ptr<MeshFileI>      OpenFile(std::string& filePath) override;
};

// Sanity checks
static_assert(sizeof(aiVector3D) == sizeof(Vector3));
static_assert(sizeof(aiVector2D) == sizeof(Vector2));