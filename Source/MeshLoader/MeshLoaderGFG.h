#pragma once

#include "EntryPoint.h"
#include <gfg/GFGFileLoader.h>
#include <fstream>

class MeshFileGFG : public MeshFileI
{
    using OptionalComponent = Optional<GFGVertexComponent>;

    private:
    std::ifstream           file;
    GFGFileReaderSTL        reader;
    mutable GFGFileLoader   loader;
    std::string             fileName;
    uint32_t                innerIndex;

    OptionalComponent       FindComponent(PrimitiveAttributeLogic) const;
    static MRayDataTypeRT   GFGDataTypeToMRayDataType(GFGDataType);

    public:
                    MeshFileGFG(const std::string& filePath,
                                uint32_t internalIndex = 0);

    AABB3           AABB() const override;
    uint32_t        MeshPrimitiveCount() const override;
    uint32_t        MeshAttributeCount() const override;
    std::string     Name() const override;

   bool            HasAttribute(PrimitiveAttributeLogic) const override;
   TransientData   GetAttribute(PrimitiveAttributeLogic) const override;
   MRayDataTypeRT  AttributeLayout(PrimitiveAttributeLogic) const override;

};

class MeshLoaderGFG : public MeshLoaderI
{
    public:
    static constexpr std::string_view   Tag = "gfg";

    private:
    public:
                                    MeshLoaderGFG() = default;
    std::unique_ptr<MeshFileI>      OpenFile(std::string& filePath,
                                             uint32_t internalIndex) override;
};