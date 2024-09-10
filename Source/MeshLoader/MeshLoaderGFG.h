#pragma once

#include "EntryPoint.h"
#include <gfg/GFGFileLoader.h>
#include <fstream>

class MeshFileGFG;

class MeshViewGFG : public MeshFileViewI
{
    friend class MeshFileGFG;
    using OptionalComponent = Optional<GFGVertexComponent>;

    private:
    uint32_t                innerIndex;
    const MeshFileGFG&      gfgFile;

    OptionalComponent       FindComponent(PrimitiveAttributeLogic) const;
    static MRayDataTypeRT   GFGDataTypeToMRayDataType(GFGDataType);
    //
                            MeshViewGFG(uint32_t innerIndex,
                                        const MeshFileGFG& gfgFile);
    public:
    AABB3           AABB() const override;
    uint32_t        MeshPrimitiveCount() const override;
    uint32_t        MeshAttributeCount() const override;
    std::string     Name() const override;
    uint32_t        InnerIndex() const override;

    bool            HasAttribute(PrimitiveAttributeLogic) const override;
    TransientData   GetAttribute(PrimitiveAttributeLogic) const override;
    MRayDataTypeRT  AttributeLayout(PrimitiveAttributeLogic) const override;
};

class MeshFileGFG : public MeshFileI
{
    friend class MeshViewGFG;

    private:
    std::ifstream           file;
    GFGFileReaderSTL        reader;
    mutable GFGFileLoader   loader;
    std::string             fileName;

    public:
    MeshFileGFG(const std::string& filePath);

    std::unique_ptr<MeshFileViewI>
    ViewMesh(uint32_t innerIndex) override;

    std::string Name() const override;
};

class MeshLoaderGFG : public MeshLoaderI
{
    public:
    static constexpr std::string_view   Tag = "gfg";

    private:
    public:
                                    MeshLoaderGFG() = default;
    std::unique_ptr<MeshFileI>      OpenFile(std::string& filePath) override;
};