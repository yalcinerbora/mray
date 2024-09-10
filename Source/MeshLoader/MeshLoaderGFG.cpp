#include "MeshLoaderGFG.h"
#include "Core/ShapeFunctions.h"
#include "Core/Error.hpp"

#include <filesystem>
#include <gfg/GFGFileExporter.h>

Optional<GFGVertexComponent> MeshViewGFG::FindComponent(PrimitiveAttributeLogic l) const
{
    for(const GFGVertexComponent& c : gfgFile.loader.Header().meshes[innerIndex].components)
    {
        using enum GFGVertexComponentLogic;
        bool eq = ((c.logic == POSITION && l == PrimitiveAttributeLogic::POSITION) ||
                   (c.logic == NORMAL && l == PrimitiveAttributeLogic::NORMAL) ||
                   (c.logic == BINORMAL && l == PrimitiveAttributeLogic::BITANGENT) ||
                   (c.logic == TANGENT && l == PrimitiveAttributeLogic::TANGENT) ||
                   (c.logic == UV && l == PrimitiveAttributeLogic::UV0) ||
                   (c.logic == WEIGHT && l == PrimitiveAttributeLogic::WEIGHT) ||
                   (c.logic == WEIGHT_INDEX && l == PrimitiveAttributeLogic::WEIGHT_INDEX));
        if(eq) return c;
    }
    return std::nullopt;
}

MRayDataTypeRT MeshViewGFG::GFGDataTypeToMRayDataType(GFGDataType t)
{
    static_assert(std::is_same_v<Float, float>,
                  "Currently \"MeshLoaderGFG\" do not support double "
                  "precision mode change this later.");

    using enum MRayDataEnum;
    switch(t)
    {
        case GFGDataType::FLOAT_1:  return MRayDataTypeRT(MRayDataType<MR_FLOAT>{});
        case GFGDataType::FLOAT_2:  return MRayDataTypeRT(MRayDataType<MR_VECTOR_2>{});
        case GFGDataType::FLOAT_3:  return MRayDataTypeRT(MRayDataType<MR_VECTOR_3>{});
        case GFGDataType::FLOAT_4:  return MRayDataTypeRT(MRayDataType<MR_VECTOR_4>{});

        case GFGDataType::INT8_1:   return MRayDataTypeRT(MRayDataType<MR_INT8>{});
        case GFGDataType::INT8_2:   return MRayDataTypeRT(MRayDataType<MR_VECTOR_2C>{});
        case GFGDataType::INT8_3:   return MRayDataTypeRT(MRayDataType<MR_VECTOR_3C>{});
        case GFGDataType::INT8_4:   return MRayDataTypeRT(MRayDataType<MR_VECTOR_4C>{});

        case GFGDataType::UINT8_1:  return MRayDataTypeRT(MRayDataType<MR_UINT8>{});
        case GFGDataType::UINT8_2:  return MRayDataTypeRT(MRayDataType<MR_VECTOR_2UC>{});
        case GFGDataType::UINT8_3:  return MRayDataTypeRT(MRayDataType<MR_VECTOR_3UC>{});
        case GFGDataType::UINT8_4:  return MRayDataTypeRT(MRayDataType<MR_VECTOR_4UC>{});

        case GFGDataType::INT16_1: return MRayDataTypeRT(MRayDataType<MR_INT16>{});
        case GFGDataType::INT16_2: return MRayDataTypeRT(MRayDataType<MR_VECTOR_2S>{});
        case GFGDataType::INT16_3: return MRayDataTypeRT(MRayDataType<MR_VECTOR_3S>{});
        case GFGDataType::INT16_4: return MRayDataTypeRT(MRayDataType<MR_VECTOR_4S>{});

        case GFGDataType::UINT16_1: return MRayDataTypeRT(MRayDataType<MR_UINT16>{});
        case GFGDataType::UINT16_2: return MRayDataTypeRT(MRayDataType<MR_VECTOR_2US>{});
        case GFGDataType::UINT16_3: return MRayDataTypeRT(MRayDataType<MR_VECTOR_3US>{});
        case GFGDataType::UINT16_4: return MRayDataTypeRT(MRayDataType<MR_VECTOR_4US>{});

        case GFGDataType::INT32_1: return MRayDataTypeRT(MRayDataType<MR_INT32>{});
        case GFGDataType::INT32_2: return MRayDataTypeRT(MRayDataType<MR_VECTOR_2I>{});
        case GFGDataType::INT32_3: return MRayDataTypeRT(MRayDataType<MR_VECTOR_3I>{});
        case GFGDataType::INT32_4: return MRayDataTypeRT(MRayDataType<MR_VECTOR_4I>{});

        case GFGDataType::UINT32_1: return MRayDataTypeRT(MRayDataType<MR_UINT32>{});
        case GFGDataType::UINT32_2: return MRayDataTypeRT(MRayDataType<MR_VECTOR_2UI>{});
        case GFGDataType::UINT32_3: return MRayDataTypeRT(MRayDataType<MR_VECTOR_3UI>{});
        case GFGDataType::UINT32_4: return MRayDataTypeRT(MRayDataType<MR_VECTOR_4UI>{});

        case GFGDataType::INT64_1: return MRayDataTypeRT(MRayDataType<MR_INT64>{});
        case GFGDataType::INT64_2: return MRayDataTypeRT(MRayDataType<MR_VECTOR_2L>{});
        case GFGDataType::INT64_3: return MRayDataTypeRT(MRayDataType<MR_VECTOR_3L>{});
        case GFGDataType::INT64_4: return MRayDataTypeRT(MRayDataType<MR_VECTOR_4L>{});

        case GFGDataType::UINT64_1: return MRayDataTypeRT(MRayDataType<MR_UINT64>{});
        case GFGDataType::UINT64_2: return MRayDataTypeRT(MRayDataType<MR_VECTOR_2UL>{});
        case GFGDataType::UINT64_3: return MRayDataTypeRT(MRayDataType<MR_VECTOR_3UL>{});
        case GFGDataType::UINT64_4: return MRayDataTypeRT(MRayDataType<MR_VECTOR_4UL>{});

        case GFGDataType::NORM8_4:  return MRayDataTypeRT(MRayDataType<MR_SNORM_4x8>{});
        case GFGDataType::UNORM8_4: return MRayDataTypeRT(MRayDataType<MR_UNORM_4x8>{});

        case GFGDataType::NORM16_2: return MRayDataTypeRT(MRayDataType<MR_SNORM_2x16>{});
        case GFGDataType::UNORM16_2:return MRayDataTypeRT(MRayDataType<MR_UNORM_2x16>{});

        case GFGDataType::QUATERNION: return MRayDataTypeRT(MRayDataType<MR_QUATERNION>{});

        // These are not supported
        // TODO: Change this
        case GFGDataType::NORM8_1:
        case GFGDataType::NORM8_2:
        case GFGDataType::NORM8_3:
        case GFGDataType::UNORM8_1:
        case GFGDataType::UNORM8_2:
        case GFGDataType::UNORM8_3:
        case GFGDataType::NORM16_1:
        case GFGDataType::NORM16_3:
        case GFGDataType::UNORM16_1:
        case GFGDataType::UNORM16_3:
        case GFGDataType::NORM32_1:
        case GFGDataType::NORM32_3:
        case GFGDataType::UNORM32_1:
        case GFGDataType::UNORM32_3:
        case GFGDataType::NORM16_4:
        case GFGDataType::NORM32_2:
        case GFGDataType::NORM32_4:
        case GFGDataType::UNORM16_4:
        case GFGDataType::UNORM32_2:
        case GFGDataType::UNORM32_4:
        case GFGDataType::HALF_1:
        case GFGDataType::HALF_2:
        case GFGDataType::HALF_3:
        case GFGDataType::HALF_4:
        case GFGDataType::QUADRUPLE_1:
        case GFGDataType::QUADRUPLE_2:
        case GFGDataType::QUADRUPLE_3:
        case GFGDataType::QUADRUPLE_4:
        case GFGDataType::NORM_2_10_10_10:
        case GFGDataType::UNORM_2_10_10_10:
        case GFGDataType::UINT_10F_11F_11F:
        case GFGDataType::CUSTOM_1_15N_16N:
        case GFGDataType::CUSTOM_TANG_H_2N:
        case GFGDataType::UNORM8_4_4:
        case GFGDataType::UNORM16_2_4:
        case GFGDataType::UINT8_4_4:
        case GFGDataType::UINT16_2_4:
        case GFGDataType::UINT_2_10_10_10:
        case GFGDataType::DOUBLE_1:
        case GFGDataType::DOUBLE_2:
        case GFGDataType::DOUBLE_3:
        case GFGDataType::DOUBLE_4:
        case GFGDataType::END:
        default: throw MRayError("GFG: Unsupported data type");
    }
}

MeshViewGFG::MeshViewGFG(uint32_t innerIndexIn,
                         const MeshFileGFG& gfgFileIn)
    : innerIndex(innerIndexIn)
    , gfgFile(gfgFileIn)
{
    if(innerIndex >= gfgFile.loader.Header().meshes.size())
        throw MRayError("GFG: Inner index out of range  \"{}\"",
                        gfgFile.Name());

    // Analyse the mesh
    for(const GFGVertexComponent& vc : gfgFile.loader.Header().meshes[innerIndex].components)
    {
        if(vc.internalOffset != 0)
            throw MRayError("GFG: Vertex data types must be \"struct of arrays\" format \"{}\"",
                            gfgFile.Name());
    }
}

AABB3 MeshViewGFG::AABB() const
{
    const GFGMeshHeader& m = gfgFile.loader.Header().meshes[innerIndex];
    AABB3 result(m.headerCore.aabb.min,
                 m.headerCore.aabb.max);
    return result;
}

uint32_t MeshViewGFG::MeshPrimitiveCount() const
{
    const GFGMeshHeader& m = gfgFile.loader.Header().meshes[innerIndex];
    return (static_cast<uint32_t>(m.headerCore.indexCount) /
            Shape::Triangle::TRI_VERTEX_COUNT);
}

uint32_t MeshViewGFG::MeshAttributeCount() const
{
    const GFGMeshHeader& m = gfgFile.loader.Header().meshes[innerIndex];
    return static_cast<uint32_t>(m.headerCore.vertexCount);
}

std::string MeshViewGFG::Name() const
{
    return gfgFile.Name();
}

uint32_t MeshViewGFG::InnerIndex() const
{
    return innerIndex;
}

bool MeshViewGFG::HasAttribute(PrimitiveAttributeLogic logic) const
{
    // By definition, GFG is indexed.
    if(logic == PrimitiveAttributeLogic::INDEX)
        return true;
    return FindComponent(logic).has_value();
}

TransientData MeshViewGFG::GetAttribute(PrimitiveAttributeLogic logic) const
{
    const auto& m = gfgFile.loader.Header().meshes[innerIndex];

    if(logic == PrimitiveAttributeLogic::INDEX)
    {
        MRayDataTypeRT dataType;
        using enum MRayDataEnum;
        switch(m.headerCore.indexSize)
        {
            case 1: dataType = MRayDataTypeRT(MRayDataType<MR_VECTOR_3UC>{}); break;
            case 2: dataType = MRayDataTypeRT(MRayDataType<MR_VECTOR_3US>{}); break;
            case 4: dataType = MRayDataTypeRT(MRayDataType<MR_VECTOR_3UI>{}); break;
            case 8: dataType = MRayDataTypeRT(MRayDataType<MR_VECTOR_3UL>{}); break;
            default: throw MRayError("GFG: Unkown index layout \"{}\"", gfgFile.Name());
        }
        return std::visit([&](auto&& v) -> TransientData
        {
            using T = std::remove_cvref_t<decltype(v)>::Type;
            size_t count = gfgFile.loader.MeshIndexDataSize(innerIndex) / v.Size;
            TransientData result(std::in_place_type_t<T>{}, count);
            result.ReserveAll();
            Span<T> span = result.AccessAs<T>();
            gfgFile.loader.MeshIndexData(reinterpret_cast<uint8_t*>(span.data()),
                                         innerIndex);
            return result;

        }, dataType);
    }
    else
    {
        OptionalComponent c = FindComponent(logic);
        if(!c.has_value())
            throw MRayError("GFG: File do not have attribute of {}, \"{}\"",
                            PrimAttributeStringifier::ToString(logic),
                            gfgFile.Name());
        const auto& comp = c.value();
        MRayDataTypeRT type = GFGDataTypeToMRayDataType(comp.dataType);

        return std::visit([&](auto&& v) -> TransientData
        {
            using T = std::remove_cvref_t<decltype(v)>::Type;
            size_t count = gfgFile.loader.MeshVertexComponentDataGroupSize(innerIndex,
                                                                           comp.logic) / v.Size;
            TransientData result(std::in_place_type_t<T>{}, count);
            result.ReserveAll();
            Span<Byte> span = result.AccessAs<Byte>();
            // TODO: Change this later
            gfgFile.loader.MeshVertexComponentDataGroup(reinterpret_cast<uint8_t*>(span.data()),
                                                        innerIndex, comp.logic);

            return result;
        }, type);
    }
}

MRayDataTypeRT MeshViewGFG::AttributeLayout(PrimitiveAttributeLogic logic) const
{
    using enum MRayDataEnum;
    const GFGMeshHeader& m = gfgFile.loader.Header().meshes[innerIndex];
    if(logic == PrimitiveAttributeLogic::INDEX)
    {
        switch(m.headerCore.indexSize)
        {
            case 1: return MRayDataTypeRT(MRayDataType<MR_VECTOR_3UC>{});
            case 2: return MRayDataTypeRT(MRayDataType<MR_VECTOR_3US>{});
            case 4: return MRayDataTypeRT(MRayDataType<MR_VECTOR_3UI>{});
            case 8: return MRayDataTypeRT(MRayDataType<MR_VECTOR_3UL>{});
            default: throw MRayError("GFG: Unkown index layout \"{}\"", gfgFile.Name());
        }
    }

    OptionalComponent c = FindComponent(logic);
    if(!c.has_value())
        throw MRayError("GFG: File do not have attribute of {}, \"{}\"",
                        PrimAttributeStringifier::ToString(logic),
                        gfgFile.Name());

    const auto& comp = c.value();
    return GFGDataTypeToMRayDataType(comp.dataType);
}

MeshFileGFG::MeshFileGFG(const std::string& filePath)
    : file(filePath, std::ofstream::binary)
    , reader(file)
    , loader(&reader)
    , fileName(std::filesystem::path(filePath).filename().string())
{
    if(!file.is_open())
        throw MRayError("GFG: Unable to open file \"{}\"", Name());

    GFGFileError err = loader.ValidateAndOpen();
    if(err != GFGFileError::OK)
    {
        throw MRayError("GFG: Corrupted file \"{}\"", Name());
    }
}

std::unique_ptr<MeshFileViewI>
MeshFileGFG::ViewMesh(uint32_t innerIndex)
{
    return std::unique_ptr<MeshFileViewI>(new MeshViewGFG(innerIndex, *this));
}

std::string MeshFileGFG::Name() const
{
    return fileName;
}

std::unique_ptr<MeshFileI> MeshLoaderGFG::OpenFile(std::string& filePath)
{
    return std::unique_ptr<MeshFileI>(new MeshFileGFG(filePath));
}