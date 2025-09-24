#pragma once

#include "TracerConstants.h"
#include "TracerEnums.h"
#include "MRayDescriptions.h"

#include "TransientPool/TransientPool.h"
#include "Core/DataStructures.h"


#define MRAY_GENERIC_ID(NAME, TYPE) enum class NAME : TYPE {}

using CommonId = MRay::CommonKey;
using CommonIdRange = Vector<2, CommonId>;

// Generic Attribute Info
struct GenericAttributeInfo
{
    std::string             name;
    MRayDataTypeRT          dataType;
    AttributeIsArray        isArray;
    AttributeOptionality    isOptional;
};

struct TexturedAttributeInfo
{
    std::string             name;
    MRayDataTypeRT          dataType;
    AttributeIsArray        isArray;
    AttributeOptionality    isOptional;
    AttributeTexturable     isTexturable;
    AttributeIsColor        isColor;
};

using GenericAttributeInfoList = StaticVector<GenericAttributeInfo,
                                              TracerConstants::MaxAttributePerGroup>;
using TexturedAttributeInfoList = StaticVector<TexturedAttributeInfo,
                                               TracerConstants::MaxAttributePerGroup>;
using TypeNameList = std::vector<std::string_view>;


// Prim related
MRAY_GENERIC_ID(PrimGroupId, CommonId);
MRAY_GENERIC_ID(PrimBatchId, CommonId);
struct PrimCount { uint32_t primCount; uint32_t attributeCount; };
using PrimBatchIdList = std::vector<PrimBatchId>;
struct PrimAttributeInfo
{
    PrimitiveAttributeLogic logic;
    MRayDataTypeRT          dataType;
    AttributeIsArray        isArray;
    AttributeOptionality    isOptional;
};
using PrimAttributeInfoList = StaticVector<PrimAttributeInfo,
                                           TracerConstants::MaxAttributePerGroup>;
// Texture Related
MRAY_GENERIC_ID(TextureId, CommonId);
// Transform Related
MRAY_GENERIC_ID(TransGroupId, CommonId);
MRAY_GENERIC_ID(TransformId, CommonId);
using TransAttributeInfo = GenericAttributeInfo;
using TransAttributeInfoList = GenericAttributeInfoList;
// Light Related
MRAY_GENERIC_ID(LightGroupId, CommonId);
MRAY_GENERIC_ID(LightId, CommonId);
using LightAttributeInfo = TexturedAttributeInfo;
using LightAttributeInfoList = TexturedAttributeInfoList;
// Camera Related
MRAY_GENERIC_ID(CameraGroupId, CommonId);
MRAY_GENERIC_ID(CameraId, CommonId);
using CamAttributeInfo = GenericAttributeInfo;
using CamAttributeInfoList = GenericAttributeInfoList;
// Material Related
MRAY_GENERIC_ID(MatGroupId, CommonId);
MRAY_GENERIC_ID(MaterialId, CommonId);
using MatAttributeInfo = TexturedAttributeInfo;
using MatAttributeInfoList = TexturedAttributeInfoList;
// Medium Related
MRAY_GENERIC_ID(MediumGroupId, CommonId);
MRAY_GENERIC_ID(MediumId, CommonId);
using MediumPair = Pair<MediumId, MediumId>;
using MediumAttributeInfo = TexturedAttributeInfo;
using MediumAttributeInfoList = TexturedAttributeInfoList;
// Surface Related
MRAY_GENERIC_ID(SurfaceId, CommonId);
MRAY_GENERIC_ID(LightSurfaceId, CommonId);
MRAY_GENERIC_ID(CamSurfaceId, CommonId);
using SurfaceMatList        = StaticVector<MaterialId, TracerConstants::MaxPrimBatchPerSurface>;
using SurfacePrimList       = StaticVector<PrimBatchId, TracerConstants::MaxPrimBatchPerSurface>;
using OptionalAlphaMapList  = StaticVector<Optional<TextureId>, TracerConstants::MaxPrimBatchPerSurface>;
using CullBackfaceFlagList  = StaticVector<bool, TracerConstants::MaxPrimBatchPerSurface>;
// Renderer Related
MRAY_GENERIC_ID(RendererId, CommonId);
using RendererAttributeInfo = GenericAttributeInfo;
using RendererAttributeInfoList = StaticVector<GenericAttributeInfo,
                                               TracerConstants::MaxRendererAttributeCount>;

using AttributeCountList = StaticVector<size_t, TracerConstants::MaxAttributePerGroup>;

// For transfer of options
struct RendererOptionPack
{
    static constexpr auto N = TracerConstants::MaxRendererAttributeCount;
    using AttributeList = StaticVector<TransientData, N>;
    //
    RendererAttributeInfoList   paramTypes;
    AttributeList               attributes;
};