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

struct RendererAttributeInfo
{
    std::string             name;
    MRayDataTypeRT          dataType;
    AttributeOptionality    isOptional;
    Optional<uint32_t>      enumerationIndex      = std::nullopt;
    uint32_t                hotKeyIndex           = std::numeric_limits<uint32_t>::max();
    bool                    requiresRenderRestart = true;
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
MRAY_GENERIC_ID(TopologyId, CommonId);
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
using MediumAttributeInfo = TexturedAttributeInfo;
using MediumAttributeInfoList = TexturedAttributeInfoList;
// Surface Related
MRAY_GENERIC_ID(SurfaceId, CommonId);
MRAY_GENERIC_ID(LightSurfaceId, CommonId);
MRAY_GENERIC_ID(CamSurfaceId, CommonId);
MRAY_GENERIC_ID(VolumeId, CommonId);

struct VolumeParams
{
    MediumId    mediumId;
    TransformId transformId;
    int32_t     priority;

    auto operator<=>(const VolumeParams&) const = default;
};

using SurfaceMatList        = StaticVector<MaterialId, TracerConstants::MaxPrimBatchPerSurface>;
using SurfacePrimList       = StaticVector<PrimBatchId, TracerConstants::MaxPrimBatchPerSurface>;
using OptionalAlphaMapList  = StaticVector<Optional<TextureId>, TracerConstants::MaxPrimBatchPerSurface>;
using CullBackfaceFlagList  = StaticVector<bool, TracerConstants::MaxPrimBatchPerSurface>;
using SurfaceVolumeList     = StaticVector<VolumeId, TracerConstants::MaxPrimBatchPerSurface>;
using BoundaryVolumeList    = StaticVector<VolumeId, TracerConstants::MaxNestedVolumes>;
//
using TopologyLayerSizeList = StaticVector<size_t, TracerConstants::MaxSparseTopologyLayers>;

// Renderer Related
MRAY_GENERIC_ID(RendererId, CommonId);
using RendererAttributeInfo = RendererAttributeInfo;
struct RendererAttributeInfoList
{
    struct EnumInfo
    {
        static constexpr auto N = 32;
        StaticVector<std::string, N> enumNames;
    };
    using EnumNameList = StaticVector<EnumInfo, TracerConstants::MaxRendererEnumCount>;
    using AttributeInfoList = StaticVector<RendererAttributeInfo, TracerConstants::MaxRendererAttributeCount>;
    //
    AttributeInfoList attributeInfos;
    EnumNameList      enumInfos;

    template<class NamedEnumT>
    static EnumInfo FromNamedEnum();
};
//
using AttributeCountList = StaticVector<size_t, TracerConstants::MaxAttributePerGroup>;

// For transfer of options with the definitions
struct RendererOptionPack
{
    private:
    static constexpr auto N = TracerConstants::MaxRendererAttributeCount;
    static constexpr auto M = 16;
    using AttributeList = StaticVector<TransientData, N>;

    public:
    RendererAttributeInfoList   paramInfos;
    AttributeList               attributes;

    template<class T> void PushAttribute(const T&);
    template<class T> void PushNamedEnum(const T&);
    template<class T> void PushString(const T&);
};

template<class NamedEnumT>
typename RendererAttributeInfoList::EnumInfo
RendererAttributeInfoList::FromNamedEnum()
{
    static_assert(EnumInfo::N > NamedEnumT::Names.size(),
                  "Enumeration names does not fit on to the renderer option pack!");
    EnumInfo e;
    for(size_t i = 0; i < NamedEnumT::Names.size(); i++)
    {
        e.enumNames.push_back(NamedEnumT::Names[i]);
    }
    return e;
}

template<class T>
inline void RendererOptionPack::PushAttribute(const T& t)
{
    attributes.push_back(TransientData(std::in_place_type_t<T>{}, 1));
    auto buffer = attributes.back().AccessAs<T>();
    auto readBuffer = Span<const T>(&t, 1);
    std::copy(readBuffer.cbegin(), readBuffer.cend(), buffer.begin());
}

template<class NamedEnumT>
inline void RendererOptionPack::PushNamedEnum(const NamedEnumT& t)
{
    using Enum = typename NamedEnumT::E;
    using IntType = std::underlying_type_t<Enum >;
    IntType tInt = IntType(static_cast<Enum>(t));

    attributes.push_back(TransientData(std::in_place_type_t<IntType >{}, 1));
    auto buffer = attributes.back().AccessAs<IntType>();
    auto readBuffer = Span<const IntType>(&tInt, 1);
    std::copy(readBuffer.cbegin(), readBuffer.cend(), buffer.begin());
}

template<class T>
inline void RendererOptionPack::PushString(const T& t)
{
    using ActualT = std::remove_cvref_t<T>;
    static_assert(std::is_same_v<ActualT, std::string> ||
                  std::is_same_v<ActualT, std::string_view>,
                  "As String only std::string, std::string_view is supported!");

    attributes.push_back(TransientData(std::in_place_type_t<std::string_view>{}, t.size()));
    auto buffer = attributes.back().AccessAsString();
    std::copy(t.cbegin(), t.cend(), buffer.begin());
}