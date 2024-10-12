#include "Tracer.h"
#include "RequestedTypes.h"

#include "Device/GPUAlgReduce.h"
#include "Device/GPUAlgGeneric.h"
#include "Device/GPUAlgRadixSort.h"

#include "Core/Error.hpp"

TypeGeneratorPack Tracer::GLOBAL_TYPE_GEN = {};

void Tracer::AddPrimGenerators(Map<std::string_view, PrimGenerator>& map)
{
    using Args = Tuple<uint32_t, const GPUSystem&>;

    Args*           resolver0 = nullptr;
    PrimGTypes*     resolver1 = nullptr;
    GenerateMapping<PrimGenerator, GenericGroupPrimitiveT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddTransformGenerators(Map<std::string_view, TransGenerator>& map)
{
    using Args = Tuple<uint32_t, const GPUSystem&>;

    Args*               resolver0 = nullptr;
    TransformGTypes*    resolver1 = nullptr;
    GenerateMapping<TransGenerator, GenericGroupTransformT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddLightGenerators(Map<std::string_view, LightGenerator>& map)
{
    using Args = Tuple<uint32_t, const GPUSystem&, const TextureViewMap&,
                       const TextureMap&, GenericGroupPrimitiveT&>;

    Args*               resolver0 = nullptr;
    LightGTypes*        resolver1 = nullptr;
    GenerateMapping<LightGenerator, GenericGroupLightT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddCamGenerators(Map<std::string_view, CamGenerator>& map)
{
    using Args = Tuple<uint32_t, const GPUSystem&>;

    Args*       resolver0 = nullptr;
    CamGTypes*  resolver1 = nullptr;
    GenerateMapping<CamGenerator, GenericGroupCameraT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddMaterialGenerators(Map<std::string_view, MatGenerator>& map)
{
    using Args = Tuple<uint32_t, const GPUSystem&,
                       const TextureViewMap&, const TextureMap&>;

    Args*       resolver0 = nullptr;
    MatGTypes*  resolver1 = nullptr;
    GenerateMapping<MatGenerator, GenericGroupMaterialT>
    (
        map, resolver0, resolver1
    );
}

void Tracer::AddMediumGenerators(Map<std::string_view, MedGenerator>& map)
{
    using Args = Tuple<uint32_t, const GPUSystem&,
                       const TextureViewMap&, const TextureMap&>;

    Args*       resolver0 = nullptr;
    MedGTypes*  resolver1 = nullptr;
    GenerateMapping<MedGenerator, GenericGroupMediumT>
    (
        map, resolver0, resolver1
    );
}

template<class AcceleratorPack>
static void AddAccelGeneratorsGeneric(Map<AcceleratorType, BaseAccelGenerator>& baseMap,
                                      Map<AcceleratorType, AccelGroupGenMap>& groupMap,
                                      Map<AcceleratorType, AccelWorkGenMap>& workMap,
                                      AcceleratorType t)
{
    // Base
    using LinAccel = typename AcceleratorPack::BaseType;
    baseMap.emplace(t, &GenerateType<BaseAcceleratorI, LinAccel,
                                     BS::thread_pool&, const GPUSystem&,
                                     const AccelGroupGenMap&,
                                     const AccelWorkGenMap&>);

    using GroupGenArgs = Tuple<uint32_t, BS::thread_pool&, const GPUSystem&,
                               const GenericGroupPrimitiveT&,
                               const AccelWorkGenMap&>;
    using AccelGTypes = typename AcceleratorPack::GroupTypes;
    GroupGenArgs*   groupResolver0 = nullptr;
    AccelGTypes*    groupResolver1 = nullptr;
    auto& genMap = groupMap.emplace(t, AccelGroupGenMap()).first->second;
    GenerateMapping<AccelGroupGenerator, AcceleratorGroupI>
    (
        genMap,
        groupResolver0,
        groupResolver1
    );

    // Works
    auto& workMapGlobal = workMap.emplace(t, AccelWorkGenMap()).first->second;
    using WorkGenArgs = Tuple<const AcceleratorGroupI&, const GenericGroupTransformT&>;
    using AccelWTypes = typename AcceleratorPack::WorkTypes;
    WorkGenArgs* workResolver0 = nullptr;
    AccelWTypes* workResolver1 = nullptr;
    GenerateMapping<AccelWorkGenerator, AcceleratorWorkI>
    (
        workMapGlobal,
        workResolver0,
        workResolver1
    );
}

void Tracer::AddAccelGenerators(Map<AcceleratorType, BaseAccelGenerator>& baseMap,
                                Map<AcceleratorType, AccelGroupGenMap>& groupMap,
                                Map<AcceleratorType, AccelWorkGenMap>& workMap)
{
    using enum AcceleratorType::E;
    AddAccelGeneratorsGeneric<DefaultLinearAccelTypePack>
    (
        baseMap,
        groupMap,
        workMap,
        AcceleratorType{ SOFTWARE_NONE }
    );

    AddAccelGeneratorsGeneric<DefaultBVHAccelTypePack>
    (
        baseMap,
        groupMap,
        workMap,
        AcceleratorType{ SOFTWARE_BASIC_BVH }
    );

    #ifdef MRAY_ENABLE_HW_ACCELERATION
        AddAccelGeneratorsGeneric<DefaultDeviceAccelTypePack>
        (
            baseMap,
            groupMap,
            workMap,
            AcceleratorType{ HARDWARE }
        );
    #endif
}