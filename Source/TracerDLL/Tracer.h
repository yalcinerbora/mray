#pragma once

#include "Tracer/TracerBase.h"

class Tracer final : public TracerBase
{
    private:
    static TypeGeneratorPack GLOBAL_TYPE_GEN;

    static void AddPrimGenerators(Map<std::string_view, PrimGenerator>&);
    static void AddTransformGenerators(Map<std::string_view, TransGenerator>&);
    static void AddLightGenerators(Map<std::string_view, LightGenerator>&);
    static void AddCamGenerators(Map<std::string_view, CamGenerator>&);
    static void AddMediumGenerators(Map<std::string_view, MedGenerator>&);
    static void AddMaterialGenerators(Map<std::string_view, MatGenerator>&);
    static void AddAccelGenerators(Map<AcceleratorType, BaseAccelGenerator>&,
                                   Map<AcceleratorType, AccelGroupGenMap>&,
                                   Map<AcceleratorType, AccelWorkGenMap>&);

    //
    //static void AddRendererGenerators_1(Map<std::string_view, PrimGenerator>&);
    //static void AddRendererGenerators_2(Map<std::string_view, PrimGenerator>&);

    public:
    Tracer(const TracerParameters& tracerParams);
};

inline Tracer::Tracer(const TracerParameters& tracerParams)
    : TracerBase(GLOBAL_TYPE_GEN, tracerParams)
{
    // Types
    AddPrimGenerators(GLOBAL_TYPE_GEN.primGenerator);
    AddCamGenerators(GLOBAL_TYPE_GEN.camGenerator);
    AddMediumGenerators(GLOBAL_TYPE_GEN.medGenerator);
    AddMaterialGenerators(GLOBAL_TYPE_GEN.matGenerator);
    AddTransformGenerators(GLOBAL_TYPE_GEN.transGenerator);
    AddLightGenerators(GLOBAL_TYPE_GEN.lightGenerator);
    AddAccelGenerators(GLOBAL_TYPE_GEN.baseAcceleratorGenerator,
                       GLOBAL_TYPE_GEN.accelGeneratorMap,
                       GLOBAL_TYPE_GEN.accelWorkGeneratorMap);
    // Related Types


    // Finally Populate the lists
    PopulateAttribInfoAndTypeLists();
}