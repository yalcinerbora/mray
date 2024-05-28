#pragma once

#include "Tracer/TracerBase.h"

class Tracer final : public TracerBase
{
    private:
    static TypeGeneratorPack GLOBAL_TYPE_GEN;

    static void AddPrimGenerators(std::map<std::string_view, PrimGenerator>&);
    static void AddTransformGenerators(std::map<std::string_view, TransGenerator>&);
    static void AddLightGenerators(std::map<std::string_view, LightGenerator>&);
    static void AddCamGenerators(std::map<std::string_view, CamGenerator>&);
    static void AddMediumGenerators(std::map<std::string_view, MedGenerator>&);
    static void AddMaterialGenerators(std::map<std::string_view, MatGenerator>&);
    static void AddAccelGenerators(std::map<AcceleratorType, BaseAccelGenerator>&,
                                   std::map<AcceleratorType, AccelGroupGenMap>&,
                                   std::map<AcceleratorType, AccelWorkGenMap>&);

    //
    //static void AddRendererGenerators_1(std::map<std::string_view, PrimGenerator>&);
    //static void AddRendererGenerators_2(std::map<std::string_view, PrimGenerator>&);

    public:
    Tracer();
};

inline Tracer::Tracer()
    : TracerBase(GLOBAL_TYPE_GEN)
{
    // Types
    AddPrimGenerators(GLOBAL_TYPE_GEN.primGenerator);
    AddCamGenerators(GLOBAL_TYPE_GEN.transGenerator);
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