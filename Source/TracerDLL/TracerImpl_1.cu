#include "Tracer.h"

#include "Tracer/PrimitiveDefaultTriangle.h"

TypeGeneratorPack Tracer::LibGlobalTypeGenerators = {};

template<class GeneratorType, class BaseType,
         class... Args, class... Types>
void GenMapping(std::map<std::string_view, GeneratorType>& map,
                // These types are here for overload resolution,
                // these will not be used directly
                Tuple<Args...>*, Tuple<Types...>*)
{
    // I've come up with this to move the Args... pack away from expansion
    // If you have asked me how does it work, I've no idea :)
    auto GenTypeWrapper = [&map]<typename T>()
    {
        map.emplace(T::TypeName(),
                    &GenerateType<BaseType, T, Args...>);
    };

    // Use comma operator expansion
    // Trick from here
    //https://codereview.stackexchange.com/questions/201106/iteration-over-zipped-tuples-for-each-in-tuples
    ((void)GenTypeWrapper.template operator()<Types>(), ...);
}

using PrimGTypes = Tuple
<
    PrimGroupTriangle,
    PrimGroupSkinnedTriangle
>;

Tracer::Tracer(BS::thread_pool& tp)
    : TracerBase(tp, LibGlobalTypeGenerators)
{
    Tuple<uint32_t, GPUSystem&>*    a = nullptr;
    PrimGTypes*                     b = nullptr;

    GenMapping<PrimGenerator, GenericGroupPrimitiveT>
    (
        LibGlobalTypeGenerators.primGenerator,
        a, b
    );
}