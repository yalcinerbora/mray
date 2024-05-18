#include "Tracer.h"

#include "Tracer/PrimitiveDefaultTriangle.h"
#include "Tracer/MaterialsDefault.h"
#include "Tracer/Transforms.h"
#include "Tracer/LightsDefault.h"
#include "Tracer/AcceleratorLinear.h"

// TODO: After Reflection, maybe convert this to it?
template <class BaseAccel, class AccelGTypes, class AccelWorkTypes>
struct AccelTypePack
{
    using BaseType = BaseAccel;
    using GroupTypes = AccelGTypes;
    using WorkTypes = AccelWorkTypes;
};

using PrimGTypes = Tuple
<
    PrimGroupTriangle,
    PrimGroupSkinnedTriangle,
    // Sphere
>;

using MatGTypes = Tuple
<
    MatGroupLambert,
    //MatGroupUnreal
>;

using TransformGTypes = Tuple
<
    TransformGroupIdentity,
    TransformGroupSingle,
    TransformGroupMulti
>;

using LightGTypes = Tuple
<
    LightGroupPrim<PrimGroupTriangle>,
    LightGroupSkysphere<CoOctoCoordConverter>
>;

using LinearAccelTypePack = AccelTypePack
<
    BaseAcceleratorLinear,
    Tuple
    <
        AcceleratorGroupLinear<PrimGroupTriangle>,
        AcceleratorGroupLinear<PrimGroupSkinnedTriangle>
    >,
    Tuple
    <
        Tuple
        <
            AcceleratorWork<AcceleratorGroupLinear<PrimGroupTriangle>, TransformGroupIdentity>,
            AcceleratorWork<AcceleratorGroupLinear<PrimGroupTriangle>, TransformGroupSingle>,
        >,
        Tuple
        <
            AcceleratorWork<AcceleratorGroupLinear<PrimGroupSkinnedTriangle>, TransformGroupMulti>
        >
    >
>;
