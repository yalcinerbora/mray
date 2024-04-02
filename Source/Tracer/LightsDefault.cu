#include "LightsDefault.h"
#include "PrimitiveDefaultTriangle.h"

template class LightGroupSkysphere<CoOctoCoordConverter>;
template class LightGroupSkysphere<SphericalCoordConverter>;

template class LightGroupPrim<PrimGroupTriangle>;