#include "TextureCUDA.h"

template class mray::cuda::TextureCUDA<2, Float>;
template class mray::cuda::TextureCUDA<2, Vector2>;
template class mray::cuda::TextureCUDA<2, Vector3>;
template class mray::cuda::TextureCUDA<2, Vector4>;

template class mray::cuda::TextureCUDA<2, uint8_t>;
template class mray::cuda::TextureCUDA<2, Vector2uc>;
template class mray::cuda::TextureCUDA<2, Vector3uc>;
template class mray::cuda::TextureCUDA<2, Vector4uc>;

template class mray::cuda::TextureCUDA<2, int8_t>;
template class mray::cuda::TextureCUDA<2, Vector2c>;
template class mray::cuda::TextureCUDA<2, Vector3c>;
template class mray::cuda::TextureCUDA<2, Vector4c>;

template class mray::cuda::TextureCUDA<2, uint16_t>;
template class mray::cuda::TextureCUDA<2, Vector2us>;
template class mray::cuda::TextureCUDA<2, Vector3us>;
template class mray::cuda::TextureCUDA<2, Vector4us>;

template class mray::cuda::TextureCUDA<2, int16_t>;
template class mray::cuda::TextureCUDA<2, Vector2s>;
template class mray::cuda::TextureCUDA<2, Vector3s>;
template class mray::cuda::TextureCUDA<2, Vector4s>;