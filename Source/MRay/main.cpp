
//#define MRAY_GPU_BACKEND_CUDA
//
//#include "Tracer/PrimitiveC.h"
//#include "Tracer/PrimitiveDefaultTriangle.h"

#include "Core/Types.h"

#include <iostream>

template <typename T> void TD() { std::cout << __FUNCSIG__ << std::endl; }

int main()
{

    TD<UniqueVariant<int, char, int, char, float, char, double>>();
}