#include <gtest/gtest.h>
#include "Tracer/Random.h"

#include "Device/GPUSystem.hpp"


MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
void KCRayConeTest(MRAY_GRID_CONSTANT RayCone cone,
                   MRAY_GRID_CONSTANT Vector3 f,
                   MRAY_GRID_CONSTANT Vector3 d)
{
    KernelCallParams kp;
    if(kp.GlobalId() != 0) return;

    auto [a1, a2] = cone.Project(f, d);

}

TEST(RayCone, Project)
{
    // Value from an assert (happens on clang)
    //static constexpr RayCone coneTest =
    //{
    //    .aperture = Float(0.000318206352),
    //    .width = Float(0.00251944619)
    //};
    //static constexpr Vector3 f = Vector3(-0, -0, -1);
    //static constexpr Vector3 d = Vector3(-0, -0, 1);
    static constexpr RayCone coneTest =
    {
        .aperture = Float(0),
        .width = Float(-5.33283590e+19)
    };
    static constexpr Vector3 f = Vector3(-0.867630541, -0.418694645, 0.268164247);
    static constexpr Vector3 d = Vector3(-0.167027116, -0.138377547, -0.976193428);


    GPUSystem gpuSystem;
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    queue.IssueBlockKernel<KCRayConeTest>
    (
        "KCRayConeTest",
        DeviceBlockIssueParams{.gridSize = 1, .blockSize = 1},
        coneTest,
        f,
        d
    );
    queue.Barrier().Wait();
}
