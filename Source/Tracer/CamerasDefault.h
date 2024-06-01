#pragma once

#include "CameraC.h"
#include "Random.h"
#include "Core/GraphicsFunctions.h"

namespace CameraDetail
{
    struct alignas(64) CamPinholeData
    {
        Span<const Vector4> fovAndPlanes;
        Span<const Vector3> position;
        Span<const Vector3> gaze;
        Span<const Vector3> up;
    };

    class CameraPinhole
    {
        public:
        using DataSoA = CamPinholeData;

        private:
        Vector3 position;
        Vector3 bottomLeft;
        Vector2 planeSize;
        Vector2 nearFar;
        Vector2 fov;
        // Basis
        Vector3 gazeDir;
        Vector3 up;
        Vector3 right;

        public:
        MRAY_HYBRID CameraPinhole(const DataSoA&, CameraKey);
        // Ray Sampling
        MRAY_HYBRID
        RaySample   SampleRay(// Input
                              const Vector2ui& generationIndex,
                              const Vector2ui& stratumCount,
                              // I-O
                              RNGDispenser&) const;
        MRAY_HYBRID
        Float       PdfRay(const Ray&) const;
        MRAY_HYBRID
        uint32_t    SampleRayRNCount() const;
        // Misc
        MRAY_HYBRID
        bool        CanBeSampled() const;
        MRAY_HYBRID
        CameraTransform GetCameraTransform() const;
        MRAY_HYBRID
        void            OverrideTransform(const CameraTransform&);
        MRAY_HYBRID
        CameraPinhole   GenerateSubCamera(const Vector2ui& regionId,
                                          const Vector2ui& regionCount) const;
    };
}

class CameraGroupPinhole : public GenericGroupCamera<CameraGroupPinhole>
{
    public:
    using DataSoA   = CameraDetail::CamPinholeData;
    using Camera    = CameraDetail::CameraPinhole;

    private:
    Span<Vector4>   dFovAndPlanes;
    Span<Vector3>   dGazePoints;
    Span<Vector3>   dPositions;
    Span<Vector3>   dUpVectors;
    DataSoA         soa;

    std::vector<CameraTransform>    hCameraTansforms;
    std::vector<Vector4>            hFovAndPlanes;

    public:
    static std::string_view TypeName();
    //
                            CameraGroupPinhole(uint32_t groupId,
                                               const GPUSystem& system);

    void                    CommitReservations() override;
    CamAttributeInfoList    AttributeInfo() const override;
    void                    PushAttribute(CameraKey camKey,
                                          uint32_t attributeIndex,
                                          TransientData data,
                                          const GPUQueue& queue) override;
    void                    PushAttribute(CameraKey camKey,
                                          uint32_t attributeIndex,
                                          const Vector2ui& subRange,
                                          TransientData data,
                                          const GPUQueue& queue) override;
    void                    PushAttribute(CameraKey idStart, CameraKey idEnd,
                                          uint32_t attributeIndex,
                                          TransientData data,
                                          const GPUQueue& queue) override;
    CameraTransform         AcquireCameraTransform(CameraKey) const override;

    DataSoA                 SoA() const;
};

#include "CamerasDefault.hpp"

static_assert(CameraC<CameraDetail::CameraPinhole>);
static_assert(CameraGroupC<CameraGroupPinhole>);

