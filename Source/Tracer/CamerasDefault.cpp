#include "CamerasDefault.h"
#include "GenericGroup.hpp"

#include "Core/TypeNameGenerators.h"

#ifdef MRAY_GPU_BACKEND_CPU
    #include "Device/GPUSystem.hpp" // IWYU pragma: keep
#endif

std::string_view CameraGroupPinhole::TypeName()
{
    using namespace TypeNameGen::CompTime;
    using namespace std::string_view_literals;
    static constexpr auto Name = "Pinhole"sv;
    return CameraTypeName<Name>;
}

CameraGroupPinhole::CameraGroupPinhole(uint32_t groupId,
                                       const GPUSystem& system)
    : GenericGroupCamera<CameraGroupPinhole>(groupId, system)
{}

void CameraGroupPinhole::CommitReservations()
{
    GenericCommit(Tie(dFovAndPlanes, dGazePoints, dPositions, dUpVectors),
                  {0, 0, 0, 0});

    soa = DataSoA
    {
        .fovAndPlanes = ToConstSpan(dFovAndPlanes),
        .position = ToConstSpan(dPositions),
        .gaze = ToConstSpan(dGazePoints),
        .up = ToConstSpan(dUpVectors)
    };
}

CamAttributeInfoList CameraGroupPinhole::AttributeInfo() const
{
    using enum MRayDataEnum;
    using enum AttributeOptionality;
    using enum AttributeIsArray;
    static const CamAttributeInfoList LogicList =
    {
        CamAttributeInfo("FovAndPlanes", MRayDataTypeRT(MR_VECTOR_4), IS_SCALAR, MR_MANDATORY),
        CamAttributeInfo("gaze",         MRayDataTypeRT(MR_VECTOR_3), IS_SCALAR, MR_MANDATORY),
        CamAttributeInfo("position",     MRayDataTypeRT(MR_VECTOR_3), IS_SCALAR, MR_MANDATORY),
        CamAttributeInfo("up",           MRayDataTypeRT(MR_VECTOR_3), IS_SCALAR, MR_MANDATORY)
    };
    return LogicList;
}

void CameraGroupPinhole::PushAttribute(CameraKey camKey,
                                       uint32_t attributeIndex,
                                       TransientData data,
                                       const GPUQueue& queue)
{
    auto PushData = [&]<class T>(const Span<T>&d)
    {
        GenericPushData(d, camKey.FetchIndexPortion(),
                        attributeIndex,
                        std::move(data), queue);
    };
    switch(attributeIndex)
    {
        case 0: PushData(dFovAndPlanes);    break;
        case 1: PushData(dGazePoints);      break;
        case 2: PushData(dPositions);       break;
        case 3: PushData(dUpVectors);       break;
        default:
        {
            MRAY_ERROR_LOG("{:s}: Unknown Attribute Index {:d}",
                           TypeName(), attributeIndex);
            return;
        }
    }
}

void CameraGroupPinhole::PushAttribute(CameraKey camKey,
                                       uint32_t attributeIndex,
                                       const Vector2ui& subRange,
                                       TransientData data,
                                       const GPUQueue& queue)
{
    auto PushData = [&]<class T>(const Span<T>&d)
    {
        GenericPushData(d, camKey.FetchIndexPortion(),
                        attributeIndex, subRange,
                        std::move(data), queue);
    };
    switch(attributeIndex)
    {
        case 0: PushData(dFovAndPlanes);    break;
        case 1: PushData(dGazePoints);      break;
        case 2: PushData(dPositions);       break;
        case 3: PushData(dUpVectors);       break;
        default:
        {
            MRAY_ERROR_LOG("{:s}: Unknown Attribute Index {:d}",
                           TypeName(), attributeIndex);
            return;
        }
    }
}

void CameraGroupPinhole::PushAttribute(CameraKey idStart, CameraKey idEnd,
                                       uint32_t attributeIndex,
                                       TransientData data,
                                       const GPUQueue& queue)
{
    auto PushData = [&]<class T>(const Span<T>&d)
    {
        auto idRange = Vector<2, IdInt>(idStart.FetchIndexPortion(),
                                        idEnd.FetchIndexPortion());
        GenericPushData(d, idRange,
                        attributeIndex,
                        std::move(data), queue);
    };
    switch(attributeIndex)
    {
        case 0: PushData(dFovAndPlanes);    break;
        case 1: PushData(dGazePoints);      break;
        case 2: PushData(dPositions);       break;
        case 3: PushData(dUpVectors);       break;
        default:
        {
            MRAY_ERROR_LOG("{:s}: Unknown Attribute Index {:d}",
                           TypeName(), attributeIndex);
            return;
        }
    }
}

CameraTransform CameraGroupPinhole::AcquireCameraTransform(CameraKey k) const
{
    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
    CommonKey kIndex = k.FetchIndexPortion();

    CameraTransform result;
    queue.MemcpyAsync(Span<Vector3>(&result.gazePoint, 1),
                      ToConstSpan(dGazePoints.subspan(kIndex, 1)));
    queue.MemcpyAsync(Span<Vector3>(&result.position, 1),
                      ToConstSpan(dPositions.subspan(kIndex, 1)));
    queue.MemcpyAsync(Span<Vector3>(&result.up, 1),
                      ToConstSpan(dUpVectors.subspan(kIndex, 1)));
    queue.Barrier().Wait();

    return result;
}

typename CameraGroupPinhole::DataSoA
CameraGroupPinhole::SoA() const
{
    return soa;
}