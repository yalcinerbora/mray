#include "CamerasDefault.h"
#include "Core/TypeNameGenerators.h"


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
    auto [fnp, g, p, u] = GenericCommit<Vector4, Vector3,
                                        Vector3, Vector3>({0, 0, 0, 0});
    dFovAndPlanes = fnp;
    dGazePoints = g;
    dPositions = p;
    dUpVectors = u;

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
    using enum PrimitiveAttributeLogic;
    using enum AttributeOptionality;
    using enum AttributeIsArray;
    static const CamAttributeInfoList LogicList =
    {
        CamAttributeInfo("FovAndPlanes", MRayDataType<MR_VECTOR_4>(), IS_SCALAR, MR_MANDATORY),
        CamAttributeInfo("gaze",         MRayDataType<MR_VECTOR_3>(), IS_SCALAR, MR_MANDATORY),
        CamAttributeInfo("position",     MRayDataType<MR_VECTOR_3>(), IS_SCALAR, MR_MANDATORY),
        CamAttributeInfo("up",           MRayDataType<MR_VECTOR_3>(), IS_SCALAR, MR_MANDATORY)
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

CameraTransform CameraGroupPinhole::AcquireCameraTransform(CameraKey) const
{
    throw MRayError("TODO: Implement");
}

typename CameraGroupPinhole::DataSoA
CameraGroupPinhole::SoA() const
{
    return soa;
}