#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "TracerTypes.h"

#include "GenericGroup.h"

// TODO: Move camera transform somewhere proper
#include "Common/AnalyticStructs.h"

struct CameraRayOutput
{
    Ray             ray;
    Vector2         tMinMax;
    ImageCoordinate imgCoords;
    RayCone         rayCone;
};
using CameraRaySample = SampleT<CameraRayOutput>;

template<class CameraType>
concept CameraC = requires(CameraType c,
                           RNGDispenser& rng)
{
    typename CameraType::DataSoA;

    // API
    CameraType(typename CameraType::DataSoA{}, CameraKey{});

    // RN Count
    CameraType::SampleRayRNList;
    requires std::is_same_v<decltype(CameraType::SampleRayRNList), const RNRequestList>;

    {c.SampleRay(Vector2ui{}, Vector2ui{}, rng)
    } -> std::same_as<CameraRaySample>;
    {c.EvaluateRay(Vector2ui{}, Vector2ui{}, Vector2{}, Vector2{})
    }->std::same_as<CameraRaySample>;
    {c.ReconstructRay(ImageCoordinate{}, Vector2ui{})
    }->std::same_as<CameraRayOutput>;
    {c.PdfRay(Ray{})} -> std::same_as<Float>;
    {c.CanBeSampled()} -> std::same_as<bool>;
    {c.GetCameraTransform()} -> std::same_as<CameraTransform>;
    {c.OverrideTransform(CameraTransform{})} -> std::same_as<void>;
    {c.GenerateSubCamera(Vector2ui{}, Vector2ui{})} -> std::same_as<CameraType>;
    {c.GetCameraPosition()} -> std::same_as<Vector3>;

    // Type traits
    requires std::is_trivially_copyable_v<CameraType>;
    requires std::is_trivially_destructible_v<CameraType>;
    requires std::is_move_assignable_v<CameraType>;
    requires std::is_move_constructible_v<CameraType>;
};

template<class CGType>
concept CameraGroupC = requires(CGType cg)
{
    // Internal Camera type that satisfies its concept
    typename CGType::Camera;
    requires CameraC<typename CGType::Camera>;
    //
    CGType::SampleRayRNList;
    requires std::is_same_v<decltype(CGType::SampleRayRNList), const RNRequestList>;

    // SoA fashion camera data. This will be used to access internal
    // of the camera with a given an index
    typename CGType::DataSoA;
    requires std::is_same_v<typename CGType::DataSoA,
                            typename CGType::Camera::DataSoA>;

    // Acquire SoA struct of this primitive group
    {cg.SoA()} -> std::same_as<typename CGType::DataSoA>;
};

class GenericGroupCameraT : public GenericGroupT<CameraKey, CamAttributeInfo>
{
    private:

    public:
    GenericGroupCameraT(uint32_t groupId,
                        const GPUSystem& sys,
                        size_t allocationGranularity = 2_MiB,
                        size_t initialReservationSize = 4_MiB);

    virtual CameraTransform AcquireCameraTransform(CameraKey) const = 0;
};

using CameraGroupPtr      = std::unique_ptr<GenericGroupCameraT>;

template <class Child>
class GenericGroupCamera : public GenericGroupCameraT
{
    public:
                     GenericGroupCamera(uint32_t groupId,
                                        const GPUSystem& sys,
                                        size_t allocationGranularity = 2_MiB,
                                        size_t initialReservationSize = 4_MiB);
    std::string_view Name() const override;
};

namespace CameraDetail
{
    class CameraNull
    {
        public:
        using DataSoA = EmptyType;
        static constexpr RNRequestList SampleRayRNList = GenRNRequestList<0>();

        private:
        public:
        constexpr       CameraNull() = default;
        MR_PF_DECL_V    CameraNull(const DataSoA&, CameraKey);
        // Ray Sampling
        MR_PF_DECL
        CameraRaySample SampleRay(// Input
                                  const Vector2ui& generationIndex,
                                  const Vector2ui& stratumCount,
                                  // I-O
                                  RNGDispenser&) const;
        MR_PF_DECL
        CameraRaySample EvaluateRay(const Vector2ui& generationIndex,
                                    const Vector2ui& stratumCount,
                                    const Vector2& stratumOffset,
                                    const Vector2& stratumRange) const;

        MR_PF_DECL
        Float           PdfRay(const Ray&) const;

        MR_PF_DECL
        CameraRayOutput ReconstructRay(const ImageCoordinate&,
                                       const Vector2ui& stratumCount) const;
        // Misc
        MR_PF_DECL
        bool            CanBeSampled() const;
        MR_PF_DECL
        CameraTransform GetCameraTransform() const;
        MR_PF_DECL_V
        void            OverrideTransform(const CameraTransform&);
        MR_PF_DECL
        CameraNull      GenerateSubCamera(const Vector2ui& regionId,
                                          const Vector2ui& regionCount) const;
        MR_PF_DECL
        Vector3         GetCameraPosition() const;
    };
}

class CameraGroupNull : public GenericGroupCamera<CameraGroupNull>
{
    public:
    using DataSoA   = EmptyType;
    using Camera    = CameraDetail::CameraNull;
    static constexpr RNRequestList SampleRayRNList = Camera::SampleRayRNList;

    private:
    public:
    static std::string_view TypeName();
    //
                            CameraGroupNull(uint32_t groupId,
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


namespace CameraDetail
{

MR_PF_DEF_V
CameraNull::CameraNull(const DataSoA&, CameraKey)
{}

MR_PF_DEF
CameraRaySample
CameraNull::SampleRay(// Input
                      const Vector2ui&,
                      const Vector2ui&,
                      // I-O
                      RNGDispenser&) const
{
    return CameraRaySample
    {
        .value =
        {
            .ray = Ray(Vector3::Zero(), Vector3::Zero()),
            .tMinMax = Vector2::Zero(),
            .imgCoords =
            {
                .pixelIndex = Vector2us::Zero(),
                .offset     = SNorm2x16(0, 0)
            },
            .rayCone = RayCone
            {
                .aperture = Float(0),
                .width    = Float(0)
            }
        },
        .pdf = Float(1.0)
    };
}

MR_PF_DEF
CameraRaySample
CameraNull::EvaluateRay(const Vector2ui&,
                        const Vector2ui&,
                        const Vector2&,
                        const Vector2&) const
{
    return CameraRaySample
    {
        .value =
        {
            .ray = Ray(Vector3::Zero(), Vector3::Zero()),
            .tMinMax = Vector2::Zero(),
            .imgCoords =
            {
                .pixelIndex = Vector2us::Zero(),
                .offset     = SNorm2x16(0, 0)
            },
            .rayCone = RayCone
            {
                .aperture = Float(0),
                .width    = Float(0)
            }
        },
        .pdf = Float(1.0)
    };
}

MR_PF_DEF
Float CameraNull::PdfRay(const Ray&) const
{
    return Float(0.0);
}

MR_PF_DEF
CameraRayOutput
CameraNull::ReconstructRay(const ImageCoordinate&,
                           const Vector2ui&) const
{
    return CameraRayOutput
    {
        .ray = Ray(Vector3::Zero(), Vector3::Zero()),
        .tMinMax = Vector2::Zero(),
        .imgCoords =
        {
            .pixelIndex = Vector2us::Zero(),
            .offset = SNorm2x16(0, 0)
        },
        .rayCone = RayCone
        {
            .aperture = Float(0),
            .width = Float(0)
        }
    };
}

MR_PF_DEF
bool CameraNull::CanBeSampled() const
{
    return false;
}

MR_PF_DEF
CameraTransform CameraNull::GetCameraTransform() const
{
    return CameraTransform
    {
        .position   = Vector3::Zero(),
        .gazePoint  = Vector3::Zero(),
        .up         = Vector3::Zero(),
    };
}

MR_PF_DEF_V
void CameraNull::OverrideTransform(const CameraTransform&)
{}

MR_PF_DEF
CameraNull CameraNull::GenerateSubCamera(const Vector2ui&,
                                         const Vector2ui&) const
{
    return CameraNull{};
}

MR_PF_DEF
Vector3 CameraNull::GetCameraPosition() const
{
    return Vector3::Zero();
}

}

inline
GenericGroupCameraT::GenericGroupCameraT(uint32_t groupId,
                                         const GPUSystem& sys,
                                         size_t allocationGranularity,
                                         size_t initialReservationSize)
    :GenericGroupT<CameraKey, CamAttributeInfo>(groupId, sys,
                                                allocationGranularity,
                                                initialReservationSize)
{}

template <class C>
GenericGroupCamera<C>::GenericGroupCamera(uint32_t groupId,
                                          const GPUSystem& sys,
                                          size_t allocationGranularity,
                                          size_t initialReservationSize)
    : GenericGroupCameraT(groupId, sys,
                          allocationGranularity,
                          initialReservationSize)
{}

template <class C>
std::string_view GenericGroupCamera<C>::Name() const
{
    return C::TypeName();
}