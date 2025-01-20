#pragma once

#include "Core/Definitions.h"
#include "Core/Types.h"
#include "TracerTypes.h"

#include "GenericGroup.h"

struct RaySampleT
{
    Ray             ray;
    Vector2         tMinMax;
    ImageCoordinate imgCoords;
    RayCone         rayCone;
};
using RaySample = SampleT<RaySampleT>;

template<class CameraType>
concept CameraC = requires(CameraType c,
                           RNGDispenser& rng)
{
    typename CameraType::DataSoA;

    // API
    CameraType(typename CameraType::DataSoA{}, CameraKey{});

    // RN Count
    CameraType::SampleRayRNCount;
    requires std::is_same_v<decltype(CameraType::SampleRayRNCount), const uint32_t>;

    {c.SampleRay(Vector2ui{}, Vector2ui{}, rng)
    } -> std::same_as<RaySample>;
    {c.EvaluateRay(Vector2ui{}, Vector2ui{}, Vector2{}, Vector2{})
    }->std::same_as<RaySample>;
    {c.PdfRay(Ray{})} -> std::same_as<Float>;
    {c.CanBeSampled()} -> std::same_as<bool>;
    {c.GetCameraTransform()} -> std::same_as<CameraTransform>;
    {c.OverrideTransform(CameraTransform{})} -> std::same_as<void>;
    {c.GenerateSubCamera(Vector2ui{}, Vector2ui{})} -> std::same_as<CameraType>;

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
    CGType::SampleRayRNCount;

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
                        size_t initialReservartionSize = 4_MiB);

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
                                        size_t initialReservartionSize = 4_MiB);
    std::string_view Name() const override;
};

inline
GenericGroupCameraT::GenericGroupCameraT(uint32_t groupId,
                                         const GPUSystem& sys,
                                         size_t allocationGranularity,
                                         size_t initialReservartionSize)
    :GenericGroupT<CameraKey, CamAttributeInfo>(groupId, sys,
                                                allocationGranularity,
                                                initialReservartionSize)
{}

template <class C>
GenericGroupCamera<C>::GenericGroupCamera(uint32_t groupId,
                                          const GPUSystem& sys,
                                          size_t allocationGranularity,
                                          size_t initialReservartionSize)
    : GenericGroupCameraT(groupId, sys,
                          allocationGranularity,
                          initialReservartionSize)
{}

template <class C>
std::string_view GenericGroupCamera<C>::Name() const
{
    return C::TypeName();
}