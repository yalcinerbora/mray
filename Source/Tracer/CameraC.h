#pragma once

class GenericGroupCameraT : public GenericGroupT<CameraKey, CamAttributeInfo>
{
    private:

    public:
    GenericGroupCameraT(uint32_t groupId,
                        const GPUSystem& sys);

    virtual CameraTransform AcquireCameraTransform(CameraKey) const = 0;
};

using CameraGroupPtr      = std::unique_ptr<GenericGroupCameraT>;

template <class Child>
class GenericGroupCamera : public GenericGroupCameraT
{
    public:
                     GenericGroupCamera(uint32_t groupId,
                                        const GPUSystem& sys);
    std::string_view Name() const override;
};

inline
GenericGroupCameraT::GenericGroupCameraT(uint32_t groupId,
                                         const GPUSystem& sys)
    :GenericGroupT<CameraKey, CamAttributeInfo>(groupId, sys)
{}

template <class C>
GenericGroupCamera<C>::GenericGroupCamera(uint32_t groupId,
                                          const GPUSystem& sys)
    : GenericGroupCameraT(groupId, sys)
{}

template <class C>
std::string_view GenericGroupCamera<C>::Name() const
{
    return C::TypeName();
}