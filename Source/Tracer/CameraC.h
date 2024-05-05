#pragma once

using GenericGroupCameraT = GenericGroupT<CameraKey, CamAttributeInfo>;
using CameraGroupPtr      = std::unique_ptr<GenericGroupCameraT>;

template <class Child>
class GenericGroupCamera : public GenericGroupCameraT
{
    public:
                     GenericGroupCamera(uint32_t groupId,
                                        const GPUSystem& sys);
    std::string_view Name() const override;
};

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