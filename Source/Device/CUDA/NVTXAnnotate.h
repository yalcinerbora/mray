#pragma once

#include <nvtx3/nvToolsExt.h>
#include <string_view>

namespace mray::cuda
{
using AnnotationHandle = void*;

class NVTXKernelName;

class NVTXAnnotate
{
    private:
    nvtxDomainHandle_t  d;
    public:
                        NVTXAnnotate(const NVTXKernelName& kernelName);
                        NVTXAnnotate(const NVTXAnnotate&) = delete;
    NVTXAnnotate&       operator=(const NVTXAnnotate&) = delete;
                        ~NVTXAnnotate();
};

class NVTXKernelName
{
    friend class            NVTXAnnotate;
    nvtxDomainHandle_t      nvtxDomain;
    nvtxEventAttributes_t   eventAttrib = {0};
    public:
                            NVTXKernelName(AnnotationHandle domain,
                                           std::string_view name);
    NVTXAnnotate            Annotate() const;
};

inline NVTXAnnotate::NVTXAnnotate(const NVTXKernelName& kernelName)
    : d(kernelName.nvtxDomain)
{
    nvtxDomainRangePushEx(d, &(kernelName.eventAttrib));
}

inline NVTXAnnotate::~NVTXAnnotate()
{
    nvtxDomainRangePop(d);
}

inline NVTXKernelName::NVTXKernelName(AnnotationHandle domain,
                                      std::string_view name)
    : nvtxDomain(static_cast<nvtxDomainHandle_t>(domain))
{
    eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    eventAttrib.message.registered = nvtxDomainRegisterStringA(nvtxDomain,
                                                               name.data());
}

inline NVTXAnnotate NVTXKernelName::Annotate() const
{
    return NVTXAnnotate(*this);
}

}