#pragma once

#include "EntryPoint.h"

class MeshLoaderGFG : public MeshLoaderI
{
    public:
    static constexpr std::string_view   Tag = "gfg";

    private:

    public:
                                    MeshLoaderGFG();
    std::unique_ptr<MeshFileI>      OpenFile(std::string& filePath) override;
};