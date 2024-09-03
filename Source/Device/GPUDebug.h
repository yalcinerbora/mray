#pragma once

#include <string_view>
#include <fstream>
#include <iostream>
#include "GPUSystemForward.h"

#pragma message ("WARNING! \"GPUDebug.h\" is included, don't forget to remove when " \
                 "debugging is done.")

namespace DeviceDebug
{

using namespace std::string_view_literals;

template<class T>
void DumpGPUMemToStream(std::ostream& s,
                        Span<const T> data,
                        const GPUQueue& queue,
                        std::string_view seperator = "\n"sv)
{
    std::vector<T> hostBuffer(data.size());
    queue.MemcpyAsync(Span<T>(hostBuffer), data);
    queue.Barrier().Wait();

    for(const T& d : hostBuffer)
    {
        s << MRAY_FORMAT("{}{:s}", d, seperator);
    }
}

template<class T>
void DumpGPUMemToFile(const std::string& fName,
                      Span<const T> data,
                      const GPUQueue& queue,
                      std::string_view seperator = "\n"sv)
{
    std::ofstream file(fName);
    DumpGPUMemToStream(file, data, queue, seperator);
}

template<class T>
void DumpGPUMemToStdOut(std::string_view header,
                        Span<const T> data,
                        const GPUQueue& queue,
                        std::string_view seperator = "\n"sv)
{
    if(!header.empty()) MRAY_DEBUG_LOG("{}", header);
    DumpGPUMemToStream(std::cout, data, queue, seperator);
    if(!header.empty()) MRAY_DEBUG_LOG("-------------");
}

}