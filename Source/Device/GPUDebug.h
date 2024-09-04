#pragma once

#include <string_view>
#include <fstream>
#include <iostream>
#include "GPUSystemForward.h"

#pragma message ("WARNING! \"GPUDebug.h\" is included, don't forget to remove when " \
                 "debugging is done.")

namespace DeviceDebug
{

    enum WriteMode
    {
        DEFAULT,
        HEXEDECIMAL,
        BINARY
    };

using namespace std::string_view_literals;

template<WriteMode MODE = DEFAULT, class T>
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
        // Mode is specifically compile time (template)
        // paramater, because fmt validates in compile time
        // some types may not have hex formatting which will not
        // make this code to compile.
        //
        // So we do  constexpr if to eliminate that code generation
        // If user wants hex, he/she can specify the param
        if constexpr(MODE == DEFAULT)
            s << MRAY_FORMAT("{}{:s}", d, seperator);
        else if constexpr(MODE == HEXEDECIMAL)
            s << MRAY_FORMAT("{:x}{:s}", d, seperator);
        else if constexpr(MODE == BINARY)
            s << MRAY_FORMAT("{:b}{:s}", d, seperator);
        else static_assert(MODE < BINARY, "Unkown print mode!");
    }
}

template<WriteMode MODE = DEFAULT, class T>
void DumpGPUMemToFile(const std::string& fName,
                      Span<const T> data,
                      const GPUQueue& queue,
                      std::string_view seperator = "\n"sv)
{
    std::ofstream file(fName);
    DumpGPUMemToStream<MODE>(file, data, queue, seperator);
}

template<WriteMode MODE = DEFAULT, class T>
void DumpGPUMemToStdOut(std::string_view header,
                        Span<const T> data,
                        const GPUQueue& queue,
                        std::string_view seperator = "\n"sv)
{
    if(!header.empty()) MRAY_DEBUG_LOG("{}", header);
    DumpGPUMemToStream<MODE>(std::cout, data, queue, seperator);
    if(!header.empty()) MRAY_LOG("-------------");
}

}