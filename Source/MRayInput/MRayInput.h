#pragma once

#include <memory_resource>

#include "Core/System.h"
#include "Core/Types.h"

#ifdef MRAY_INPUT_DLL_EXPORT
    #define MRAY_INPUT_DLL MRAY_DLL_EXPORT
#else
    #define MRAY_INPUT_DLL MRAY_DLL_IMPORT
#endif

namespace MRayInputDetail
{
    class MRayInput;
}

MRAY_INPUT_DLL extern void* MRayInputIssueBufferForDestruction(MRayInputDetail::MRayInput buffer);
MRAY_INPUT_DLL extern void MRayInputDestroyCallback(void* ptr);

namespace MRayInputDetail
{

template <class T>
concept StringC = std::disjunction_v
<
    std::is_same<T, std::string>,
    std::is_same<T, std::string_view>
>;

class MRayInput
{
    friend void ::MRayInputDestroyCallback(void*);
    MRayInput() = default;

    private:
    size_t      typeHash;
    Span<Byte>  ownedMem;
    size_t      usedBytes;
    size_t      alignment;

    public:
    template<ImplicitLifetimeC T>
                MRayInput(std::in_place_type_t<T>, size_t count);
                MRayInput(const MRayInput&) = delete;
                MRayInput(MRayInput&&);
    MRayInput&  operator=(const MRayInput&) = delete;
    MRayInput&  operator=(MRayInput&&);
                ~MRayInput();

    template<ImplicitLifetimeC T>
    void            Push(Span<const T>);
    template<ImplicitLifetimeC T>
    Span<const T>   AccessAs() const;
    template<ImplicitLifetimeC T>
    Span<T>         AccessAs();

    // =========================== //
    //    String Specialization    //
    // =========================== //
    template<StringC T>
                        MRayInput(std::in_place_type_t<T>,
                                  size_t charCount);
    template<StringC T>
    void                Push(Span<const T, 1>);
    std::string_view    AccessAsString() const;
    std::string_view    AccessAsString();
};

// I could not utilize, linked-list here
// Rolling a simple free-list impl
struct FreeListNode
{
    MRayInput input;
    FreeListNode* prev = nullptr;
    FreeListNode* next = nullptr;
};

class FreeList
{
    using MonoBuffer = std::pmr::monotonic_buffer_resource;

    private:
    MonoBuffer      monoBuffer;
    FreeListNode*   freeHead;
    std::mutex      m;

    public:
                    FreeList();
    FreeListNode*   GetALocation(MRayInput buffer);
    void            GiveTheLocation(FreeListNode* node);

};

using PoolMemResource = std::pmr::synchronized_pool_resource;

MRAY_INPUT_DLL extern PoolMemResource   mainR;
MRAY_INPUT_DLL extern FreeList          freeList;

}

#include "MRayInput.hpp"

// Only MRayInput is exposed to the user
using MRayInput = MRayInputDetail::MRayInput;

template<ImplicitLifetimeC T>
inline Span<T> ToSpan(const MRayInput& v)
{
    return v.AccessAs<T>();
}