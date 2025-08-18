#pragma once

#include <memory_resource>
#include <mutex>
#include <cstring>

#include "Core/Definitions.h"
#include "Core/System.h"
#include "Core/Types.h"
#include "Core/Span.h"

#ifdef MRAY_TRANSIENT_POOL_SHARED_EXPORT
    #define MRAY_TRANSIENT_POOL_ENTRYPOINT MRAY_DLL_EXPORT
#else
    #define MRAY_TRANSIENT_POOL_ENTRYPOINT MRAY_DLL_IMPORT
#endif

namespace TransientPoolDetail
{
    class TransientData;
}

MRAY_TRANSIENT_POOL_ENTRYPOINT extern void* TransientPoolIssueBufferForDestruction(TransientPoolDetail::TransientData buffer);
MRAY_TRANSIENT_POOL_ENTRYPOINT extern void TransientPoolDestroyCallback(void* ptr);

namespace TransientPoolDetail
{

// TODO: Make this Preprocessor def
static constexpr bool EnableTypeCheck = true;

template <class T>
concept StringC = std::disjunction_v
<
    std::is_same<T, std::string>,
    std::is_same<T, std::string_view>
>;

class TransientData
{
    friend void     ::TransientPoolDestroyCallback(void*);
                    TransientData() = default;

    private:
    size_t          typeHash;
    Span<Byte>      ownedMem;
    size_t          usedBytes;
    size_t          alignment;

    public:
    template<ImplicitLifetimeC T>
                    TransientData(std::in_place_type_t<T>, size_t count);
                    TransientData(const TransientData&) = delete;
                    TransientData(TransientData&&);
    TransientData&  operator=(const TransientData&) = delete;
    TransientData&  operator=(TransientData&&);
                    ~TransientData();

    bool            IsEmpty() const;
    bool            IsFull() const;
    void            ReserveAll();
    template<ImplicitLifetimeC T>
    void            Push(Span<const T>);
    template<ImplicitLifetimeC T>
    Span<const T>   AccessAs() const;
    template<ImplicitLifetimeC T>
    Span<T>         AccessAs();
    template<ImplicitLifetimeC T>
    size_t          Size() const;

    // =========================== //
    //    String Specialization    //
    // =========================== //
    template<StringC T>
    void                Push(Span<const T, 1>);

    std::string_view    AccessAsString() const;
    Span<char>          AccessAsString();
};

// I could not utilize, linked-list here
// Rolling a simple free-list impl
struct FreeListNode
{
    TransientData   input;
    FreeListNode*   prev = nullptr;
    FreeListNode*   next = nullptr;
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
    FreeListNode*   GetALocation(TransientData buffer);
    void            GiveTheLocation(FreeListNode* node);

};

using PoolMemResource = std::pmr::synchronized_pool_resource;

MRAY_TRANSIENT_POOL_ENTRYPOINT extern PoolMemResource   mainR;
MRAY_TRANSIENT_POOL_ENTRYPOINT extern FreeList          freeList;

}

#include "TransientPool.hpp"

// Only TransientData is exposed to the user
using TransientData = TransientPoolDetail::TransientData;

template<ImplicitLifetimeC T>
inline Span<T> ToSpan(const TransientData& v)
{
    return v.AccessAs<T>();
}