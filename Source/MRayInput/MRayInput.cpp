
#include <cassert>
#include <memory>

#include "MRayInput.h"
#include "Core/MemAlloc.h"

// TODO: I really did not understand the parameters from cppreference.
// If I understand correctly, there will be multiple chunks, probably starts with
// word size (4 or 8 bytes)
// Chunk0: (4 Byte allocations) 32 * ??? Blocks (Dunno what ??? is)
// Chunk1: (8 Byte allocations)                 ''
// Chunk2: (16 Byte allocations)                ''
// ...
// ChunkN-1: (1024 Byte allocations)
// ChunkN  : (2048 Byte allocations)   <- (Dunno if inclusive or exclusive)
// Larger allocations will use operator new directly
//
// In our case, ChunkN is 32_MiB
//
// TODO: Check if this is wrong and adjust accordingly
static constexpr auto POOL_OPTIONS = std::pmr::pool_options
{
    .max_blocks_per_chunk = 64,
    .largest_required_pool_block = 32_MiB
};

namespace MRayInputDetail
{

// Put constructor here as well I dunno "new_delete_resource"
// is really process-wide (MSVC it maybe DLL-wide just to be sure)
FreeList::FreeList()
    : monoBuffer(MonoBuffer(32_KiB, std::pmr::new_delete_resource()))
    , freeHead(nullptr)
{}


// TODO: Destruction order here is important?
// "freeList" holds pointers to the "PoolMemResource"
// so in a translation unit the init order is top down?
// Then destruction order is bottom-up?
// Guarantee that this is true
MRAY_INPUT_ENTRYPOINT PoolMemResource mainR = PoolMemResource(POOL_OPTIONS, std::pmr::new_delete_resource());
MRAY_INPUT_ENTRYPOINT FreeList freeList;

}

MRAY_INPUT_ENTRYPOINT void* MRayInputIssueBufferForDestruction(MRayInputDetail::MRayInput buffer)
{
    using namespace MRayInputDetail;
    return reinterpret_cast<void*>(freeList.GetALocation(std::move(buffer)));
}

MRAY_INPUT_ENTRYPOINT void MRayInputDestroyCallback(void* ptr)
{
    using namespace MRayInputDetail;
    FreeListNode* nodePtr = std::launder(reinterpret_cast<FreeListNode*>(ptr));
    // Now before entering to the lock destroy the Input buffer
    // If we do this inside the lock there is a potential deadlock
    // (maybe?) since MRayInput uses synchronized_pool_resource
    // and it may have a mutex
    nodePtr->input = MRayInput();
    freeList.GiveTheLocation(nodePtr);
}