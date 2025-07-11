#pragma once

#include <cstdint>

#include "Core/DataStructures.h"

#include "Device/GPUTexture.h"

#include "StreamingTextureView.h"

// ============================================== //
//       THIS SYSTEM IS IN CONCEPTION PHASE		  //
//  CODE IS ARBITRARY AND MAY NOT WORK AT ALL     //
// ============================================== //
// Texture Streaming system of MRay
//
// The system does not touch the virtual texture memory system
// to make it compatible between future APIs (HIP and SCYL).
// However, we utilize "cudaArrayDeferredMapping" to alias memory
// between textures.
//
// The design is similar to the Unreal's MegaTextures system as far
// as i understand from the documentation. I did not look at the source
// for obvious reasons.
//
// Other approaches include the NVIDIA's OptixToolkit(OTK) Demand texture
// implementation (https://github.com/NVIDIA/optix-toolkit/tree/master/DemandLoading/DemandLoading)
// which fully utilizes the hardware's sparse texture / virtual memory system
//
// ==================================================
// Design Principles
// ==================================================
//  - This should work for every texture type MRay supports (including
//    non-mipmapped types (i.e., png), compressed types (i.e., jpeg)
//    and block-compressed types (i.e, dds)
//
//  - No disk caching to improve performance of subsequent runs
//    of the same scene. Every run must start on fresh. Ideally
//    user will have all of its textures as mipmapped. This will be the
//    somewhat ideal on disk cache anyways. We can provide an offline tool
//    to convert the textures on a ideal tiled/duplicated format later.
//    The control must be on the user.
//
//  - Exact texture request must be fulfilled. MRay; in heart, is
//    research-oriented renderer. So we do not cut corners such as
//    persistently storing the last mip (1x1) and returning that
//    when virtual page is not resident. User's texture tap
//    must be fulfilled as precisely as possible.
//
//  - This also means we do not generalize texture formats to a single
//    format.
//
// ==================================================
// Design Details
// ==================================================
// GPU Side:
//  - We have N large buffers backing multiple textures.
//
//  - There are two type of buffers (1:1 and 2:1 aspect ratio).
//
//  - Depending on the user parameters and usage N is split between
//    these two types.
//
//  - These two buffers split into 64k-sized tiles:
//  - For 1:1 tiles these correspond to:
//     -- R_8        (256x256)
//     -- RGBA_8     (128x128)
//     -- RG_16      (128x128)
//     -- R_32F      (128x128)
//     -- RGBA_32F   (64x64)
//     -- BC2        (256x256) (64x64 Blocks)
//     -- BC3           "           "
//     -- BC3           "           "
//     -- BC5           "           "
//     -- BC6           "           "
//     -- BC7           "           "
//
//  - For 2:1 tiles these correspond to:
//     -- RG_8       (256 x 128)
//     -- R_16		 (256 x 128)
//     -- RGBA_16    (128x64)
//     -- RG_32F     (128x64)
//     -- BC1		 (512x128) (128x64 Blocks)
//     -- BC4           "           "
//
//  - Since we store different virtual textures on the same virtual texture
//    inter-texture filtering can be a problem, so we add +-2 pixel padding to
//    each texture. It is implied above the tile sizes. So virtual tile size is
//    4 pixel less than the stated above.
//
//  - This means we can't fully enable anisotropic filtering(AF). 2-pixel border
//    means we can at most enable 4x AF (I do not know how the AF system works
//    on the HW so this may be wrong).
//
//  - To utilize inter-mip filtering each of the texture's above also have an
//    extra mip that will next virtual mip that should be sampled.
//
//  - All in all, virtual tiles consists of (N x N) base tile (if 2:1 aspect N x N/2)
//    and (N/2 x N/2) mip tile (with padding). This is the streamable unit
//    of the system. This prevents extra lookup of the virtual to physical
//    translation system (a hash table) but increases the memory consumption
//    %50 percent (since this is a streaming system, higher memory consumption
//    means less memory utilization per texture). This is not final, we may
//    fall back to two-tap version after benchmarks. Please check
//    StreamingTextureView class for the implementation.
//
//  - The system by nature do not store mips that has smaller than (N/2 x N/2).
//    Mip filtering is clamped on that limit. Aliasing should be minimal for
//    these cases since this system is used in offline path tracing. Monte Carlo
//    should filter the aliasing.
//
//  - Textures smaller than (N x N) can be stored but will waste memory.
//    NVIDIA allocs minimally allocs 64k chunks for its textures anyway so
//    at best we do not get full HW filtering of mips.
//
//  - Each virtual texture has a footprint on the GPU as well. One obvious
//    one is the parameters (size, edge-resolve, channel info etc.) another
//    one is the page requested bitset. For each texture's virtual tiles,
//    we store a single bit. For extreme case scenario (8k RGBA_32F texture),
//    this footprint will store 137 * 137 * 1.5 = ~28,154 bits (virtual tile
//    size of  RGBA_32F tex is 60x60 due to 2pix padding, so 8192 / 60 = 136.5)
//    which is around ~3.5 KiB. So on average each texture will have somewhat
//    negligible memory footprint. For more common case (8k RGBA_8 texture)
//    it is 0.8KiB.
//  - This request bitset is set by the texture taps automatically.
//    Texture will return optional<T> where the pixel type is T
//    (float, Vector2 etc.). It is shaders or other systems' responsibility
//    to act accordingly.
//  - After these bits are set, user will call a function on the CPU side
//    these bits are transferred back to the CPU processed and new virtual
//    tiles are processed and send back to the GPU.
//
// CPU Side:
//  - Since we need to support non-mippmapped types, we have an aggressively
//    allocated CPU side tile pool (probably couple of GiB) since we do not
//    use the CPU memory much. When a tile is requested we calculate mipmaps
//    on the GPU and store it on the physical tiles. When the tile is evicted
//    from the GPU it will cached on the CPU. When CPU cache is full it will
//    be discarded etc.
//
//
// Potential edge cases:
// Case1: Material taps are too sparse and request exceeds the maximum physical
//        tile size. On average MRay sends 2M rays each hitting different
//        tiles (and materials will have multiple textures) is not unlikely.
//  - All current physical pages will be evicted, and implementation defined
//    order of new requests will fill the physical tiles.
//  - GPU will tap the textures and use them, unissued taps will set the buffer
//    again.
//  - This loop is continued until all texture requests are finalized.
//  - This will reduce the performance but for massively parallel hardware
//    such as GPUs it is unavoidable.
//
//

//// Page Allocator
//// "Page" is not the OS/HW page but a exactly same sized allocations
////
//// It supports
//class PageAllocator
//{
//	enum class Handle : uint64_t {};
//
//	struct FreeListNode
//	{
//		FreeListNode*	next;
//		FreeListNode*	prev;
//		std::byte*		memory;
//	};
//	using MonoAllocator = std::pmr::monotonic_buffer_resource;
//	using PoolAllocator = std::pmr::unsynchronized_pool_resource;
//
//	private:
//	//
//	uint32_t		pageSize;
//	uint32_t		pageAlignment;
//	uint32_t		curPageCount;
//	uint32_t		maxPageCount;
//	FreeListNode*	freeList;
//	MonoAllocator	baseAlloc;
//	PoolAllocator	poolAlloc;
//
//	protected:
//	public:
//	// Constructors & Destructor
//					PageAllocator(uint32_t pageSize, uint32_t maxPageCount);
//					PageAllocator(const PageAllocator&) = delete;
//					PageAllocator(PageAllocator&&) = default;
//	PageAllocator&	operator=(const PageAllocator&) = delete;
//	PageAllocator&	operator=(PageAllocator&&) = default;
//					~PageAllocator() = default;
//
//	std::byte* AllocPage();
//			//AllocPages(uint32_t pageCount);
//	void RemovePage(uint32_t pageId);
//	void RemovePages(std::vector<uint32_t> pageIds);
//};
//
//inline PageAllocator::PageAllocator(uint32_t pageSize,
//									uint32_t pageAlignment,
//									uint32_t maxPageCount)
//	: pageSize(pageSize)
//	, curPageCount(0)
//	, maxPageCount(maxPageCount)
//	, freeList(nullptr)
//	, poolAlloc(std::pmr::pool_options{.largest_required_pool_block = pageSize},
//				&baseAlloc)
//{}
//
//std::byte* PageAllocator::AllocPage()
//{
//	Handle handle = Handle(0);
//	std::byte* mem = nullptr;
//
//	// Full mem edge case
//	if(curPageCount == maxPageCount)
//		return nullptr;
//
//	if(freeList)
//	{
//		FreeListNode* flNode = freeList;
//		freeList = freeList->next;
//		mem = freeList->memory;
//		poolAlloc.deallocate(flNode, sizeof(FreeListNode), alignof(FreeListNode));
//
//	}
//		return
//
//}
//
//
struct StreamingTexturePack
{
	template <class T>
	using TextureList = StaticVector<Texture<2, T>, StreamingTexParams::MaxPhysicalTextureCount>;

	private:
	// Aliases - Pair 1 - Square tiles
	// These textures will have different sizes
	// 16k x 16k (256 x 256)
	TextureList<uint8_t>	texList_R8_UNORM;
	TextureList<int8_t>		texList_R8_SNORM;
	// 8k x 8k (128 x 128)
	TextureList<Vector4uc>	texList_RGBA8_UNORM;
	TextureList<Vector4c>	texList_RGBA8_SNORM;
	// 8k x 8k (128 x 128)
	TextureList<Vector2us>	texList_RG16_UNORM;
	TextureList<Vector2s>	texList_RG16_SNORM;
	// 8k x 8k (128 x 128)
	TextureList<float>		texList_R32F;
	// 4k x 4k (64 x 64)
	TextureList<Vector4f>	texList_RGBA32F;
	// 16k x 16k (256 x 256) (64 x 64 Blocks)
	TextureList<PixelBC2>	texList_BC2;
	TextureList<PixelBC3>	texList_BC3;
	TextureList<PixelBC5U>	texList_BC5U;
	TextureList<PixelBC5S>	texList_BC5S;
	TextureList<PixelBC6U>	texList_BC6U;
	TextureList<PixelBC6S>	texList_BC6S;
	TextureList<PixelBC7>	texList_BC7;
	//
	// Aliases - Pair 2 - Rectangle Tiles
	// 8k x 8k (256 x 128)
	TextureList<Vector2uc>	texList_RG8_UNORM;
	TextureList<Vector2c>	texList_RG8_SNORM;
	// 16k x 8k (256 x 128)
	TextureList<uint16_t>	texList_R16_UNORM;
	TextureList<int16_t>	texList_R16_SNORM;
	// 8k x 4k (128 x 64)
	TextureList<Vector4us>	texList_RGBA16_UNORM;
	TextureList<Vector4s>	texList_RGBA16_SNORM;
	// 8k x 4k (128 x 64)
	TextureList<Vector2f>	texList_RG32F;
	// 32k x 16k (512 x 128) (128 x 64 Blocks)
	TextureList<PixelBC1>	texList_BC1;
	TextureList<PixelBC4U>	texList_BC4U;
	TextureList<PixelBC4S>	texList_BC4S;
};

struct StreamingTexture
{
	std::string				absoluteFilePath;
	MRayTextureParameters	texParams;
};

class StreamingTextureCache
{
	private:
	StreamingTexturePack concreteTextures;

	//// Aliases - Pair 1 - Square tiles
	//// These textures will have different sizes
	//// 16k x 16k (256 x 256)
	//TextureList<uint8_t>	texList_R8_UNORM;
	//TextureList<int8_t>		texList_R8_SNORM;
	//// 8k x 8k (128 x 128)
	//TextureList<Vector4uc>	texList_RGBA8_UNORM;
	//TextureList<Vector4c>	texList_RGBA8_SNORM;
	//// 8k x 8k (128 x 128)
	//TextureList<Vector2us>	texList_RG16_UNORM;
	//TextureList<Vector2s>	texList_RG16_SNORM;
	//// 8k x 8k (128 x 128)
	//TextureList<float>		texList_R32F;
	//// 4k x 4k (64 x 64)
	//TextureList<Vector4f>	texList_RGBA32F;
	//// 16k x 16k (256 x 256) (64 x 64 Blocks)
	//TextureList<PixelBC2>	texList_BC2;
	//TextureList<PixelBC3>	texList_BC3;
	//TextureList<PixelBC5U>	texList_BC5U;
	//TextureList<PixelBC5S>	texList_BC5S;
	//TextureList<PixelBC6U>	texList_BC6U;
	//TextureList<PixelBC6S>	texList_BC6S;
	//TextureList<PixelBC7>	texList_BC7;
	////
	//// Aliases - Pair 2 - Rectangle Tiles
	//// 8k x 8k (256 x 128)
	//TextureList<Vector2uc>	texList_RG8_UNORM;
	//TextureList<Vector2c>	texList_RG8_SNORM;
	//// 16k x 8k (256 x 128)
	//TextureList<uint16_t>	texList_R16_UNORM;
	//TextureList<int16_t>	texList_R16_SNORM;
	//// 8k x 4k (128 x 64)
	//TextureList<Vector4us>	texList_RGBA16_UNORM;
	//TextureList<Vector4s>	texList_RGBA16_SNORM;
	//// 8k x 4k (128 x 64)
	//TextureList<Vector2f>	texList_RG32F;
	//// 32k x 16k (512 x 128) (128 x 64 Blocks)
	//TextureList<PixelBC1>	texList_BC1;
	//TextureList<PixelBC4U>	texList_BC4U;
	//TextureList<PixelBC4S>	texList_BC4S;
	// All of the tile sizes are fixed and is 64k
	//
	// Aliasing enables easier load-balancing
	// since we need to keep track of two values
	//
	// Currently only CUDA supports aliasing
	// (even though it is not documented but
	// "cudaArrayDeferredMapping" enables such behaviour.
	//
	// For other backends this will be a hassle to optimize
	// between **24** different texture types!!!
	// Probably we will not support streaming on other platforms
	// until SYCL and HIP enables such futures.
	//
	// These backends are not even in conception phase, but
	// future proofing this system will come in handy.


	//
};