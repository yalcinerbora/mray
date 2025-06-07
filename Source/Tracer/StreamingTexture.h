#pragma once

#include <cstdint>
#include <array>

#include "Device/GPUTypes.h"
#include "Device/GPUTextureView.h"

#include "Core/GraphicsFunctions.h"
#include "Core/DataStructures.h"

#include "Random.h"
#include "Bitspan.h"

namespace StreamingTexParams
{
	// Type indices of the texture view
	// This value will reside in VirtualTexId
	enum TextureViewTypeIndices : uint32_t
	{
		// 1 Channel
		R8_UNORM = 0,
		R8_SNORM = 1,
		R16_UNORM = 2,
		R16_SNORM = 3,
		R32F = 4,
		BC4_UNORM = 5,
		BC4_SNORM = 6,
		END_1C = 7,
		// 2 Channel
		RG8_UNORM = 0,
		RG8_SNORM = 1,
		RG16_UNORM = 2,
		RG16_SNORM = 3,
		RG32F = 4,
		BC5_UNORM = 5,
		BC5_SNORM = 6,
		END_2C = 7,

		// 3-4 channel
		RGBA8_UNORM = 0,
		RGBA8_SNORM = 1,
		RGBA16_UNORM = 2,
		RGBA16_SNORM = 3,
		RGBA32F = 4,
		BC1 = 5,
		BC2 = 6,
		BC3 = 7,
		BC6_UNORM = 8,
		BC6_SNORM = 9,
		BC7 = 10,
		END_4C = 11
	};
	// This is by design, must not be changed
	static constexpr uint32_t	MaxPackCount = 2u;
	// Some Automatic calculations due to bit
	static constexpr uint32_t	TypeIndexBits = std::max({Bit::RequiredBitsToRepresent<uint32_t>(END_1C),
													      Bit::RequiredBitsToRepresent<uint32_t>(END_2C),
													      Bit::RequiredBitsToRepresent<uint32_t>(END_4C)});
	static constexpr uint32_t	 VirtualTexIndexBits = (sizeof(uint32_t) * CHAR_BIT -
														TypeIndexBits -
														Bit::RequiredBitsToRepresent(MaxPackCount));
	// This is total maximum total alised texture per pack
	// We have two packs total (1:1 aspect 2:1 aspect tiled textures)
	static constexpr uint32_t	MaxPhysicalTextureCount = 1024u;
	// Bits left on the 32-bit virtual tex id.
	static constexpr uint32_t	MaxVirtualTextures = (1u << VirtualTexIndexBits);
	//
	static constexpr uint32_t	MaxPhysicalTileIndexBits = (sizeof(uint32_t) * CHAR_BIT -
															Bit::RequiredBitsToRepresent(MaxPhysicalTextureCount));
	static constexpr uint32_t	MaxPhysicalTilePerTex = (1u << MaxPhysicalTileIndexBits);

	static constexpr uint32_t	PaddedPixels = 4u;
	// TODO: Check this due to padding
	static constexpr Float		MaxAnisotropy = Float(4);
	static constexpr Float		MaxInvAnisotropy = Float(1) / MaxAnisotropy;
}

struct VirtualTexInfo
{
	enum TextureType : uint8_t
	{
		R8_UNORM,
		R8_SNORM,
		R16_UNORM,
		R16_SNORM,
		R32F,
		BC4_UNORM,
		BC4_SNORM,

		RG8_UNORM,
		RG8_SNORM,
		RG16_UNORM,
		RG16_SNORM,
		RG32F,
		BC5_UNORM,
		BC5_SNORM,

		RGBA8_UNORM,
		RGBA8_SNORM,
		RGBA16_UNORM,
		RGBA16_SNORM,
		RGBA32F,
		BC1,
		BC2,
		BC3,
		BC6_UNORM,
		BC6_SNORM,
		BC7
	};

	Vector2ui	dim;
	uint32_t	tileBitStart;
	TextureType texType;
	uint8_t		mipCount;
};

// Id types
// TODO: why not C's bit parameters?
// I have had issues with it in the past but I forgot
class PhysicalTileId
{
	// Bit pattern
	static constexpr uint32_t PA_BITS_END = sizeof(uint32_t) * CHAR_BIT;
	static constexpr uint32_t PA_BITS_START = PA_BITS_END -
		Bit::RequiredBitsToRepresent(StreamingTexParams::MaxPhysicalTextureCount);
	//
	static constexpr uint32_t TI_BITS_END = PA_BITS_START;
	static constexpr uint32_t TI_BITS_START = 0;
	// Sanity check
	static_assert((PA_BITS_END - PA_BITS_START +
				   TI_BITS_END - TI_BITS_START) == sizeof(uint32_t) * CHAR_BIT);
	public:
	static constexpr uint32_t PA_BITS = PA_BITS_END - PA_BITS_START;
	static constexpr uint32_t TI_BITS = TI_BITS_END - TI_BITS_START;

	private:
	uint32_t value;
	public:
	// Constructors & Destructor
	MRAY_HYBRID PhysicalTileId(uint32_t physicalArrayIndex,
							   uint32_t physicalTileIndex);
	//
	MRAY_HYBRID explicit	operator uint32_t();
	MRAY_HYBRID uint32_t	ArrayIndex();
	MRAY_HYBRID uint32_t	TileIndex();
};

class VirtualTextureId
{
	// Bit pattern
	static constexpr uint32_t PI_BITS_END = sizeof(uint32_t) * CHAR_BIT;
	static constexpr uint32_t PI_BITS_START = PI_BITS_END -
		Bit::RequiredBitsToRepresent(StreamingTexParams::MaxPackCount);
	//
	static constexpr uint32_t TI_BITS_END = PI_BITS_START;
	static constexpr uint32_t TI_BITS_START = TI_BITS_END - StreamingTexParams::TypeIndexBits;
	//
	static constexpr uint32_t TEX_BITS_END = TI_BITS_START;
	static constexpr uint32_t TEX_BITS_START = 0;
	// Sanity check
	static_assert((TEX_BITS_END - TEX_BITS_START +
				   TI_BITS_END - TI_BITS_START +
				   PI_BITS_END - PI_BITS_START) == sizeof(uint32_t) * CHAR_BIT);
	public:
	static constexpr uint32_t PI_BITS	= PI_BITS_END - PI_BITS_START;
	static constexpr uint32_t TI_BITS	= TI_BITS_END - TI_BITS_START;
	static constexpr uint32_t TEX_BITS	= TEX_BITS_END - TEX_BITS_START;

	private:
	uint32_t value;
	public:
	// Constructors & Destructor
	MRAY_HYBRID VirtualTextureId(uint32_t packIndex,
								 uint32_t typeIndex,
								 uint32_t textureIndex);
	//
	MRAY_HYBRID explicit	operator uint32_t();
	MRAY_HYBRID uint32_t	PackIndex();
	MRAY_HYBRID uint32_t	TypeIndex();
	MRAY_HYBRID uint32_t	TextureIndex();
};

struct StreamingTextureViewPack
{
	private:
	static constexpr size_t COUNT_1C_TEX = StreamingTexParams::TextureViewTypeIndices::END_1C;
	static constexpr size_t COUNT_2C_TEX = StreamingTexParams::TextureViewTypeIndices::END_2C;
	static constexpr size_t COUNT_4C_TEX = StreamingTexParams::TextureViewTypeIndices::END_4C;

	static constexpr auto PTC = StreamingTexParams::MaxPhysicalTextureCount;
	template<uint32_t C, class T>
	using ArrayOfArrays = std::array<std::array<TextureView<2, T>, PTC>, C>;

	public:
	ArrayOfArrays<COUNT_1C_TEX, Float>		texViews1C;
	ArrayOfArrays<COUNT_2C_TEX, Vector2>	texViews2C;
	ArrayOfArrays<COUNT_4C_TEX, Vector4>	texViews4C;
};

struct TextureStreamingContext
{
	struct PhysicalTileHTStrat
	{
		using K = int64_t;
		using H = uint32_t;
		static constexpr H EMPTY_VAL	= std::numeric_limits<H>::max();
		static constexpr H SENTINEL_VAL = EMPTY_VAL - H(1);
		//
		MRAY_HYBRID
		static H Hash(K v)
		{
			uint64_t hash = RNGFunctions::HashPCG64::Hash(v);
			uint32_t hashH = uint32_t(Bit::FetchSubPortion(hash, {32, 64}));
			// Dump values correspond to these set to zero and one
			// We could uniformly distribute these with double hashing maybe
			// but these should not come up that often
			if(hashH == EMPTY_VAL) return 0u;
			if(hashH == SENTINEL_VAL) return 1u;
			return hashH;
		}
		MRAY_HYBRID
		static bool IsSentinel(H h) { return h == SENTINEL_VAL; }
		MRAY_HYBRID
		static bool IsEmpty(H h) { return h == EMPTY_VAL; }
	};

	public:
	using TextureViews = StreamingTextureViewPack;
	using PhysicalTileHT = LookupTable<uint64_t, PhysicalTileId,
		                               uint32_t, 4u, PhysicalTileHTStrat>;
	private:
	// Aliased texture views for each type
	TextureViews			textureViews;
	PhysicalTileHT			tileLookup[2];
	// Persistent Data
	Span<VirtualTexInfo>	texInfoList;
	Bitspan<uint32_t>		residencyBits;
	Bitspan<uint32_t>		requestBits;


	private:
	MRAY_HYBRID static
	std::array<Vector2, 3> CalculatePhysicalUVsGrad(PhysicalTileId pTid,
													Vector2 uv,
													Vector2 dpdx,
													Vector2 dpdy);
	//
	public:
	MRAY_HYBRID static constexpr
	uint32_t 		TileBitOffset(const VirtualTexInfo& texInfo,
								  uint32_t virtualMipIndex,
								  uint32_t virtualTileIndex);

	MRAY_HYBRID static constexpr
	uint32_t		FetchTileSize(uint32_t typeId, bool isPhysical);

	MRAY_HYBRID static constexpr
	uint64_t		GenerateVirtualTextureKey(uint32_t virtualTextureIndex,
											  uint32_t virtualMipIndex,
											  uint32_t virtualTileIndex);
	MRAY_HYBRID static
	Vector2ui 		GenVirtualMipAndTileLod(const VirtualTexInfo& texInfo,
											Vector2 texel, Float lod);
	MRAY_HYBRID static
	Vector2ui		GenVirtualMipAndTileGrad(const VirtualTexInfo& texInfo,
											 Vector2 uv, Vector2 ddx, Vector2 ddy);
	public:
	template<class T>
	MRAY_GPU
	Optional<T> FetchTextureBase(VirtualTextureId, Vector2 uv);

	template<class T>
	MRAY_GPU
	Optional<T> FetchTextureGrad(VirtualTextureId, Vector2 uv,
								 Vector2 dpdx, Vector2 dpdy);

	template<class T>
	MRAY_GPU
	Optional<T> FetchTextureLod(VirtualTextureId, Vector2 uv,
								Float mipLevel);
};

template<class T>
class StreamingTextureView
{
    using UV = UVType<2>;

    private:
	TextureStreamingContext*	context;
	VirtualTextureId			texId;

	public:
	MRAY_HYBRID	StreamingTextureView(VirtualTextureId,
									 TextureStreamingContext*);

	MRAY_GPU
	Optional<T> operator()(UV uv) const;

	MRAY_GPU
	Optional<T> operator()(UV uv, UV dpdx, UV dpdy) const;

	MRAY_GPU
	Optional<T> operator()(UV uv, Float lod) const;
};

MRAY_HYBRID MRAY_CGPU_INLINE
PhysicalTileId::PhysicalTileId(uint32_t physicalArrayIndex,
							   uint32_t physicalTileIndex)
	: value(Bit::Compose<PA_BITS, TI_BITS>(physicalArrayIndex,
										   physicalTileIndex))
{}

MRAY_HYBRID MRAY_CGPU_INLINE
PhysicalTileId::operator uint32_t()
{
	return value;
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t PhysicalTileId::ArrayIndex()
{
	return Bit::FetchSubPortion(value, {PA_BITS_START, PA_BITS_END});
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t PhysicalTileId::TileIndex()
{
	return Bit::FetchSubPortion(value, {TI_BITS_START, TI_BITS_END});
}

MRAY_HYBRID MRAY_CGPU_INLINE
VirtualTextureId::VirtualTextureId(uint32_t packIndex,
								   uint32_t typeIndex,
								   uint32_t textureIndex)
	: value(Bit::Compose<PI_BITS, TI_BITS, TEX_BITS>(packIndex, typeIndex, textureIndex))
{}

MRAY_HYBRID MRAY_CGPU_INLINE
VirtualTextureId::operator uint32_t()
{
	return value;
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t VirtualTextureId::PackIndex()
{
	return Bit::FetchSubPortion(value, {PI_BITS_START, PI_BITS_END});
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t VirtualTextureId::TypeIndex()
{
	return Bit::FetchSubPortion(value, {TI_BITS_START, TI_BITS_END});
}

MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t VirtualTextureId::TextureIndex()
{
	return Bit::FetchSubPortion(value, {TEX_BITS_START, TEX_BITS_END});
}

MRAY_HYBRID MRAY_CGPU_INLINE
constexpr uint32_t TextureStreamingContext::TileBitOffset(const VirtualTexInfo& texInfo,
														  uint32_t virtMipIndex,
														  uint32_t virtTileIndex)
{
	// TODO: We load the texInfo here as well, compiler may
	Vector2ui tileSize2D = Vector2ui(FetchTileSize(texInfo.texType, false));
	Vector2ui virtTexBaseTileCount = Math::DivideUp(texInfo.dim, tileSize2D);
	uint32_t offset = Graphics::TextureMipPixelStart(virtTexBaseTileCount,
													 virtMipIndex);
	offset += virtTileIndex;
	offset += texInfo.tileBitStart;
	return offset;
}

MRAY_HYBRID MRAY_CGPU_INLINE
constexpr uint32_t TextureStreamingContext::FetchTileSize(uint32_t typeId, bool isPhysical)
{
	// Must match VirtualTexInfo::TextureType
	constexpr std::array TextureTileSizes =
	{
		128u
	};

	uint32_t tileSize = TextureTileSizes[typeId];
	if(!isPhysical)
		tileSize -= StreamingTexParams::PaddedPixels * 2;
	return tileSize;
}

MRAY_HYBRID MRAY_CGPU_INLINE
std::array<Vector2, 3>
TextureStreamingContext::CalculatePhysicalUVsGrad(PhysicalTileId pTid,
												  Vector2 uv,
												  Vector2 dpdx,
												  Vector2 dpdy)
{
	return {};
}

MRAY_HYBRID MRAY_CGPU_INLINE
constexpr  uint64_t TextureStreamingContext::GenerateVirtualTextureKey(uint32_t virtTextureIndex,
																	   uint32_t virtMipIndex,
																	   uint32_t virtTileIndex)
{
	// 24-bit virtual tile id (supports 16kx16k, 10x10, rgb32f udims)
	// Because we assume udim's are single large texture.
	// 17-bit mip level id 160k x 160k input texture
	// 23-bit texture id = 8.3M textures
	static_assert(VirtualTextureId::PI_BITS <= 23,
				  "There is a mismatch of bits between Virtual Texture Key"
				  "Generation and VirtualTextureId class");
	return Bit::Compose<23, 17, 24>(uint64_t(virtTextureIndex),
									uint64_t(virtMipIndex),
									uint64_t(virtTileIndex));
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector2ui TextureStreamingContext::GenVirtualMipAndTileLod(const VirtualTexInfo& texInfo,
														   Vector2 uv, Float lod)
{
	// UV to mip level texel
	uint32_t baseMip = uint32_t(std::floor(lod));
	Vector2ui mipDims = Graphics::TextureMipSize(texInfo.dim, baseMip);
	Vector2ui mipTexel = Vector2ui(uv * Vector2(mipDims));

	// Texel to tile
	uint32_t virtTileSize = FetchTileSize(texInfo.texType, false);
	Vector2ui mipTile2D = mipTexel / Vector2ui(virtTileSize);
	uint32_t virtTileCountX = Math::DivideUp(mipDims[0], virtTileSize);
	uint32_t mipTileLinear = mipTile2D[1] * virtTileCountX + mipTile2D[0];

	return Vector2ui(baseMip, mipTileLinear);
}

MRAY_HYBRID MRAY_CGPU_INLINE
Vector2ui TextureStreamingContext::GenVirtualMipAndTileGrad(const VirtualTexInfo& texInfo,
															Vector2 uv, Vector2 ddx,
															Vector2 ddy)
{
	// https://registry.khronos.org/OpenGL/extensions/EXT/EXT_texture_filter_anisotropic.txt
	ddx *= Vector2(texInfo.dim);
	ddy *= Vector2(texInfo.dim);
	Float pX = ddx.Length();
	Float pY = ddy.Length();
	Float pMax = std::max(pX, pY);
	Float pMin = std::min(pX, pY);
	Float n = std::min(std::ceil(pMax / pMin), Float(StreamingTexParams::MaxAnisotropy));
	Float lod = std::log2(pMax / n);

	return GenVirtualMipAndTileLod(texInfo, uv, lod);
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T>
TextureStreamingContext::FetchTextureBase(VirtualTextureId tid, Vector2 uv)
{
	return std::nullopt;
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T>
TextureStreamingContext::FetchTextureGrad(VirtualTextureId tid, Vector2 uv,
										  Vector2 dpdx, Vector2 dpdy)
{
	uint32_t packIndex = tid.PackIndex();
	uint32_t typeIndex = tid.TypeIndex();
	uint32_t virtTexIndex = tid.TextureIndex();
	VirtualTexInfo texInfo = texInfoList[virtTexIndex];
	// Find the virtual tile
	auto [virtMip, virtTile] = GenVirtualMipAndTileGrad(texInfo, uv, dpdx, dpdy).AsArray();
	// HT Lookup
	uint64_t key = GenerateVirtualTextureKey(virtTexIndex, virtMip, virtTile);
	Optional<const PhysicalTileId*> physicalTilePtr = tileLookup[packIndex].Search(key);
	if(!physicalTilePtr.has_value())
	{
		uint32_t tileBitOffset = TileBitOffset(texInfo, virtMip, virtTile);
		// Request the tile by setting it bit to one
		requestBits.SetBitParallel(tileBitOffset, true);
		return std::nullopt;
	}
	// Calculate the fetch parameters
	PhysicalTileId physicalTilePack = *physicalTilePtr.value();
    uint32_t physicalTexIndex = physicalTilePack.ArrayIndex();
	auto [uvP, dpdxP, dpdyP] = CalculatePhysicalUVsGrad(physicalTilePack,
													    uv, dpdx, dpdy);

	if constexpr(std::is_same_v<T, Float>)
	{
		TextureView<2, Float> tex = textureViews.texViews1C[typeIndex][physicalTexIndex];
		return tex(uvP, dpdxP, dpdyP);
	}
	else if constexpr(std::is_same_v<T, Vector2>)
	{
		TextureView<2, Vector2> tex = textureViews.texViews2C[typeIndex][physicalTexIndex];
		return tex(uvP, dpdxP, dpdyP);
	}
	else if constexpr(std::is_same_v<T, Vector3> ||
					  std::is_same_v<T, Vector4>)
	{
		TextureView<2, Vector4> tex = textureViews.texViews4C[typeIndex][physicalTexIndex];
		Optional<Vector4> data = tex(uvP, dpdxP, dpdyP);

		if constexpr(std::is_same_v<T, Vector3>)
		{
			if(data.has_value())
				return Optional<Vector3>(Vector3(*data));
			else return std::nullopt;
		}
		else return data;
	}
}

template<class T>
Optional<T>
MRAY_GPU MRAY_GPU_INLINE
TextureStreamingContext::FetchTextureLod(VirtualTextureId tid, Vector2 uv,
										 Float mipLevel)
{
	return std::nullopt;
}

template<class T>
MRAY_HYBRID MRAY_CGPU_INLINE
StreamingTextureView<T>::StreamingTextureView(VirtualTextureId tid,
											  TextureStreamingContext* c)
	: texId(tid)
	, context(c)
{}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T>
StreamingTextureView<T>::operator()(UV uv) const
{
	return context->FetchTextureBase<T>(texId, uv);
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T>
StreamingTextureView<T>::operator()(UV uv, UV dpdx, UV dpdy) const
{
	return context->FetchTextureGrad<T>(texId, uv, dpdx, dpdy);
}

template<class T>
MRAY_GPU MRAY_GPU_INLINE
Optional<T>
StreamingTextureView<T>::operator()(UV uv, Float lod) const
{
	return context->FetchTextureLod<T>(texId, uv, lod);
}

template class StreamingTextureView<Vector3>;