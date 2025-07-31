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
	static constexpr uint32_t	VirtualTexIndexBits = (sizeof(uint32_t) * CHAR_BIT -
													   TypeIndexBits -
													   Bit::RequiredBitsToRepresent(MaxPackCount));
	// This is total maximum total aliased texture per pack
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

	// This should not be changed (not tested)
	static constexpr uint32_t	PhysicalTileSize		= 65536u;
	static constexpr Vector2ui	TilePerPhysicalTexMip0	= Vector2ui(64, 64);
	static constexpr Vector2ui	TilePerPhysicalTexMip1	= Vector2ui(TilePerPhysicalTexMip0[0] >> 1,
																	TilePerPhysicalTexMip0[1] >> 1);
	static constexpr uint32_t	PhysicalTexByteSize		= (TilePerPhysicalTexMip0.Multiply() * PhysicalTileSize +
														   TilePerPhysicalTexMip1.Multiply() * PhysicalTileSize);
	static constexpr uint32_t	PhysicalTexMemAlignment = PhysicalTileSize;
	//
	template<MRayPixelEnum E>
	constexpr Vector2ui TilePixSize()
	{
		using PT = MRayPixelType<E>;
		using InnerType = typename PT::Type;
		//
		uint32_t tileWidth = 0;
		if constexpr(PT::IsBCPixel)
		{
			tileWidth = PhysicalTileSize / InnerType::BlockSize;
			static_assert(PhysicalTileSize % InnerType::BlockSize == 0,
						  "PhysicalTileSize must be evenly divisible by BC block size");
		}
		else
		{
			tileWidth = PhysicalTileSize / PT::PaddedPixelSize;
			static_assert(PhysicalTileSize % PT::PaddedPixelSize == 0,
						  "PhysicalTileSize must be evenly divisible by pixels size");
		}

		uint32_t tZeroes = Bit::CountTZero(tileWidth);
		uint32_t wExponent = (tZeroes + 1) / 2;
		uint32_t hExponent = tZeroes % 2 == 0 ? wExponent : (wExponent - 1);
		Vector2ui result = Vector2ui(1u << wExponent, 1u << hExponent);
		if constexpr(PT::IsBCPixel)
			result *= Vector2ui(InnerType::TileSize);
		return result;
	}

	static constexpr std::array TypeTileSizeList =
	{
		// UNORMS
		TilePixSize<MRayPixelEnum::MR_R8_UNORM>(),
		TilePixSize<MRayPixelEnum::MR_RG8_UNORM>(),
		TilePixSize<MRayPixelEnum::MR_RGB8_UNORM>(),
		TilePixSize<MRayPixelEnum::MR_RGBA8_UNORM>(),
		TilePixSize<MRayPixelEnum::MR_R16_UNORM>(),
		TilePixSize<MRayPixelEnum::MR_RG16_UNORM>(),
		TilePixSize<MRayPixelEnum::MR_RGB16_UNORM>(),
		TilePixSize<MRayPixelEnum::MR_RGBA16_UNORM>(),
		// SNORM
		TilePixSize<MRayPixelEnum::MR_R8_SNORM>(),
		TilePixSize<MRayPixelEnum::MR_RG8_SNORM>(),
		TilePixSize<MRayPixelEnum::MR_RGB8_SNORM>(),
		TilePixSize<MRayPixelEnum::MR_RGBA8_SNORM>(),
		TilePixSize<MRayPixelEnum::MR_R16_SNORM>(),
		TilePixSize<MRayPixelEnum::MR_RG16_SNORM>(),
		TilePixSize<MRayPixelEnum::MR_RGB16_SNORM>(),
		TilePixSize<MRayPixelEnum::MR_RGBA16_SNORM>(),
		// FLOAT
		TilePixSize<MRayPixelEnum::MR_R_HALF>(),
		TilePixSize<MRayPixelEnum::MR_RG_HALF>(),
		TilePixSize<MRayPixelEnum::MR_RGB_HALF>(),
		TilePixSize<MRayPixelEnum::MR_RGBA_HALF>(),
		TilePixSize<MRayPixelEnum::MR_R_FLOAT>(),
		TilePixSize<MRayPixelEnum::MR_RG_FLOAT>(),
		TilePixSize<MRayPixelEnum::MR_RGB_FLOAT>(),
		TilePixSize<MRayPixelEnum::MR_RGBA_FLOAT>(),

		// Block Compressed
		TilePixSize<MRayPixelEnum::MR_BC1_UNORM>(),
		TilePixSize<MRayPixelEnum::MR_BC2_UNORM>(),
		TilePixSize<MRayPixelEnum::MR_BC3_UNORM>(),
		TilePixSize<MRayPixelEnum::MR_BC4_UNORM>(),
		TilePixSize<MRayPixelEnum::MR_BC4_SNORM>(),
		TilePixSize<MRayPixelEnum::MR_BC5_UNORM>(),
		TilePixSize<MRayPixelEnum::MR_BC5_SNORM>(),
		TilePixSize<MRayPixelEnum::MR_BC6H_UFLOAT>(),
		TilePixSize<MRayPixelEnum::MR_BC6H_SFLOAT>(),
		TilePixSize<MRayPixelEnum::MR_BC7_UNORM>()
	};

}

struct VirtualTexInfo
{
	Vector2ui		dim;
	uint32_t		tileBitStart;
	MRayPixelEnum	pixType;
	uint8_t			mipCount;
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
	MR_HF_DECL PhysicalTileId(uint32_t physicalArrayIndex,
							  uint32_t physicalTileIndex);
	//
	MR_HF_DECL explicit	operator uint32_t();
	MR_HF_DECL uint32_t	ArrayIndex();
	MR_HF_DECL uint32_t	TileIndex();
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
	MR_HF_DECL VirtualTextureId(uint32_t packIndex,
								uint32_t typeIndex,
								uint32_t textureIndex);
	//
	MR_HF_DECL explicit	operator uint32_t();
	MR_HF_DECL uint32_t	PackIndex();
	MR_HF_DECL uint32_t	TypeIndex();
	MR_HF_DECL uint32_t	TextureIndex();
};

struct StreamingTextureDeviceData
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
		using K = uint64_t;
		using H = uint32_t;
		static constexpr H EMPTY_VAL	= std::numeric_limits<H>::max();
		static constexpr H SENTINEL_VAL = EMPTY_VAL - H(1);
		//
		MR_HF_DECL
		static H Hash(K v)
		{
			uint64_t hash = RNGFunctions::HashPCG64::Hash(v);
			uint32_t hashH = uint32_t(Bit::FetchSubPortion(hash, {32, 64}));
			uint32_t hashL = uint32_t(Bit::FetchSubPortion(hash, {0, 32}));
			uint32_t hashFold = hashH + hashL;
			// Dump values correspond to these set to zero and one
			// We could uniformly distribute these with double hashing maybe
			// but these should not come up that often
			if(hashFold == EMPTY_VAL)		return 0u;
			if(hashFold == SENTINEL_VAL)	return 1u;
			return hashFold;
		}
		MR_HF_DECL
		static bool IsSentinel(H h) { return h == SENTINEL_VAL; }
		MR_HF_DECL
		static bool IsEmpty(H h) { return h == EMPTY_VAL; }
	};

	public:
	using TextureViews = StreamingTextureDeviceData;
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
	MR_HF_DECL static
	std::array<Vector2, 3>		CalculatePhysicalUVsGrad(const VirtualTexInfo& texInfo,
													     PhysicalTileId pTid,
													     Vector2 uv,
													     Vector2 dpdx,
													     Vector2 dpdy);
	MR_HF_DECL static
	std::pair<Vector2, Float>	CalculatePhysicalUVsLod(const VirtualTexInfo& texInfo,
													    PhysicalTileId,
													    Vector2 uv, Float lod);
	public:
	MR_HF_DECL static constexpr
	uint32_t 		TileBitOffset(const VirtualTexInfo& texInfo,
								  uint32_t virtualMipIndex,
								  uint32_t virtualTileIndex);

	MR_HF_DECL static constexpr
	uint64_t		GenerateVirtualTextureKey(uint32_t virtualTextureIndex,
											  uint32_t virtualMipIndex,
											  uint32_t virtualTileIndex);

	MR_HF_DECL static constexpr
	uint32_t		FetchChannelCount(MRayPixelEnum pixType);
	MR_HF_DECL static constexpr
	Vector2ui		FetchTileSize(MRayPixelEnum pixType, bool isPhysical);
	MR_HF_DECL static
	Vector2ui 		GenVirtualMipAndTileLod(const VirtualTexInfo& texInfo,
											Vector2 texel, Float lod);
	MR_HF_DECL static
	Vector2ui		GenVirtualMipAndTileGrad(const VirtualTexInfo& texInfo,
											 Vector2 uv, Vector2 ddx, Vector2 ddy);
	public:
	MR_GF_DECL Vector4 FetchTextureBase(VirtualTextureId, Vector2 uv);
	MR_GF_DECL Vector4 FetchTextureGrad(VirtualTextureId, Vector2 uv,
									  Vector2 dpdx, Vector2 dpdy);
	MR_GF_DECL Vector4 FetchTextureLod(VirtualTextureId, Vector2 uv,
								     Float mipLevel);
};

class StreamingTextureView
{
    using UV = UVType<2>;

    private:
	TextureStreamingContext*	context;
	VirtualTextureId			texId;

	public:
	MR_HF_DECL	StreamingTextureView(VirtualTextureId,
									 TextureStreamingContext*);

	MR_GF_DECL Vector4 operator()(UV uv) const;
	MR_GF_DECL Vector4 operator()(UV uv, UV dpdx, UV dpdy) const;
	MR_GF_DECL Vector4 operator()(UV uv, Float lod) const;
};

MR_HF_DEF
PhysicalTileId::PhysicalTileId(uint32_t physicalArrayIndex,
							   uint32_t physicalTileIndex)
	: value(Bit::Compose<PA_BITS, TI_BITS>(physicalArrayIndex,
										   physicalTileIndex))
{}

MR_HF_DEF
PhysicalTileId::operator uint32_t()
{
	return value;
}

MR_HF_DEF
uint32_t PhysicalTileId::ArrayIndex()
{
	return Bit::FetchSubPortion(value, {PA_BITS_START, PA_BITS_END});
}

MR_HF_DEF
uint32_t PhysicalTileId::TileIndex()
{
	return Bit::FetchSubPortion(value, {TI_BITS_START, TI_BITS_END});
}

MR_HF_DEF
VirtualTextureId::VirtualTextureId(uint32_t packIndex,
								   uint32_t typeIndex,
								   uint32_t textureIndex)
	: value(Bit::Compose<PI_BITS, TI_BITS, TEX_BITS>(packIndex, typeIndex, textureIndex))
{}

MR_HF_DEF
VirtualTextureId::operator uint32_t()
{
	return value;
}

MR_HF_DEF
uint32_t VirtualTextureId::PackIndex()
{
	return Bit::FetchSubPortion(value, {PI_BITS_START, PI_BITS_END});
}

MR_HF_DEF
uint32_t VirtualTextureId::TypeIndex()
{
	return Bit::FetchSubPortion(value, {TI_BITS_START, TI_BITS_END});
}

MR_HF_DEF
uint32_t VirtualTextureId::TextureIndex()
{
	return Bit::FetchSubPortion(value, {TEX_BITS_START, TEX_BITS_END});
}

MR_HF_DEF
constexpr uint32_t TextureStreamingContext::TileBitOffset(const VirtualTexInfo& texInfo,
														  uint32_t virtMipIndex,
														  uint32_t virtTileIndex)
{
	Vector2ui tileSize2D = Vector2ui(FetchTileSize(texInfo.pixType, false));
	Vector2ui virtTexBaseTileCount = Math::DivideUp(texInfo.dim, tileSize2D);
	uint32_t offset = Graphics::TextureMipPixelStart(virtTexBaseTileCount,
													 virtMipIndex);
	offset += virtTileIndex;
	offset += texInfo.tileBitStart;
	return offset;
}

MR_HF_DEF
std::array<Vector2, 3>
TextureStreamingContext::CalculatePhysicalUVsGrad(const VirtualTexInfo& texInfo,
												  PhysicalTileId,
												  Vector2 uv,
												  Vector2 dpdx,
												  Vector2 dpdy)
{
	using namespace StreamingTexParams;
	//static constexpr Vector2ui TilesPerTex = TilePerPhysicalTexMip0;
	//static constexpr auto TileSizeListGPU = TypeTileSizeList;
	Vector2ui tileSize = FetchTileSize(texInfo.pixType, true);
	Vector2ui physicalTexResolution = tileSize * Vector2ui(TilePerPhysicalTexMip0);
	// We scale the differentials and uv with this ratio
	Vector2 dimRatio = Vector2(texInfo.dim) / Vector2(physicalTexResolution);
	return
	{
		uv,
		dpdx * dimRatio,
		dpdy * dimRatio
	};
}

MR_HF_DEF
std::pair<Vector2, Float>
TextureStreamingContext::CalculatePhysicalUVsLod(const VirtualTexInfo&,
												  PhysicalTileId,
												  Vector2 uv, Float lod)
{
	return std::pair{uv, lod};
}

MR_HF_DEF constexpr
uint64_t TextureStreamingContext::GenerateVirtualTextureKey(uint32_t virtTextureIndex,
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

MR_HF_DEF constexpr
uint32_t TextureStreamingContext::FetchChannelCount(MRayPixelEnum pixType)
{
	using namespace StreamingTexParams;
	switch(pixType)
	{
		using enum MRayPixelEnum;
		case MR_R8_UNORM:
		case MR_R8_SNORM:
		case MR_R16_UNORM:
		case MR_R16_SNORM:
		case MR_R_HALF:
		case MR_R_FLOAT:
		case MR_BC4_UNORM:
		case MR_BC4_SNORM:
			return 1;

		case MR_RG8_UNORM:
		case MR_RG8_SNORM:
		case MR_RG16_UNORM:
		case MR_RG16_SNORM:
		case MR_RG_HALF:
		case MR_RG_FLOAT:
		case MR_BC5_UNORM:
		case MR_BC5_SNORM:
			return 2;
		// Fake 3 channels
		case MR_RGB8_UNORM:
		case MR_RGB8_SNORM:
		case MR_RGB16_UNORM:
		case MR_RGB16_SNORM:
		case MR_RGB_HALF:
		case MR_RGB_FLOAT:
		// 4 channels
		case MR_RGBA8_UNORM:
		case MR_RGBA8_SNORM:
		case MR_RGBA16_UNORM:
		case MR_RGBA16_SNORM:
		case MR_RGBA_HALF:
		case MR_RGBA_FLOAT:
		case MR_BC1_UNORM:
		case MR_BC2_UNORM:
		case MR_BC3_UNORM:
		case MR_BC6H_UFLOAT:
		case MR_BC6H_SFLOAT:
		case MR_BC7_UNORM:
			return 4;
		default:
		{
			assert(false);
			// Immediately crash on release
			return std::numeric_limits<uint32_t>::max();
		}
	}
}

MR_HF_DEF constexpr
Vector2ui TextureStreamingContext::FetchTileSize(MRayPixelEnum pixType, bool isPhysical)
{
	using namespace StreamingTexParams;
	// We converted the TypeTileSizeList to switch/case
	// to transfer the conditionality to the instruction space
	// (hopefully, compiler may do whatever it wants)
	// This statement:
	// "static constexpr auto TileSizeListGPU = TypeTileSizeList;"
	// loads the data to constant memory and fetches it.
	// With switch case we hope the constexpr values to
	// propagate
	Vector2ui tileSize = Vector2ui::Zero();
	switch(pixType)
	{
		using enum MRayPixelEnum;
		case MR_R8_UNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_R8_UNORM)]; break;
		case MR_RG8_UNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_RG8_UNORM)]; break;
		case MR_RGB8_UNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_RGB8_UNORM)]; break;
		case MR_RGBA8_UNORM:	tileSize = TypeTileSizeList[static_cast<int>(MR_RGBA8_UNORM)]; break;
		case MR_R16_UNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_R16_UNORM)]; break;
		case MR_RG16_UNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_RG16_UNORM)]; break;
		case MR_RGB16_UNORM:	tileSize = TypeTileSizeList[static_cast<int>(MR_RGB16_UNORM)]; break;
		case MR_RGBA16_UNORM:	tileSize = TypeTileSizeList[static_cast<int>(MR_RGBA16_UNORM)]; break;
		case MR_R8_SNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_R8_SNORM)]; break;
		case MR_RG8_SNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_RG8_SNORM)]; break;
		case MR_RGB8_SNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_RGB8_SNORM)]; break;
		case MR_RGBA8_SNORM:	tileSize = TypeTileSizeList[static_cast<int>(MR_RGBA8_SNORM)]; break;
		case MR_R16_SNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_R16_SNORM)]; break;
		case MR_RG16_SNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_RG16_SNORM)]; break;
		case MR_RGB16_SNORM:	tileSize = TypeTileSizeList[static_cast<int>(MR_RGB16_SNORM)]; break;
		case MR_RGBA16_SNORM:	tileSize = TypeTileSizeList[static_cast<int>(MR_RGBA16_SNORM)]; break;
		case MR_R_HALF:			tileSize = TypeTileSizeList[static_cast<int>(MR_R_HALF)]; break;
		case MR_RG_HALF:		tileSize = TypeTileSizeList[static_cast<int>(MR_RG_HALF)]; break;
		case MR_RGB_HALF:		tileSize = TypeTileSizeList[static_cast<int>(MR_RGB_HALF)]; break;
		case MR_RGBA_HALF:		tileSize = TypeTileSizeList[static_cast<int>(MR_RGBA_HALF)]; break;
		case MR_R_FLOAT:		tileSize = TypeTileSizeList[static_cast<int>(MR_R_FLOAT)]; break;
		case MR_RG_FLOAT:		tileSize = TypeTileSizeList[static_cast<int>(MR_RG_FLOAT)]; break;
		case MR_RGB_FLOAT:		tileSize = TypeTileSizeList[static_cast<int>(MR_RGB_FLOAT)]; break;
		case MR_RGBA_FLOAT:		tileSize = TypeTileSizeList[static_cast<int>(MR_RGBA_FLOAT)]; break;
		case MR_BC1_UNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_BC1_UNORM)]; break;
		case MR_BC2_UNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_BC2_UNORM)]; break;
		case MR_BC3_UNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_BC3_UNORM)]; break;
		case MR_BC4_UNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_BC4_UNORM)]; break;
		case MR_BC4_SNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_BC4_SNORM)]; break;
		case MR_BC5_UNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_BC5_UNORM)]; break;
		case MR_BC5_SNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_BC5_SNORM)]; break;
		case MR_BC6H_UFLOAT:	tileSize = TypeTileSizeList[static_cast<int>(MR_BC6H_UFLOAT)]; break;
		case MR_BC6H_SFLOAT:	tileSize = TypeTileSizeList[static_cast<int>(MR_BC6H_SFLOAT)]; break;
		case MR_BC7_UNORM:		tileSize = TypeTileSizeList[static_cast<int>(MR_BC7_UNORM)]; break;
		default:				MRAY_UNREACHABLE;
	}

	if(!isPhysical)
		tileSize -= Vector2ui(PaddedPixels * 2);
	return tileSize;
}

MR_HF_DEF
Vector2ui TextureStreamingContext::GenVirtualMipAndTileLod(const VirtualTexInfo& texInfo,
														   Vector2 uv, Float lod)
{
	// UV to mip level texel
	uint32_t baseMip = uint32_t(Math::Floor(lod));
	Vector2ui mipDims = Graphics::TextureMipSize(texInfo.dim, baseMip);
	Vector2ui mipTexel = Vector2ui(uv * Vector2(mipDims));

	// Texel to tile
	Vector2ui virtTileSize = FetchTileSize(texInfo.pixType, false);
	Vector2ui mipTile2D = mipTexel / virtTileSize;
	uint32_t virtTileCountX = Math::DivideUp(mipDims[0], virtTileSize[0]);
	uint32_t mipTileLinear = mipTile2D[1] * virtTileCountX + mipTile2D[0];

	return Vector2ui(baseMip, mipTileLinear);
}

MR_HF_DEF
Vector2ui TextureStreamingContext::GenVirtualMipAndTileGrad(const VirtualTexInfo& texInfo,
															Vector2 uv, Vector2 ddx,
															Vector2 ddy)
{
	// https://registry.khronos.org/OpenGL/extensions/EXT/EXT_texture_filter_anisotropic.txt
	ddx *= Vector2(texInfo.dim);
	ddy *= Vector2(texInfo.dim);
	Float pX = Math::Length(ddx);
	Float pY = Math::Length(ddy);
	Float pMax = Math::Max(pX, pY);
	Float pMin = Math::Min(pX, pY);
	Float n = Math::Min(Math::Ceil(pMax / pMin), Float(StreamingTexParams::MaxAnisotropy));
	Float lod = Math::Log2(pMax / n);

	return GenVirtualMipAndTileLod(texInfo, uv, lod);
}

MR_HF_DEF
StreamingTextureView::StreamingTextureView(VirtualTextureId tid,
										   TextureStreamingContext* c)
	: context(c)
	, texId(tid)
{}