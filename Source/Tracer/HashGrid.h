#pragma once

#include "Random.h"

#include "Core/Definitions.h"
#include "Core/BitFunctions.h"
#include "Core/GraphicsFunctions.h"
#include "Core/Span.h"
#include "Core/Types.h"
#include "Core/AABB.h"

#include "Device/GPUMemory.h"
#include "Device/GPUAtomic.h"

#include <numeric>

class SpatioDirCode
{
    public:
    static constexpr uint32_t LEVEL_BITS = 6;           // At most 256-level (overkill but bits left)
    static constexpr uint32_t NORMAL_BITS_PER_DIM = 2;  // 16-cardinal normal directions
                                                        // (We could've put bits here but then it will be too much)
    static constexpr uint32_t MORTON_BITS_PER_DIM = 18; // 2^54 3D-positions

    // Sanity check
    static_assert(LEVEL_BITS + NORMAL_BITS_PER_DIM * 2 +
                  MORTON_BITS_PER_DIM * 3 == sizeof(uint64_t) * CHAR_BIT);

    // Computed offset of the bits
    static constexpr std::array<uint64_t, 4> BIT_OFFSETS = []()
    {
        // LSB to MSB Order
        constexpr std::array<uint64_t, 3> BIT_COUNTS =
        {
            MORTON_BITS_PER_DIM * 3,
            NORMAL_BITS_PER_DIM * 2,
            LEVEL_BITS
        };
        std::array<uint64_t, 4> result = {};
        std::inclusive_scan(BIT_COUNTS.cbegin(), BIT_COUNTS.cend(), result.begin() + 1);
        return result;
    }();

    MR_PF_DECL static uint32_t MaxLevel();
    MR_PF_DECL static uint32_t MaxDirMC();
    MR_PF_DECL static uint64_t MaxPosMC();

    private:
    uint64_t hashCode;

    public:
    // Constructors & Destructor
                          SpatioDirCode() = default;
    MR_PF_DECL_V explicit SpatioDirCode(uint64_t hash);
    MR_PF_DECL_V explicit SpatioDirCode(uint64_t mcPos, uint32_t normalMC, uint32_t level);
    //
    MR_PF_DECL uint64_t PosCode() const;
    MR_PF_DECL uint32_t NormalCode() const;
    MR_PF_DECL uint32_t Level() const;
    //
    constexpr explicit operator uint64_t() const;
    constexpr auto operator<=>(const SpatioDirCode&) const = default;
};

MR_PF_DEF_V
SpatioDirCode::SpatioDirCode(uint64_t hash)
    : hashCode(hash)
{}

// GPU-View of the hash grid
struct HashGridView
{
    static constexpr auto EMPTY_VAL = SpatioDirCode(UINT64_MAX);
    static constexpr auto SENTINEL_VAL = SpatioDirCode(UINT64_MAX - 1);
    struct InsertResult
    {
        uint32_t  allocationIndex;
        bool      isInserted;
    };

    Span<SpatioDirCode> dHashes;
    // Parameters
    Vector3             camLocation;
    AABB3               hashGridRegion;  // Region of the hash grid, probably scene AABB.
                                         // (Unless it is teapot in a stadium kind of scene)
    Float               baseRegionDelta; // Minimum region cell size of the positional codes.
    uint32_t            baseRegionDim;   // Minimum region size of the positional codes.
    Float               normalDelta;     // Same stuff but for normals.
    uint32_t            normalRegionDim; //
    uint32_t            maxLevel;        // Maximum level that the grid can achieve
    // Similar to ray cones, tangent of the ray's aperture.
    // (which is quite larger than the actual ray differential
    // to prevent explosion of nodes).
    // I've tried to use actual cone's ray differentials
    // but could not make it work. It did give great results for magification scenes
    // (Scene in the actual ray differentials paper)
    // But could not prevent node explosion.
    // So we use user defined angle
    Float               tanConeHalfTimes2;

    MR_HF_DECL
    SpatioDirCode       GenCode(const Vector3& pos,
                                const Vector3& normal) const;
    public:
    MR_HF_DECL
    Optional<uint32_t>  Search(const Vector3& pos,
                               const Vector3& normal) const;
    MR_HF_DECL
    InsertResult        TryInsert(const Vector3& pos,
                                  const Vector3& normal) const;
    MR_GF_DECL
    InsertResult        TryInsertAtomic(const Vector3& pos,
                                        const Vector3& normal) const;
};

MR_PF_DEF
uint32_t SpatioDirCode::MaxLevel()
{
    // -1 is here to create a unique value(s) that is never used
    // which will be used as hash table empty marker and sentinel
    constexpr auto R = ((1u << LEVEL_BITS) - 1);
    return R;
}

MR_PF_DEF
uint32_t SpatioDirCode::MaxDirMC()
{
    constexpr auto R = (1u << (NORMAL_BITS_PER_DIM * 2)) - 1;
    return R;
}

MR_PF_DEF
uint64_t SpatioDirCode::MaxPosMC()
{
    constexpr auto R = (1ull << (MORTON_BITS_PER_DIM * 3)) - 1;
    return R;
}

MR_PF_DEF_V
SpatioDirCode::SpatioDirCode(uint64_t mcPos, uint32_t normalMC, uint32_t level)
    : hashCode(Bit::Compose<MORTON_BITS_PER_DIM * 3, NORMAL_BITS_PER_DIM * 2, LEVEL_BITS>(mcPos, normalMC, level))
{
    assert(mcPos <= MaxPosMC());
    assert(level <= MaxLevel());
    assert(normalMC <= MaxDirMC());
}

MR_PF_DEF
uint64_t SpatioDirCode::PosCode() const
{
    return Bit::FetchSubPortion(hashCode, {BIT_OFFSETS[0], BIT_OFFSETS[1]});
}

MR_PF_DEF
uint32_t SpatioDirCode::NormalCode() const
{
    return uint32_t(Bit::FetchSubPortion(hashCode, {BIT_OFFSETS[1], BIT_OFFSETS[2]}));
}

MR_PF_DEF
uint32_t SpatioDirCode::Level() const
{
    return uint32_t(Bit::FetchSubPortion(hashCode, {BIT_OFFSETS[2], BIT_OFFSETS[3]}));
}

constexpr SpatioDirCode::operator uint64_t() const
{
    return hashCode;
}

MR_HF_DEF
SpatioDirCode HashGridView::GenCode(const Vector3& pos,
                                    const Vector3& normal) const
{
    // Level
    Float coneWidth = tanConeHalfTimes2 * Math::Length(pos - camLocation);
    uint32_t ratio = uint32_t(Math::RoundInt(coneWidth * baseRegionDelta));
    uint32_t level = Bit::RequiredBitsToRepresent(ratio);
    level = Math::Min(level, SpatioDirCode::MaxLevel());
    // Position
    Vector3 locF = (pos - hashGridRegion.Min()) * baseRegionDelta;
    Vector3i loc = Math::Clamp(Math::RoundInt(locF),
                               Vector3i::Zero(),
                               Vector3i(baseRegionDim - 1));
    uint64_t mcPos = Graphics::MortonCode::Compose3D<uint64_t>(Vector3ui(loc));
    static constexpr auto MAX_SHIFT = uint32_t(sizeof(uint64_t) * CHAR_BIT - 1);
    mcPos >>= Math::Min(MAX_SHIFT, level * 3u);

    // And finally, normal
    // Normal
    // TODO: Numeric precision stuff (should we wrap here maybe?)
    Vector2 encodedN = Graphics::DirectionToConcentricOctahedral(normal);
    Vector2ui texelN = Vector2ui(Math::RoundInt(encodedN * normalDelta));
    texelN = Math::Clamp(texelN, Vector2ui::Zero(), Vector2ui(normalRegionDim - 1));
    uint32_t mcNormal = Graphics::MortonCode::Compose2D<uint32_t>(texelN);
    //
    return SpatioDirCode(mcPos, mcNormal, level);
}

MR_HF_DEF
Optional<uint32_t> HashGridView::Search(const Vector3& pos,
                                        const Vector3& normal) const
{
    [[maybe_unused]]
    static constexpr auto S_VAL = SENTINEL_VAL;
    static constexpr auto E_VAL = EMPTY_VAL;

    uint32_t tableSize = dHashes.size();
    uint32_t divMask = uint32_t(tableSize - 1);
    SpatioDirCode code = GenCode(pos, normal);
    uint64_t codeInt = static_cast<uint64_t>(code);
    assert(code != S_VAL);
    assert(code != E_VAL);

    // Hash table with linear probing
    using RNGFunctions::HashPCG64::Hash;
    uint32_t index = uint32_t(Hash(codeInt, codeInt) >> 32) & divMask;
    for(uint32_t i = index;; i = (i + 1) & divMask)
    {
        SpatioDirCode checkedCode = dHashes[i];
        // Probe chain is broken and we could not find the key, GG.
        if(checkedCode == E_VAL) return std::nullopt;
        // Found the entry
        if(checkedCode == code)  return i;
        // We looped all the way around and could'nt find the key.
        if(i == index - 1)       break;
    }
    // HT is full probably, so we looped all the entries and could not find
    // the data GG
    return std::nullopt;
}

MR_HF_DEF
typename HashGridView::InsertResult
HashGridView::TryInsert(const Vector3& pos, const Vector3& normal) const
{
    static constexpr auto E_VAL = EMPTY_VAL;
    static constexpr auto S_VAL = SENTINEL_VAL;

    uint32_t tableSize = dHashes.size();
    uint32_t divMask = uint32_t(tableSize - 1);
    SpatioDirCode code = GenCode(pos, normal);
    uint64_t codeInt = static_cast<uint64_t>(code);
    assert(code != S_VAL);
    assert(code != E_VAL);
    // Hash table with linear probing
    using RNGFunctions::HashPCG64::Hash;
    uint32_t index = uint32_t(Hash(codeInt, codeInt) >> 32) & divMask;
    for(uint32_t i = index;; i = (i + 1) & divMask)
    {
        SpatioDirCode checkedCode = dHashes[i];
        if(checkedCode == code) return InsertResult{i, false};
        //
        if(checkedCode == S_VAL || checkedCode == E_VAL)
        {
            dHashes[i] = code;
            return InsertResult{i, true};
        }
        if(i == index - 1) break;
    }
    assert(false && "HT Full!");
    return InsertResult{UINT32_MAX, false};
}

MR_GF_DEF
typename HashGridView::InsertResult
HashGridView::TryInsertAtomic(const Vector3& pos,
                              const Vector3& normal) const
{
    [[maybe_unused]]
    static constexpr auto S_VAL = SENTINEL_VAL;
    static constexpr auto E_VAL = EMPTY_VAL;

    uint32_t tableSize = dHashes.size();
    uint32_t divMask = uint32_t(tableSize - 1);
    SpatioDirCode code = GenCode(pos, normal);
    uint64_t codeInt = static_cast<uint64_t>(code);
    assert(code != E_VAL);
    assert(code != S_VAL);
    // Hash table with linear probing
    using RNGFunctions::HashPCG64::Hash;
    uint32_t index = uint32_t(Hash(codeInt, codeInt) >> 32) & divMask;
    for(uint32_t i = index;; i = (i + 1) & divMask)
    {
        // Pre-load without atomic to feel the region.
        // Whatever we get (except if it is EMPTY) will not change.
        SpatioDirCode checkedCode = dHashes[i];

        // We will do a atomic CAS shortly. We need to have a single state
        // so that we compare with it (if only we systems had "compare <"
        // instead of "compare =="). So we assume HT does not have any
        // sentinels (deleted entries).
        //
        // These entries are assumed to be filled by another kernel.
        assert(checkedCode != S_VAL);
        // Got lucky, this entry is filled by someone else.
        if(checkedCode == code) return InsertResult{i, false};
        // Now we can commit to the atomic operation
        // iff we've seen "EMPTY"
        if(checkedCode == E_VAL)
        {
            using DeviceAtomic::AtomicCompSwap;
            SpatioDirCode old = AtomicCompSwap(dHashes[i], E_VAL, code);
            // Successfull insert.
            if(old == E_VAL) return InsertResult{i, true};
        }
        //
        if(i == index - 1)   break;
    }
    assert(false && "HT Full!");
    return InsertResult{UINT32_MAX, false};
}

class HashGrid
{
    // Rational Number
    static constexpr Float BASE_LOAD_FACTOR = Float(0.6);
    private:
    const GPUSystem&    gpuSystem;
    DeviceMemory        mem;
    Span<SpatioDirCode> dSpatialCodes;
    Span<Byte>          dTransformReduceTempMem;
    Span<uint32_t>      dCountBuffer;

    AABB3       regionAABB;
    Vector3     camLocation;
    uint32_t    baseLevelPositionBits;
    uint32_t    normalBits;
    uint32_t    maxLevel;
    Float       coneAperture;

    public:
    // Constructors & Destructor
                HashGrid(const GPUSystem&);
                HashGrid(const HashGrid&) = delete;
                HashGrid(HashGrid&&) = delete;
    HashGrid&   operator=(const HashGrid&) = delete;
    HashGrid&   operator=(HashGrid&&) = delete;
                ~HashGrid() = default;

    //
    void        Reset(AABB3 regionAABB, Vector3 camLocation,
                      uint32_t baseLevelPositionBits,
                      uint32_t normalBits, uint32_t maxLevel,
                      Float coneApertureDegrees,
                      uint32_t maxEntryCount, const GPUQueue&);

    HashGridView View() const;
    size_t       GPUMemoryUsage() const;
    uint32_t     CalculateUsedGridCount(const GPUQueue&) const;
    uint32_t     EntryCapacity() const;
};

inline
HashGridView HashGrid::View() const
{
    uint32_t baseLevelGridCount = 1u << baseLevelPositionBits;
    Vector3d size = Vector3d(regionAABB.GeomSpan());
    double delta = double(baseLevelGridCount) / size[size.Maximum()];
    // CoOcta Mapping is between [0,1]
    uint32_t normalRegionDim = (1 << normalBits);
    Float normalDelta = Float(normalRegionDim);
    // Pre-calculate the factor here since it is constant
    Float tanHalfTimes2 = Math::Tan(coneAperture * Float(0.5)) * 2;
    //
    return HashGridView
    {
        .dHashes           = dSpatialCodes,
        .camLocation       = camLocation,
        .hashGridRegion    = regionAABB,
        .baseRegionDelta   = Float(delta),
        .baseRegionDim     = baseLevelGridCount,
        .normalDelta       = normalDelta,
        .normalRegionDim   = normalRegionDim,
        .maxLevel          = maxLevel,
        .tanConeHalfTimes2 = tanHalfTimes2
    };
}

inline
size_t HashGrid::GPUMemoryUsage() const
{
    return mem.Size();
}

inline
uint32_t HashGrid::EntryCapacity() const
{
    return dSpatialCodes.size();
}
















// Slightly compressed Markov Chain data of
// Alber et al. (2025).
struct alignas(16) MCAlber2025BaseParams
{
    UNorm2x16   wTarget;
    Float       wTargetScale;
    Float       wSum;
    Float       wCos;
};

struct MarkovChainCountParam
{
    //uint16_t mcN;
    //uint16_t irradN;

    uint32_t mcN;
    uint32_t irradN;
};

struct IrradianceParam
{
    Float irrad;
};

static constexpr uint64_t DATA_COUNT = 10'500'000;
//static constexpr uint64_t TABLE_SIZE = Math::NextPrime(DATA_COUNT * 2);
static constexpr uint64_t TABLE_SIZE = Math::NextPowerOfTwo(DATA_COUNT * 2);


static constexpr auto HASH_PER_GRID = sizeof(SpatioDirCode);
static constexpr auto DATA_PER_GRID = (sizeof(MCAlber2025BaseParams) +
                                       sizeof(MarkovChainCountParam) +
                                       sizeof(IrradianceParam));

static constexpr auto TOTAL_DATA_SIZE = DATA_COUNT * DATA_PER_GRID;
static constexpr auto HASH_DATA_SIZE = DATA_COUNT * HASH_PER_GRID;

static constexpr auto TOTAL_DATA_MIB = double(TOTAL_DATA_SIZE) / 1024. / 1024.;
static constexpr auto HASH_DATA_MIB = double(HASH_DATA_SIZE) / 1024. / 1024.;

static constexpr auto FULL_TOTAL_MIB = TOTAL_DATA_MIB + HASH_DATA_MIB;

