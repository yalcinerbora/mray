
#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <random>

#include "Core/GraphicsFunctions.h"
#include "Core/BitFunctions.h"

#include "Random.h"

#include "Device/GPUSystem.hpp"
#include "DistributionFunctions.h"
#include "Device/GPUWarp.h"

struct RadianceSampleSoA : public SoASpan<UNorm2x16, Float,
                                          Float, Float, Float>
{
    using Base = SoASpan<UNorm2x16, Float, Float, Float, Float>;
    enum PathStateIndex
    {
        DIRECTION,
        POS_X,
        POS_Y,
        POS_Z,
        SCALE
    };
    using Base::Base;

    MR_PF_DECL Vector3 GetPos(uint32_t i) const;
    MR_PF_DECL Vector3 GetDir(uint32_t i) const;
    MR_PF_DECL Float   GetScale(uint32_t i) const;
};

class SpatialCode
{
    static constexpr uint32_t LEVEL_BITS = 4;
    static constexpr uint32_t MORTON_BITS_PER_DIM = 20;
    // Sanity check
    static_assert(LEVEL_BITS + MORTON_BITS_PER_DIM * 3 == sizeof(uint64_t) * CHAR_BIT);

    public:
    MR_PF_DECL static uint32_t MaxLevel();
    MR_PF_DECL static uint64_t MaxMorton();

    private:
    uint64_t mortonCodeAndLevel;

    public:
    // Constructors & Destructor
    SpatialCode() = default;
    MR_PF_DECL_V explicit SpatialCode(uint64_t);
    MR_PF_DECL_V explicit SpatialCode(uint64_t mc, uint32_t level);
    //
    MR_PF_DECL uint64_t Code() const;
    MR_PF_DECL uint32_t Level() const;

    //
    constexpr explicit operator uint64_t() const;
    constexpr auto operator<=>(const SpatialCode&) const = default;
};

class LookupTableSpat
{
    public:
    static constexpr auto EMPTY_VAL = std::numeric_limits<uint64_t>::max();
    static constexpr auto SENTINEL_VAL = std::numeric_limits<uint64_t>::max() - 1;

    private:
    Span<SpatialCode>   keys;
    Span<uint32_t>      values;

    MR_PF_DECL
    static uint64_t Hash(SpatialCode);

    public:
    MR_HF_DECL      LookupTableSpat(const Span<SpatialCode>& keys,
                                    const Span<uint32_t>& values);

    MR_HF_DECL
    Optional<uint32_t>          Search(SpatialCode) const;
    MR_HF_DECL
    Pair<const uint32_t*, bool> Insert(SpatialCode, uint32_t) const;

};

class SpatialDatasetView
{
    private:
    AABB3       sceneSize;
    Vector3     delta;
    uint32_t    octreeL0Size;

    public:
    LookupTableSpat lt;

    public:
    // Constructors & Destructor
    MR_HF_DECL
    SpatialDatasetView(AABB3 sceneSize,
                       uint32_t octreeL0Size,
                       LookupTableSpat lt);

    MR_PF_DECL
    uint64_t ScenePosToMorton(const Vector3& pos) const;
};

//using GaussLobe = Distribution::GaussianLobe;
//using GaussLobeDirParams = Distribution::GaussLobe::GaussLobeDirParams;
//using GaussLobeMeanParams = Distribution::GaussLobe::GaussLobeMeanParams;
//using GaussLobeScaleParams = Distribution::GaussLobe::GaussLobeScaleParams;
//using GaussLobeSoA = Distribution::GaussLobe::GaussLobeSoA;
//static constexpr uint32_t GAUSS_PER_ENTRY = 8;
//
//MR_PF_DEF
//Vector3 RadianceSampleSoA::GetPos(uint32_t i) const
//{
//    return Vector3(Get<POS_X>()[i],
//                   Get<POS_Y>()[i],
//                   Get<POS_Z>()[i]);
//}
//
//MR_PF_DEF
//Vector3 RadianceSampleSoA::GetDir(uint32_t i) const
//{
//    UNorm2x16 dirPacked = Get<DIRECTION>()[i];
//    return Graphics::ConcentricOctahedralToDirection(Vector2(dirPacked));
//}
//
//MR_PF_DEF
//Float RadianceSampleSoA::GetScale(uint32_t i) const
//{
//    return Get<SCALE>()[i];
//}
//
//MR_PF_DEF
//uint32_t SpatialCode::MaxLevel()
//{
//    // -1 is here to create a unique value(s) that is never used
//    // which will be used as hash table empty marker and sentinel
//    return uint32_t(1ull << LEVEL_BITS) - 1;
//}
//
//MR_PF_DEF
//uint64_t SpatialCode::MaxMorton()
//{
//    return (1ull << (MORTON_BITS_PER_DIM * 3));
//}
//
//// Constructors & Destructor
//MR_PF_DEF_V
//SpatialCode::SpatialCode(uint64_t v)
//    : mortonCodeAndLevel(v)
//{}
//
//MR_PF_DEF_V
//SpatialCode::SpatialCode(uint64_t mc, uint32_t level)
//    : mortonCodeAndLevel(Bit::Compose<MORTON_BITS_PER_DIM * 3, LEVEL_BITS>(mc, level))
//{
//    assert(mc < MaxMorton());
//    assert(level < MaxLevel());
//}
//
//MR_PF_DEF
//uint64_t SpatialCode::Code() const
//{
//    constexpr std::array<uint64_t, 2> RANGE =
//    {
//        0, MORTON_BITS_PER_DIM * 3
//    };
//    return Bit::FetchSubPortion(mortonCodeAndLevel, RANGE);
//}
//
//MR_PF_DEF
//uint32_t SpatialCode::Level() const
//{
//    constexpr std::array<uint64_t, 2> RANGE =
//    {
//        MORTON_BITS_PER_DIM * 3,
//        MORTON_BITS_PER_DIM * 3 + LEVEL_BITS
//    };
//    return uint32_t(Bit::FetchSubPortion(mortonCodeAndLevel, RANGE));
//}
//
//constexpr SpatialCode::operator uint64_t() const
//{
//    return mortonCodeAndLevel;
//}
//
//MR_PF_DEF
//uint64_t LookupTableSpat::Hash(SpatialCode sc)
//{
//    // Do nothing, code itself is pretty unique
//    // We may add some hashing later if needed
//    uint64_t hash = static_cast<uint64_t>(sc);
//
//    //hash = Bit::FetchSubPortion(hash, {0, 32}) ^ Bit::FetchSubPortion(hash, {32, 64});
//
//    return hash;
//}
//
//MR_HF_DECL
//LookupTableSpat::LookupTableSpat(const Span<SpatialCode>& keys,
//                                 const Span<uint32_t>& values)
//    : keys(keys)
//    , values(values)
//{
//    assert(keys.size() == values.size());
//    assert(Bit::PopC(keys.size()) == 1);
//}
//
//MR_HF_DEF
//Optional<uint32_t> LookupTableSpat::Search(SpatialCode key) const
//{
//    uint32_t tableSize = keys.size();
//    uint32_t divMask = uint32_t(tableSize - 1);
//    uint64_t hash = Hash(key);
//    assert(hash != SENTINEL_VAL);
//    assert(hash != EMPTY_VAL);
//
//    // Hash table with linear probing
//    uint32_t index = uint32_t(hash) & divMask;
//    for(uint32_t i = index;; i = (i + 1) & divMask)
//    {
//        uint64_t checkedKey = static_cast<uint64_t>(keys[i]);
//        // Probe chain is broken, GG
//        if(checkedKey == EMPTY_VAL) return std::nullopt;
//        if(checkedKey == uint64_t(key))
//            return values[i];
//
//        if(i == index - 1) break;
//    }
//    // HT is full probably, so we looped all the entries and could not find
//    // the data GG
//    return std::nullopt;
//}
//
//MR_HF_DEF
//Pair<const uint32_t*, bool>
//LookupTableSpat::Insert(SpatialCode key, uint32_t value) const
//{
//    using ResultT = Pair<const uint32_t*, bool>;
//
//    uint32_t tableSize = keys.size();
//    uint32_t divMask = uint32_t(tableSize - 1);
//    uint64_t hash = Hash(key);
//    assert(hash != SENTINEL_VAL);
//    assert(hash != EMPTY_VAL);
//
//    // Hash table with linear probing
//    uint32_t index = uint32_t(hash) & divMask;
//    for(uint32_t i = index;; i = (i + 1) & divMask)
//    {
//        uint64_t checkedKey = static_cast<uint64_t>(keys[i]);
//        if(checkedKey == uint64_t(key)) return ResultT(&values[i], false);
//        //
//        if(checkedKey == SENTINEL_VAL || checkedKey == EMPTY_VAL)
//        {
//            values[i] = value;
//            keys[i] = key;
//            return ResultT(&values[i], true);
//        }
//
//        if(i == index - 1) break;
//    }
//    assert(false && "HT Full!");
//    return ResultT(nullptr, false);
//}
//
//MR_HF_DEF
//SpatialDatasetView::SpatialDatasetView(AABB3 sceneSize, uint32_t octreeL0Size,
//                                       LookupTableSpat lt)
//    : sceneSize(sceneSize)
//    , delta(Vector3(octreeL0Size) / sceneSize.GeomSpan())
//    , octreeL0Size(octreeL0Size)
//    , lt(lt)
//{}
//
//MR_PF_DEF
//uint64_t SpatialDatasetView::ScenePosToMorton(const Vector3& pos) const
//{
//    Vector3 locF = (pos - sceneSize.Min()) * delta;
//    Vector3ui loc = Vector3ui(Math::RoundInt(locF));
//    loc = Math::Clamp(loc, 0, octreeL0Size);
//
//    return Graphics::MortonCode::Compose3D<uint64_t>(loc);
//}
//
//MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
//void KCQueryDataStruct(MRAY_GRID_CONSTANT const Span<uint32_t> dDataIndices,
//                       MRAY_GRID_CONSTANT const Span<const SpatialCode> dCodes,
//                       // Constants
//                       MRAY_GRID_CONSTANT const SpatialDatasetView ds)
//{
//    KernelCallParams kp;
//    for(uint32_t i = kp.GlobalId(); i < dCodes.size(); i+= kp.TotalSize())
//    {
//        Optional<uint32_t> dataIndex = ds.lt.Search(dCodes[i]);
//        dDataIndices[i] = dataIndex.value_or(std::numeric_limits<uint32_t>::max());
//    }
//}
//
//MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
//void KCQueryDataStructAndFakeWork(MRAY_GRID_CONSTANT const Span<uint32_t> dDataIndices,
//                                  MRAY_GRID_CONSTANT const Span<const SpatialCode> dCodes,
//                                  // Constants
//                                  MRAY_GRID_CONSTANT const SpatialDatasetView ds,
//                                  MRAY_GRID_CONSTANT const GaussLobeSoA gaussLobeSoA)
//{
//    KernelCallParams kp;
//    for(uint32_t i = kp.GlobalId(); i < dCodes.size(); i += kp.TotalSize())
//    {
//        Optional<uint32_t> dataIndex = ds.lt.Search(dCodes[i]);
//        uint32_t gaussI = dataIndex.value_or(std::numeric_limits<uint32_t>::max());
//        if(gaussI != std::numeric_limits<uint32_t>::max())
//        {
//            uint32_t xx = 0;
//            for(uint32_t j = 0; j < GAUSS_PER_ENTRY; j++)
//            {
//                Vector2ui xiI = Vector2ui(gaussI, gaussI + 1227);
//                Vector2 xi = Vector2 (RNGFunctions::ToFloat01<Float>(xiI[0]),
//                                      RNGFunctions::ToFloat01<Float>(xiI[1]));
//
//                GaussLobe lobe(gaussLobeSoA, gaussI * GAUSS_PER_ENTRY + j);
//                auto sample = lobe.Sample(xi);
//                Float r = (sample.value[0] + sample.value[1] +
//                           sample.value[2] + sample.pdf);
//                xx += uint32_t(r);
//            }
//            dDataIndices[i] = xx;
//        }
//        else dDataIndices[i] = gaussI;
//    }
//}
//
//MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
//void KCQueryDataStructAndFakeWorkWarp(MRAY_GRID_CONSTANT const Span<uint32_t> dDataIndices,
//                                      MRAY_GRID_CONSTANT const Span<const SpatialCode> dCodes,
//                                      // Constants
//                                      MRAY_GRID_CONSTANT const SpatialDatasetView ds,
//                                      MRAY_GRID_CONSTANT const GaussLobeSoA gaussLobeSoA)
//{
//    #ifdef MRAY_DEVICE_CODE_PATH
//
//    static_assert(WarpSize() % GAUSS_PER_ENTRY == 0);
//    KernelCallParams kp;
//
//    static constexpr auto THREAD_PER_WARP = GAUSS_PER_ENTRY;
//
//    uint32_t laneId          = kp.threadId % THREAD_PER_WARP;
//    uint32_t globalWarpId    = kp.GlobalId() / THREAD_PER_WARP;
//    uint32_t globalWarpCount = kp.TotalSize() / THREAD_PER_WARP;
//
//    bool isLeader = (laneId == 0);
//    for(uint32_t i = globalWarpId; i < dCodes.size(); i += globalWarpCount)
//    {
//        uint32_t gaussI;
//        if(isLeader)
//            gaussI = ds.lt.Search(dCodes[i]).value_or(std::numeric_limits<uint32_t>::max());
//        gaussI = DeviceWarp::WarpBroadcast<THREAD_PER_WARP>(0, gaussI);
//        WarpSynchronize<THREAD_PER_WARP>();
//
//        if(gaussI != std::numeric_limits<uint32_t>::max())
//        {
//            using RNGFunctions::HashPCG64::Hash;
//            uint32_t h0 = uint32_t(Hash(gaussI, laneId));
//            uint32_t h1 = uint32_t(Hash(h0, 1));
//            Vector2 xi = Vector2(RNGFunctions::ToFloat01<Float>(h0),
//                                 RNGFunctions::ToFloat01<Float>(h1));
//
//            GaussLobe lobe(gaussLobeSoA, gaussI * GAUSS_PER_ENTRY + laneId);
//            auto sample = lobe.Sample(xi);
//            Float r = (sample.value[0] + sample.value[1] + sample.value[2] + sample.pdf);
//
//            uint32_t xx = uint32_t(r);
//            xx = DeviceWarp::WarpReduceAdd<THREAD_PER_WARP>(xx);
//            WarpSynchronize<THREAD_PER_WARP>();
//
//            if(isLeader)
//                dDataIndices[i] = xx;
//        }
//        else dDataIndices[i] = gaussI;
//    }
//
//    #endif
//}
//
//MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
//void KCQueryDataStructInterp(MRAY_GRID_CONSTANT const Span<uint32_t> dDataIndices,
//                             MRAY_GRID_CONSTANT const Span<const SpatialCode> dCodes,
//                             // Constants
//                             MRAY_GRID_CONSTANT const SpatialDatasetView ds)
//{
//    KernelCallParams kp;
//    for(uint32_t i = kp.GlobalId(); i < dCodes.size(); i += kp.TotalSize())
//    {
//        SpatialCode code = dCodes[i];
//        uint32_t level = code.Level();
//        uint64_t mortonCode = code.Code();
//
//        // Fake an interpolation
//        Vector3ui index = Graphics::MortonCode::Decompose3D<uint64_t>(mortonCode);
//
//        using Graphics::MortonCode::Compose3D;
//        std::array<uint64_t, 8> mortonCodes =
//        {
//            Compose3D<uint64_t>(index + Vector3ui(0, 0, 0)),
//            Compose3D<uint64_t>(index + Vector3ui(1, 0, 0)),
//            Compose3D<uint64_t>(index + Vector3ui(0, 1, 0)),
//            Compose3D<uint64_t>(index + Vector3ui(1, 1, 0)),
//            Compose3D<uint64_t>(index + Vector3ui(0, 0, 1)),
//            Compose3D<uint64_t>(index + Vector3ui(1, 0, 1)),
//            Compose3D<uint64_t>(index + Vector3ui(0, 1, 1)),
//            Compose3D<uint64_t>(index + Vector3ui(1, 1, 1))
//        };
//        Optional<uint32_t> dI0 = ds.lt.Search(SpatialCode(mortonCodes[0], level));
//        Optional<uint32_t> dI1 = ds.lt.Search(SpatialCode(mortonCodes[1], level));
//        Optional<uint32_t> dI2 = ds.lt.Search(SpatialCode(mortonCodes[2], level));
//        Optional<uint32_t> dI3 = ds.lt.Search(SpatialCode(mortonCodes[3], level));
//        Optional<uint32_t> dI4 = ds.lt.Search(SpatialCode(mortonCodes[4], level));
//        Optional<uint32_t> dI5 = ds.lt.Search(SpatialCode(mortonCodes[5], level));
//        Optional<uint32_t> dI6 = ds.lt.Search(SpatialCode(mortonCodes[6], level));
//        Optional<uint32_t> dI7 = ds.lt.Search(SpatialCode(mortonCodes[7], level));
//
//        uint32_t localOut = 0;
//        localOut += dI0.value_or(std::numeric_limits<uint32_t>::max());
//        localOut += dI1.value_or(std::numeric_limits<uint32_t>::max());
//        localOut += dI2.value_or(std::numeric_limits<uint32_t>::max());
//        localOut += dI3.value_or(std::numeric_limits<uint32_t>::max());
//        localOut += dI4.value_or(std::numeric_limits<uint32_t>::max());
//        localOut += dI5.value_or(std::numeric_limits<uint32_t>::max());
//        localOut += dI6.value_or(std::numeric_limits<uint32_t>::max());
//        localOut += dI7.value_or(std::numeric_limits<uint32_t>::max());
//        dDataIndices[i] = localOut;
//    }
//}
//
//
//namespace UpdateMixtureKernelParams
//{
//    static constexpr Vector2ui THRD_RANGE       = Vector2ui(0, 4);
//    static constexpr Vector2ui WARP_4_RANGE     = Vector2ui(4, 8);
//    static constexpr Vector2ui WARP_8_RANGE     = Vector2ui(8, 16);
//    static constexpr Vector2ui WARP_16_RANGE    = Vector2ui(16, 32);
//    static constexpr Vector2ui WARP_32_RANGE    = Vector2ui(32, 64);
//    static constexpr Vector2ui BLOCK_64_RANGE   = Vector2ui(64, 128);
//    static constexpr Vector2ui BLOCK_128_RANGE  = Vector2ui(128, std::numeric_limits<uint32_t>::max());
//
//    MR_PF_DECL
//    bool SkipRangeThread(Vector2ui sampleRange)
//    {
//        uint32_t count = sampleRange[1] - sampleRange[0];
//        return !(THRD_RANGE[0] < count && count < THRD_RANGE[1]);
//    }
//
//    template <uint32_t WARP_SIZE>
//    MR_PF_DECL
//    bool SkipRangeWarp(Vector2ui sampleRange)
//    {
//        uint32_t count = sampleRange[1] - sampleRange[0];
//             if constexpr(WARP_SIZE == 4)  return !(WARP_4_RANGE[0] <= count && count < WARP_4_RANGE[1]);
//        else if constexpr(WARP_SIZE == 8)  return !(WARP_8_RANGE[0] <= count && count < WARP_8_RANGE[1]);
//        else if constexpr(WARP_SIZE == 16) return !(WARP_16_RANGE[0] <= count && count < WARP_16_RANGE[1]);
//        else if constexpr(WARP_SIZE == 32) return !(WARP_32_RANGE[0] <= count && count < WARP_32_RANGE[1]);
//                                           return true;
//    }
//
//    template <uint32_t BLOCK_SIZE>
//    MR_PF_DECL
//    bool SkipRangeBlock(Vector2ui sampleRange)
//    {
//        uint32_t count = sampleRange[1] - sampleRange[0];
//             if constexpr(BLOCK_SIZE ==  64) return !(BLOCK_64_RANGE[0] <= count && count < BLOCK_64_RANGE[1]);
//        else if constexpr(BLOCK_SIZE == 128) return !(BLOCK_128_RANGE[0] <= count && count < BLOCK_128_RANGE[1]);
//                                             return true;
//    }
//};
//
//MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
//void KCUpdateMixturesThread(// I-O
//                            MRAY_GRID_CONSTANT const GaussLobeSoA dGaussLobeSoA,
//                            MRAY_GRID_CONSTANT const Span<Vector4uc> dMixtureCountsPacked,
//                            // Input
//                            MRAY_GRID_CONSTANT const Span<SpatialCode> dCodes,
//                            MRAY_GRID_CONSTANT const Span<Vector2ui> dSegmentRanges,
//                            MRAY_GRID_CONSTANT const Span<uint32_t> dPathSoAIndices,
//                            MRAY_GRID_CONSTANT const RadianceSampleSoA dPathSoA,
//                            MRAY_GRID_CONSTANT const SpatialDatasetView dSpatialTable)
//{
//    //using namespace UpdateMixtureKernelParams;
//    //static constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();
//
//    //KernelCallParams kp;
//    //// Dedicate a thread for each lobe (basic case, CPU probably will call this)
//    //// Grid-stride loop
//    //for(uint32_t cellId = kp.GlobalId(); cellId < dCodes.size(); cellId += kp.TotalSize())
//    //{
//    //    // Defer the computation to the warp kernel
//    //    Vector2ui range = dSegmentRanges[cellId];
//    //    uint32_t totalSampleCount = range[1] - range[0];
//    //    if(SkipRangeThread(range)) continue;
//
//
//
//    //    // Load etc
//    //    uint32_t gaussI = dSpatialTable.lt.Search(dCodes[cellId]).value_or(INVALID_INDEX);
//    //    if(gaussI == INVALID_INDEX) continue;
//
//    //    // Load the loabs to local memory.
//    //    GaussLobe lobe[GAUSS_PER_ENTRY];
//    //    uint32_t h = gaussI / 4;
//    //    uint32_t l = gaussI % 4;
//    //    uint32_t curMixCount = dMixtureCountsPacked[h][l];
//    //    assert(curMixCount <= GAUSS_PER_ENTRY);
//
//    //    for(uint32_t i = 0; i < curMixCount; i++)
//    //        lobe[i] = GaussLobe(dGaussLobeSoA, gaussI * GAUSS_PER_ENTRY + i);
//
//    //    // Do statistics pass in batch
//    //    Float sumWeight;
//    //    Float logLhood;
//    //    Float oldLogLhood;
//    //    //
//    //    static constexpr auto SAMPLE_PER_BATCH = GAUSS_PER_ENTRY;
//    //    static constexpr auto MAX_ITER = 100;
//
//    //    // Stats buffer for the current iteration
//    //    Vector3 sufficientStats[SAMPLE_PER_BATCH];
//    //    Vector3 sufficientStats[SAMPLE_PER_BATCH];
//    //    // Batchify the samples
//    //    uint32_t batchCount = Math::DivideUp(totalSampleCount, SAMPLE_PER_BATCH);
//    //    for(uint32_t bI = 0; bI < batchCount; bI++)
//    //    {
//    //        // EM For a batch of data
//    //        for(uint32_t _ = 0; _ < MAX_ITER; _++)
//    //        {
//    //            Float totalStatWeight = 0;
//    //            for(uint32_t i = 0; i < SAMPLE_PER_BATCH; i++)
//    //                sufficientStats[SAMPLE_PER_BATCH] = Vector3::Zero();
//
//    //            //===============================//
//    //            //  Sufficient stats generation  //
//    //            //===============================//
//    //            uint32_t startI = bI * batchCount;
//    //            uint32_t endI = Math::Max((bI + 1) * batchCount, totalSampleCount);
//    //            uint32_t batchSampleCount = endI - startI;
//    //            for(uint32_t i = 0; i < batchSampleCount; i++)
//    //            {
//    //                uint32_t readIndex = range[0] + i;
//    //                Vector2 dirC = Vector2(dPathSoA.Get<RadianceSampleSoA::DIRECTION>()[readIndex]);
//    //                Vector3 dir = Graphics::ConcentricOctahedralToDirection(dirC);
//
//    //                Float mixturePDF = 0;
//    //                Float softAssignments[SAMPLE_PER_BATCH];
//    //                for(uint32_t lobeIndex = 0; lobeIndex < curMixCount; lobeIndex++)
//    //                {
//    //                    softAssignments[lobeIndex] = lobe[lobeIndex].Pdf(dir);
//    //                    mixturePDF += softAssignments[lobeIndex];
//    //                }
//    //                //
//    //                Float mixPDFRecip = Float(1) / mixturePDF;
//    //                for(uint32_t lobeIndex = 0; lobeIndex < curMixCount; lobeIndex++)
//    //                    softAssignments[lobeIndex] *= mixPDFRecip;
//
//    //                // Assign these to each lobe
//    //                Float scale = dPathSoA.Get<RadianceSampleSoA::SCALE>()[readIndex];
//    //                for(uint32_t lobeIndex = 0; lobeIndex < curMixCount; lobeIndex++)
//    //                {
//    //                    Float assignWeight = softAssignments[lobeIndex] * scale;
//    //                    sufficientStats[lobeIndex] += dir * assignWeight;
//    //                    totalStatWeight += assignWeight;
//    //                }
//    //            }
//    //            //===============================//
//    //            //       Mixture Update          //
//    //            //===============================//
//    //            for(uint32_t lobeIndex = 0; lobeIndex < curMixCount; lobeIndex++)
//    //            {
//    //                Float maxMeanCosine = 0;
//    //                Float mixWeight = lobe.;
//    //            }
//    //        }
//    //    }
//    //        MRAY_GRID_CONSTANT const Span<Vector2ui> dSegmentRanges,
//
//
//    //        gMM.
//    //    }
//
//    //    //
//
//    //    //// EM-pass
//    //    //Vector3 data0;
//    //    //Float sumWeight;
//    //    //static constexpr uint32_t SAMPLE_PER_PASS = 16;
//
//
//    //    ////
//    //    //Float sampleSoftAssignWeights[GAUSS_PER_ENTRY];
//    //    //Float mixturePDF;
//
//
//    //    //Vector3 st
//
//
//    //    //for(uint32_t i = 0; i < 100; i++)
//    //    //{
//    //    //    // For each pass
//    //    //}
//
//    //    //// Read samples
//    //    //Vector2ui range = dSegmentRanges[tId];
//    //    //for(uint32_t i = range[0]; i < range[1]; i++)
//    //    //{
//    //    //    uint32_t sampleI = dPathSoAIndices[i];
//
//    //    //    Vector3 dir = dPathSoA.GetDir(sampleI);
//    //    //    Float scale = dPathSoA.GetScale(sampleI);
//
//
//    //    //}
//    //}
//}
//
//template<uint32_t LOGICAL_WARP_SIZE>
//MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_DEFAULT
//void KCUpdateMixturesWarp(// I-O
//                          MRAY_GRID_CONSTANT const GaussLobeSoA dGaussLobeSoA,
//                          // Input
//                          MRAY_GRID_CONSTANT const Span<SpatialCode> dCodes,
//                          MRAY_GRID_CONSTANT const Span<Vector2ui> dSegmentRanges,
//                          MRAY_GRID_CONSTANT const Span<uint32_t> dPathSoAIndices,
//                          MRAY_GRID_CONSTANT const RadianceSampleSoA dPathSoA,
//                          MRAY_GRID_CONSTANT const SpatialDatasetView dSpatialTable)
//{
//    using namespace UpdateMixtureKernelParams;
//    KernelCallParams kp;
//
//    uint32_t laneId = kp.threadId % LOGICAL_WARP_SIZE;
//    uint32_t globalWarpId = kp.GlobalId() / LOGICAL_WARP_SIZE;
//    uint32_t globalWarpCount = kp.TotalSize() / LOGICAL_WARP_SIZE;
//    // Warp-stride Loop (logical warp)
//    bool isLeader = (laneId == 0);
//    for(uint32_t cellId = globalWarpId; cellId < dCodes.size(); cellId += globalWarpCount)
//    {
//        // Defer computation to block
//        Vector2ui range = dSegmentRanges[cellId];
//        if(SkipRangeWarp<LOGICAL_WARP_SIZE>(range)) continue;
//    }
//}
//
//template<uint32_t BLOCK_SIZE>
//MRAY_KERNEL MRAY_DEVICE_LAUNCH_BOUNDS_CUSTOM(BLOCK_SIZE)
//void KCUpdateMixturesBlock(// I-O
//                           MRAY_GRID_CONSTANT const GaussLobeSoA dGaussLobeSoA,
//                           // Input
//                           MRAY_GRID_CONSTANT const Span<SpatialCode> dCodes,
//                           MRAY_GRID_CONSTANT const Span<Vector2ui> dSegmentRanges,
//                           MRAY_GRID_CONSTANT const Span<uint32_t> dPathSoAIndices,
//                           MRAY_GRID_CONSTANT const RadianceSampleSoA dPathSoA,
//                           MRAY_GRID_CONSTANT const SpatialDatasetView dSpatialTable)
//{
//    KernelCallParams kp;
//
//    // Block-stride loop
//    bool isLeader = (kp.threadId == 0);
//    for(uint32_t cellId = kp.blockId; cellId < dCodes.size(); cellId += kp.gridSize)
//    {
//        // Defer computation to block
//        Vector2ui range = dSegmentRanges[cellId];
//        if(SkipRangeBlock<BLOCK_SIZE>(range)) continue;
//    }
//}
//
//
//TEST(SpatialQuery, Perf)
//{
//    GPUSystem gpuSystem;// (false, 1);
//    const GPUQueue& queue = gpuSystem.BestDevice().GetComputeQueue(0);
//
//    static constexpr unsigned int Seed = 0;
//    static constexpr size_t RayCount = 2'000'000;
//    static constexpr size_t OctreeLeafSize = 2048;
//    // Emulating an sparse voxel octree SVO here, last level is 1M voxels
//    // since it is surface discretization it should reduce by 4 (instead of 8)
//    //static constexpr size_t EntryL0Count = 1'500'000;
//    //static constexpr size_t EntryL1Count =   375'000;
//    //static constexpr size_t EntryL2Count =    93'750;
//    static constexpr size_t EntryL0Count = 500'000;
//    static constexpr size_t EntryL1Count = 125'000;
//    static constexpr size_t EntryL2Count = 31'250;
//
//
//    static constexpr size_t TotalEntries = EntryL0Count + EntryL1Count + EntryL2Count;
//    static constexpr size_t LTSize = Math::NextPowerOfTwo(TotalEntries * 2);
//
//    static constexpr Vector2 SceneRange = Vector2(-50, 120);
//
//    std::mt19937 rng(Seed);
//    // Find random AABB
//    std::uniform_real_distribution aabbDist(SceneRange[0], SceneRange[1]);
//    Vector3 aabbMin(aabbDist(rng), aabbDist(rng), aabbDist(rng));
//    Vector3 aabbMax(aabbDist(rng), aabbDist(rng), aabbDist(rng));
//    for(uint32_t i = 0; i < 3; i++)
//    {
//        if(aabbMin[i] < aabbMax[i]) continue;
//        std::swap(aabbMin[i], aabbMax[i]);
//    }
//    AABB3 sceneAABB = AABB3(aabbMin, aabbMax);
//
//    //
//    std::vector<SpatialCode> hCodes(LTSize, SpatialCode(LookupTableSpat::EMPTY_VAL));
//    std::vector<uint32_t>    hValues(LTSize);
//    //
//    LookupTableSpat lt(hCodes, hValues);
//    SpatialDatasetView spatialDS(sceneAABB, OctreeLeafSize, lt);
//
//    std::uniform_real_distribution distX(sceneAABB.Min()[0], sceneAABB.Max()[0]);
//    std::uniform_real_distribution distY(sceneAABB.Min()[1], sceneAABB.Max()[1]);
//    std::uniform_real_distribution distZ(sceneAABB.Min()[2], sceneAABB.Max()[2]);
//    std::uniform_int_distribution<uint32_t> distLevel(0, 2);
//
//    // Load HT
//    uint32_t dataI = 0;
//    auto AddLevelEntries = [&](uint32_t N, uint32_t levelNo)
//    {
//        for(uint32_t _ = 0; _ < N; _++)
//        {
//            uint32_t index = dataI++;
//            SpatialCode sc;
//            do
//            {
//                // No duplicates
//                Vector3 randomPos(distX(rng), distY(rng), distZ(rng));
//                uint64_t c = spatialDS.ScenePosToMorton(randomPos);
//                sc = SpatialCode(c, levelNo);
//            }
//            while(!(lt.Insert(sc, index).second));
//        }
//    };
//    AddLevelEntries(EntryL0Count, 0);
//    AddLevelEntries(EntryL1Count, 1);
//    AddLevelEntries(EntryL2Count, 2);
//    MRAY_LOG("Insert Done!");
//    // Randomly generate values from filled data, since we always do
//    // find a value
//    std::vector<SpatialCode> hQueryCodes; hQueryCodes.reserve(RayCount);
//    // Every 10 ray hitting the same location is a good estimate
//    static constexpr auto QueryCount = RayCount / 10;
//    static_assert(RayCount % QueryCount == 0);
//    {
//        std::vector<SpatialCode> tmpCodes = hCodes;
//        std::sort(tmpCodes.begin(), tmpCodes.end());
//        tmpCodes.erase(std::unique(tmpCodes.begin(), tmpCodes.end()) - 1, tmpCodes.end());
//        std::shuffle(tmpCodes.begin(), tmpCodes.end(), rng);
//
//        while(tmpCodes.size() < QueryCount)
//            tmpCodes.insert(tmpCodes.end(), tmpCodes.begin(), tmpCodes.end());
//
//        tmpCodes.erase(tmpCodes.end() - (tmpCodes.size() - QueryCount), tmpCodes.end());
//        for(uint32_t i = 0; i < RayCount / QueryCount; i++)
//            hQueryCodes.insert(hQueryCodes.end(), tmpCodes.begin(), tmpCodes.begin() + QueryCount);
//    }
//    // The difference between these are dramatic (non-interpolated access)
//    // On 3070ti Mobile:
//    //     sort    -> 158.8us (microseconds!)
//    //     shuffle -> 1.18ms
//    //
//    // Average 32-bit multi-partition/sort takes around 500us so this can be a performance benefit
//    std::shuffle(hQueryCodes.begin(), hQueryCodes.end(), rng);
//    //std::sort(hQueryCodes.begin(), hQueryCodes.end());
//    assert(hQueryCodes.size() == RayCount);
//
//    // Push HT to the GPU
//    DeviceMemory htMem(gpuSystem.AllGPUs(), 2_MiB, 2_MiB);
//    Span<SpatialCode> dCodes;
//    Span<uint32_t>    dValues;
//    MemAlloc::AllocateMultiData(Tie(dCodes, dValues),
//                                htMem, {hCodes.size(), hValues.size()});
//    //
//    DeviceMemory gaussLobeMem(gpuSystem.AllGPUs(), 2_MiB, 2_MiB);
//    Span<GaussLobeDirParams> dGaussDirParams;
//    Span<GaussLobeMeanParams> dGaussMeanParams;
//    Span<GaussLobeScaleParams> dGaussScaleParams;
//    MemAlloc::AllocateMultiData(Tie(dGaussDirParams,
//                                    dGaussMeanParams,
//                                    dGaussScaleParams),
//                                gaussLobeMem,
//                                {TotalEntries * GAUSS_PER_ENTRY,
//                                 TotalEntries * GAUSS_PER_ENTRY,
//                                 TotalEntries * GAUSS_PER_ENTRY});
//    GaussLobeSoA gaussSoA(dGaussDirParams, dGaussMeanParams, dGaussScaleParams);
//    //
//    Span<SpatialCode> dQueriedCodes;
//    Span<uint32_t>    dOutputIndices;
//    DeviceMemory testIOMem(gpuSystem.AllGPUs(), 2_MiB, 2_MiB);
//    MemAlloc::AllocateMultiData(Tie(dQueriedCodes, dOutputIndices),
//                                testIOMem, {RayCount, RayCount});
//
//    MRAY_LOG("Total HT Memory {}MiB", double(htMem.Size()) / 1024 / 1024);
//    MRAY_LOG("Total Gauss Memory {}MiB", double(gaussLobeMem.Size()) / 1024 / 1024);
//    MRAY_LOG("Total Test IO Memory {}MiB", double(testIOMem.Size()) / 1024 / 1024);
//
//    // Load data
//    queue.MemcpyAsync(dCodes, Span<const SpatialCode>(hCodes));
//    queue.MemcpyAsync(dValues, Span<const uint32_t>(hValues));
//    queue.MemcpyAsync(dQueriedCodes, Span<const SpatialCode>(hQueryCodes));
//
//    LookupTableSpat spatialLTDevice(dCodes, dValues);
//    SpatialDatasetView spatialDSDevice(sceneAABB, OctreeLeafSize, spatialLTDevice);
//    queue.IssueWorkKernel<KCQueryDataStruct>
//    (
//        "KCQueryDataStruct",
//        DeviceWorkIssueParams{.workCount = RayCount},
//        //
//        dOutputIndices,
//        ToConstSpan(dQueriedCodes),
//        spatialDSDevice
//    );
//
//    queue.IssueWorkKernel<KCQueryDataStructAndFakeWork>
//    (
//        "KCQueryDataStructAndFakeWork",
//        DeviceWorkIssueParams{.workCount = RayCount},
//        //
//        dOutputIndices,
//        ToConstSpan(dQueriedCodes),
//        spatialDSDevice,
//        gaussSoA
//    );
//    queue.IssueWorkKernel<KCQueryDataStructAndFakeWorkWarp>
//    (
//        "KCQueryDataStructAndFakeWorkWarp",
//        DeviceWorkIssueParams{.workCount = RayCount},
//        //
//        dOutputIndices,
//        ToConstSpan(dQueriedCodes),
//        spatialDSDevice,
//        gaussSoA
//    );
//
//    queue.IssueWorkKernel<KCQueryDataStructInterp>
//    (
//        "KCQueryDataStructInterp",
//        DeviceWorkIssueParams{.workCount = RayCount},
//        //
//        dOutputIndices,
//        ToConstSpan(dQueriedCodes),
//        spatialDSDevice
//    );
//
//    std::vector<uint32_t> dProbeCounts(dOutputIndices.size());
//    queue.MemcpyAsync(Span<uint32_t>(dProbeCounts), ToConstSpan(dOutputIndices));
//
//    // Check perf
//    queue.Barrier().Wait();
//
//    //for(uint32_t i : dProbeCounts)
//    //{
//    //    if(i >= 16)
//    //        MRAY_LOG("{}", i);
//    //}
//
//}
//
//
//
//
//TEST(GaussLobeMix, Update)
//{
//
//}
//
//
//namespace MCAlber2025
//{
//
//    // Slightly compressed Markov Chain data of
//    // Alber et al. (2025).
//    struct alignas(16) MCAlber2025BaseParams
//    {
//        UNorm2x16   wTarget;
//        Float       wTargetScale;
//        Float       wSum;
//        Float       wCos;
//    };
//
//    struct MarkovChainCountParam
//    {
//        uint16_t mcN;
//        uint16_t irradN;
//    };
//
//    struct IrradianceParam
//    {
//        Float irrad;
//    };
//
//
//    // Register-space class of the parameters
//    struct MCAlber2025
//    {
//        static constexpr uint32_t WINDOW_SIZE = 1024;
//
//        Vector3 wTarget;
//        Float   wTargetScale;
//        Float   wSum;
//        Float   wCos;
//        //
//        //UNorm2x16 w_tgt;
//        // Current window size (initially, state will start with 0 it'll be
//        //uint32_t curWind; // 6-bit or 22-bit left here;
//    };
//
//}