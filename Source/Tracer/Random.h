#pragma once

#include <cstdint>

#include "Core/Vector.h"
#include "Core/TypeGenFunction.h"
#include "Core/TracerEnums.h"

#include "Device/GPUSystem.h"
#include "Device/GPUMemory.h"

#include "TracerTypes.h"

class ThreadPool;
struct RenderImageParams;

namespace RNGFunctions::HashPCG64
{
    // I've checked the PBRT, and decided to use Murmur first
    // but arbitrary byte size made the NVCC not happy
    // (no register-space code generation, unaligned loads etc)
    // Insted I've come up with this paper (JCGT has nice gems)
    // https://jcgt.org/published/0009/03/02/
    //
    // PGC looks like a good choice, so we will implement it as a
    // hashing function (instead of RNG)
    //
    // To make it applicable to arbitrary sizes we will recursively
    // call it as discussed in the paper (i.e., H(x_n + H(x_{n-1} + H(...)));
    //
    // This should give the compiler enough information to generate register
    // space code when data is again in register space.

    // Some type traits to dynamically bit_cast to underlying int type
    // bit_cast will handle all the quirks (hopefully)
    // TODO: I may have written this somewhere else
    // but could not find it.
    namespace Detail
    {
        template<class T>
        struct uintX
        {
            static_assert(sizeof(T) <= sizeof(uint64_t),
                          "Hashing requires the data types to be at most 64-bit!");
        };

        template<class T>
        requires(sizeof(T) == sizeof(uint8_t))
        struct uintX<T> { using type = uint8_t; };

        template<class T>
        requires(sizeof(T) == sizeof(uint16_t))
        struct uintX<T> { using type = uint16_t; };

        template<class T>
        requires(sizeof(T) == sizeof(uint32_t))
        struct uintX<T> { using type = uint32_t; };

        template<class T>
        requires(sizeof(T) == sizeof(uint64_t))
        struct uintX<T> { using type = uint64_t; };

        template<class T>
        using uintX_t = typename uintX<T>::type;

        template<class T>
        inline constexpr bool IntCastable_v = (sizeof(T) == sizeof(uint8_t)  ||
                                               sizeof(T) == sizeof(uint16_t) ||
                                               sizeof(T) == sizeof(uint32_t) ||
                                               sizeof(T) == sizeof(uint64_t));
    }

    MR_HF_DEF constexpr uint64_t Hash(uint64_t v);

    template<class... Args>
    requires(Detail::IntCastable_v<Args> && ...)
    MR_HF_DEF constexpr uint64_t Hash(Args&&...);
}

namespace RNGFunctions
{
    template<std::floating_point F, std::unsigned_integral T>
    MR_HF_DECL static F ToFloat01(T);

    // Old code we can't use this for sobol afaik.
    //template<>
    //MR_HF_DEF float ToFloat01<float, uint32_t>(uint32_t v)
    //{
    //    // Sanity Checks
    //    static_assert(std::numeric_limits<float>::is_iec559, "Non-standard floating point!");
    //    static_assert(sizeof(uint32_t) == sizeof(float), "float is not 32-bit!");

    //    // This is simpler version also it's 24-bit instead of 23-bit
    //    // https://marc-b-reynolds.github.io/distribution/2017/01/17/DenseFloat.html
    //    float rngFloat = static_cast<float>(v >> 8);
    //    return  rngFloat * 0x1p-24f;
    //}

    // TODO: Implement double as well as other wide types.
    // i.e. get sample from 64-bit integer to float etc.
    template<>
    MR_HF_DEF float ToFloat01<float, uint32_t>(uint32_t v)
    {
        // Up to this number we get greatest float that is less than 1
        // So we must clamp here
        static_assert(float(0x0) * float(0x1.p-32f) >= float(0));
        static_assert(float(0xFFFFFF7F) * float(0x1.p-32f) != float(1));
        static_assert(float(0xFFFFFF80) * float(0x1.p-32f) == float(1));

        // Unlike the code above, the data is evenly
        // distributed in 32-bit space for sobol?
        // So we just scale the given data to 0x1.p-32f (1.0 x 2^-32)
        // to make it fit into the [0, 1).
        //
        // We hope IEEE fp convention will handle the rest...
        static constexpr float SCALE = float(0x1.p-32f);
        static constexpr float PREV_FLOAT = Math::PrevFloat(float(1));
        // TODO: is there a way to prevent this min?
        return Math::Min(float(v) * SCALE, PREV_FLOAT);
    }
}

namespace SobolDetail
{
    // Rest is at the impl. file
    struct alignas(16) LocalState
    {
        Vector2ui pixel;
        uint32_t sampleIndex;
        uint32_t seed;
    };

    struct GlobalState
    {
        Span<const uint32_t>    dSobolMatices;
        Vector2ui               resolution;
    };
}

namespace ZSobolDetail
{
    // Rest is at the impl. file
    struct alignas(16) LocalState
    {
        uint64_t pixelMortonCode;
        uint32_t sampleIndex;
        uint32_t seed;
    };

    struct GlobalState
    {
        uint32_t    initialMaxSPP;
        uint32_t    resMaxBits;
    };
}

template <std::unsigned_integral T>
struct RNGDispenserT
{
    private:
    Span<const T>   randomNumbers;
    uint32_t        globalId;
    uint32_t        stride;

    public:
    MR_HF_DECL      RNGDispenserT(Span<const T> numbers,
                                  uint32_t globalId,
                                  uint32_t stride);

    template<uint32_t I>
    MR_HF_DECL Float   NextFloat();
    template<uint32_t I>
    MR_HF_DECL Vector2 NextFloat2D();
    MR_HF_DECL void    Advance(uint32_t offset);
};

using RandomNumber = uint32_t;
using RNGDispenser = RNGDispenserT<RandomNumber>;

// RNG Request List
// Up until ZSobol implementation we just use a RN count variable
// that each material/renderer work returns. Now zsobol requires dimension
// information and corraletion between adjacent dimension when 2D sampling
// is required. So we utilize a bitset that represents this information.
// each 2bits corralates a sample which can be 1/2/3. Up to 3D sampling is supported
// per request
class RNRequestList
{
    private:
    uint64_t list = 0; // up to 32 samples, each can at most be 3D.
    uint32_t fillCounter = 0;

    static constexpr uint64_t BIT_PER_REQUEST = 2;
    static constexpr uint64_t MASK = (uint64_t(1) << BIT_PER_REQUEST) - 1;
    static constexpr uint64_t BIT_LIMIT = sizeof(uint64_t) * CHAR_BIT;

    public:
    // Constructors & Destructor
    RNRequestList() = default;

    // Utility
    MR_PF_DECL uint32_t DimensionOfRequest(uint32_t requestIndex) const;
    MR_PF_DECL uint32_t TotalRNCount() const;
    MR_PF_DECL uint32_t TotalRequestCount() const;

    MR_PF_DECL RNRequestList Append(uint32_t dimOfRequest) const;
    MR_PF_DECL RNRequestList Append(const RNRequestList&) const;

    MR_PF_DECL_V void CombineMax(const RNRequestList&);
};

// Convenience for constexpr generation
template<uint32_t... Is>
MR_PF_DECL RNRequestList GenRNRequestList();

// PCG with 32bit state
// https://www.pcg-random.org
// This RNG predominantly be used as a "backup" generator,
// It has minimal state and advancement capabilities
// (which will be usefull on volume rendering I hope)
//
// Each ray probably has one of these when it requires an undeterministic
// amount of random numbers. (Anyhit shader (alpha blend), volume rendering)
// Minimal state is usefull since we will pass these as ray payload on optix
// 64-bit version may be used here but we will try to minimize register
// usage for this generator. Because we already have high quality sample
// generator from Sobol etc.
//
// This class is designed for register space usage,
// upon destruction it will save its state to the device memory that is given.
// So there is no solid class throughout the execution of the program.
// Use static function "GenState" to create the seed.
class PermutedCG32
{
    static constexpr uint32_t Multiplier = 747796405u;
    static constexpr uint32_t Increment = 2891336453u;

    public:
    using State         = uint32_t;

    MR_HF_DECL
    static State    GenerateState(uint32_t seed);

    private:
    State&          dState;
    State           rState;

    MR_HF_DECL
    static State    Step(State r);
    MR_HF_DECL
    static uint32_t Permute(State r);

    public:
    MR_HF_DECL     PermutedCG32(State& dState);
    MR_HF_DECL     ~PermutedCG32();

    MR_HF_DECL
    uint32_t        Next();

    MR_HF_DECL
    Float           NextFloat();

    MR_HF_DECL
    void            Advance(uint32_t delta);
};

using BackupRNG = PermutedCG32;
using BackupRNGState = typename PermutedCG32::State;

// Renderers store path dimension states on their own
// Mostly as a uint16_t buffer
// We can't make these lambdas  due to an NVCC error
//
// "An extended __host__ __device__
// lambda cannot be defined inside a generic lambda expression."
//
// So we define these here
class ConstAddFunctor_U16
{
    private:
    uint16_t constant;

    public:
    ConstAddFunctor_U16(uint16_t c) : constant(c) {}

    constexpr void operator()(uint16_t& i) const noexcept
    {
        i += constant;
    }
};

class SetFunctor_U16
{
    private:
    uint16_t constant;

    public:
    SetFunctor_U16(uint16_t c) : constant(c) {}


    constexpr void operator()(uint16_t& i) const noexcept
    {
        i = constant;
    }
};

class RNGeneratorGroupI
{
    public:
    virtual         ~RNGeneratorGroupI() = default;

    virtual void    SetupRange(Vector2ui rangeStart,
                               Vector2ui rangeEnd,
                               const GPUQueue& queue) = 0;

    virtual void    GenerateNumbers(// Output
                                    Span<RandomNumber> dNumbersOut,
                                    // Constants
                                    uint16_t dimensionStart,
                                    RNRequestList rnRequests,
                                    const GPUQueue& queue) const = 0;
    virtual void    GenerateNumbersIndirect(// Output
                                            Span<RandomNumber> dNumbersOut,
                                            // Input
                                            Span<const RayIndex> dIndices,
                                            // Constants
                                            uint16_t dimensionStart,
                                            RNRequestList rnRequests,
                                            const GPUQueue& queue) const = 0;
    virtual void    GenerateNumbersIndirect(// Output
                                            Span<RandomNumber> dNumbersOut,
                                            // Input
                                            Span<const RayIndex> dIndices,
                                            Span<const uint16_t> dDimensionStart,
                                            // Constants
                                            RNRequestList rnRequests,
                                            const GPUQueue& queue) const = 0;
    virtual void    IncrementSampleId(const GPUQueue& queue) const = 0;
    virtual void    IncrementSampleIdIndirect(Span<const RayIndex> dIndices,
                                              const GPUQueue& queue) const = 0;

    virtual Span<BackupRNGState>    GetBackupStates() = 0;
    virtual size_t                  GPUMemoryUsage() const = 0;
};

using RNGGenerator = GeneratorFuncType<RNGeneratorGroupI, const RenderImageParams&,
                                       Vector2ui, uint32_t, uint64_t,
                                       const GPUSystem&, ThreadPool&>;
using RNGeneratorPtr = std::unique_ptr<RNGeneratorGroupI>;
using RNGGeneratorMap = Map<typename SamplerType::E, RNGGenerator>;

class RNGGroupIndependent : public RNGeneratorGroupI
{
    public:
    using MainRNG = PermutedCG32;
    using MainRNGState = typename MainRNG::State;
    static constexpr typename SamplerType::E TypeName = SamplerType::INDEPENDENT;

    private:
    ThreadPool&                 mainThreadPool;
    const GPUSystem&            gpuSystem;
    // Due to tiling, we save the all state of the
    // entire image in host side.
    // This will copy the data to required HW side
    Vector2ui                   generatorCount;
    std::array<Vector2ui, 2>    currentRange;

    //
    HostLocalMemory             hostMem;
    Span<BackupRNGState>        hBackupStatesAll;
    Span<MainRNGState>          hMainStatesAll;
    // Device local
    DeviceMemory                deviceMem;
    Span<BackupRNGState>        dBackupStatesLocal;
    Span<MainRNGState>          dMainStatesLocal;

    public:
    // Constructors & Destructor
            RNGGroupIndependent(const RenderImageParams& rIParams,
                                Vector2ui maxGPUPresentRNGCount,
                                uint32_t initialMaxSampleCount,
                                uint64_t seed,
                                const GPUSystem& system,
                                ThreadPool& mainThreadPool);

    void    SetupRange(Vector2ui rangeStart,
                       Vector2ui rangeEnd,
                       const GPUQueue& queue) override;

    void    GenerateNumbers(// Output
                            Span<RandomNumber> dNumbersOut,
                            // Constants
                            uint16_t dimensionStart,
                            RNRequestList rnRequests,
                            const GPUQueue& queue) const override;
    void    GenerateNumbersIndirect(// Output
                                    Span<RandomNumber> dNumbersOut,
                                    // Input
                                    Span<const RayIndex> dIndices,
                                    // Constants
                                    uint16_t dimensionStart,
                                    RNRequestList rnRequests,
                                    const GPUQueue& queue) const override;
    void    GenerateNumbersIndirect(// Output
                                    Span<RandomNumber> dNumbersOut,
                                    // Input
                                    Span<const RayIndex> dIndices,
                                    Span<const uint16_t> dDimensionStart,
                                    // Constants
                                    RNRequestList rnRequests,
                                    const GPUQueue& queue) const override;
    //
    void    IncrementSampleId(const GPUQueue& queue) const override;
    void    IncrementSampleIdIndirect(Span<const RayIndex> dIndices,
                                      const GPUQueue& queue) const override;

    Span<BackupRNGState> GetBackupStates() override;
    size_t               GPUMemoryUsage() const override;
};

// Implementation of
// https://www.pbr-book.org/4ed/Sampling_and_Reconstruction/Sobol_Samplers
//
// Which is implementation of this paper
// https://dl.acm.org/doi/10.1145/3414685.3417881
class RNGGroupZSobol : public RNGeneratorGroupI
{
    public:
    using MainRNGState = typename ZSobolDetail::LocalState;
    using MainRNGGlobalState = typename ZSobolDetail::GlobalState;
    static constexpr typename SamplerType::E TypeName = SamplerType::Z_SOBOL;

    private:
    ThreadPool&             mainThreadPool;
    const GPUSystem&        gpuSystem;
    // Due to tiling, we save the all state of the
    // entire image in host side.
    // This will copy the data to required HW side
    Vector2ui                   generatorCount;
    std::array<Vector2ui, 2>    currentRange;

    //
    HostLocalMemory             hostMem;
    Span<BackupRNGState>        hBackupStatesAll;
    Span<MainRNGState>          hMainStatesAll;
    // Device local
    DeviceMemory                deviceMem;
    Span<BackupRNGState>        dBackupStatesLocal;
    Span<MainRNGState>          dMainStatesLocal;
    MainRNGGlobalState          globalState;

    public:
    // Constructors & Destructor
            RNGGroupZSobol(const RenderImageParams& rIParams,
                           Vector2ui maxGPUPresentRNGCount,
                           uint32_t initialMaxSampleCount,
                           uint64_t seed,
                           const GPUSystem& system,
                           ThreadPool& mainThreadPool);

    void    SetupRange(Vector2ui rangeStart,
                       Vector2ui rangeEnd,
                       const GPUQueue& queue) override;

    void    GenerateNumbers(// Output
                            Span<RandomNumber> dNumbersOut,
                            // Constants
                            uint16_t dimensionStart,
                            RNRequestList rnRequests,
                            const GPUQueue& queue) const override;
    void    GenerateNumbersIndirect(// Output
                                    Span<RandomNumber> dNumbersOut,
                                    // Input
                                    Span<const RayIndex> dIndices,
                                    // Constants
                                    uint16_t dimensionStart,
                                    RNRequestList rnRequests,
                                    const GPUQueue& queue) const override;
    void    GenerateNumbersIndirect(// Output
                                    Span<RandomNumber> dNumbersOut,
                                    // Input
                                    Span<const RayIndex> dIndices,
                                    Span<const uint16_t> dDimensionStart,
                                    // Constants
                                    RNRequestList rnRequests,
                                    const GPUQueue& queue) const override;
    void    IncrementSampleId(const GPUQueue& queue) const override;
    void    IncrementSampleIdIndirect(Span<const RayIndex> dIndices,
                                      const GPUQueue& queue) const override;


    Span<BackupRNGState> GetBackupStates() override;
    size_t               GPUMemoryUsage() const override;
};

// Implementation of
// https://www.pbr-book.org/4ed/Sampling_and_Reconstruction/Sobol_Samplers
//
// Which is implementation of this paper
// https://dl.acm.org/doi/10.1145/3414685.3417881
class RNGGroupSobol : public RNGeneratorGroupI
{
    public:
    using MainRNGState = typename SobolDetail::LocalState;
    using MainRNGGlobalState = typename SobolDetail::GlobalState;
    static constexpr typename SamplerType::E TypeName = SamplerType::SOBOL;

    private:
    ThreadPool&             mainThreadPool;
    const GPUSystem&        gpuSystem;
    // Due to tiling, we save the all state of the
    // entire image in host side.
    // This will copy the data to required HW side
    Vector2ui                   generatorCount;
    std::array<Vector2ui, 2>    currentRange;

    //
    HostLocalMemory             hostMem;
    Span<BackupRNGState>        hBackupStatesAll;
    Span<MainRNGState>          hMainStatesAll;
    // Device local
    DeviceMemory                deviceMem;
    Span<uint32_t>              dSobolMatrices;
    Span<BackupRNGState>        dBackupStatesLocal;
    Span<MainRNGState>          dMainStatesLocal;
    MainRNGGlobalState          globalState;

    public:
    // Constructors & Destructor
            RNGGroupSobol(const RenderImageParams& rIParams,
                          Vector2ui maxGPUPresentRNGCount,
                          uint32_t initialMaxSampleCount,
                          uint64_t seed,
                          const GPUSystem& system,
                          ThreadPool& mainThreadPool);

    void    SetupRange(Vector2ui rangeStart,
                       Vector2ui rangeEnd,
                       const GPUQueue& queue) override;

    void    GenerateNumbers(// Output
                            Span<RandomNumber> dNumbersOut,
                            // Constants
                            uint16_t dimensionStart,
                            RNRequestList rnRequests,
                            const GPUQueue& queue) const override;
    void    GenerateNumbersIndirect(// Output
                                    Span<RandomNumber> dNumbersOut,
                                    // Input
                                    Span<const RayIndex> dIndices,
                                    // Constants
                                    uint16_t dimensionStart,
                                    RNRequestList rnRequests,
                                    const GPUQueue& queue) const override;
    void    GenerateNumbersIndirect(// Output
                                    Span<RandomNumber> dNumbersOut,
                                    // Input
                                    Span<const RayIndex> dIndices,
                                    Span<const uint16_t> dDimensionStart,
                                    // Constants
                                    RNRequestList rnRequests,
                                    const GPUQueue& queue) const override;
    void    IncrementSampleId(const GPUQueue& queue) const override;
    void    IncrementSampleIdIndirect(Span<const RayIndex> dIndices,
                                      const GPUQueue& queue) const override;


    Span<BackupRNGState> GetBackupStates() override;
    size_t               GPUMemoryUsage() const override;
};

namespace RNGFunctions
{

// 64->64 variant used to hash 64 bit types
MR_HF_DEF constexpr
uint64_t HashPCG64::Hash(uint64_t v)
{
    constexpr uint64_t Multiplier = 6364136223846793005ull;
    constexpr uint64_t Increment = 1442695040888963407ull;
    uint64_t s = v * Multiplier + Increment;
    uint64_t word = ((s >> ((s >> 59u) + 5u)) ^ s);
    word *= 12605985483714917081ull;
    return (word >> 43u) ^ word;
}

template<class... Args>
requires(HashPCG64::Detail::IntCastable_v<Args> && ...)
MR_HF_DEF constexpr
uint64_t HashPCG64::Hash(Args&&... args)
{
    using namespace HashPCG64::Detail;
    // According to the paper recursive hash and add
    // is recommended.
    // Abusing comma operator expansion here,
    // I've learned this recently and wanted to use somewhere
    // This was somewhat a good place to use this.
    // This may improve compilation times when sizeof...(Args) is large?
    uint64_t v = 0;
    (
        static_cast<void>(v = Hash(std::bit_cast<uintX_t<Args>>(args) + v)),
        ...
    );
    return v;
}

}

template <std::unsigned_integral T>
MR_HF_DEF
RNGDispenserT<T>::RNGDispenserT(Span<const T> numbers,
                                uint32_t globalId,
                                uint32_t stride)
    : randomNumbers(numbers)
    , globalId(globalId)
    , stride(stride)
{}

template <std::unsigned_integral T>
template<uint32_t I>
MR_HF_DEF
Float RNGDispenserT<T>::NextFloat()
{
    assert((globalId + I * stride) < randomNumbers.size());
    uint32_t xi0 = randomNumbers[globalId + I * stride];

    return RNGFunctions::ToFloat01<Float>(xi0);
}

template <std::unsigned_integral T>
template<uint32_t I>
MR_HF_DEF
Vector2 RNGDispenserT<T>::NextFloat2D()
{
    assert((globalId + (I + 1) * stride) < randomNumbers.size());
    uint32_t xi0 = randomNumbers[globalId + (I    ) * stride];
    uint32_t xi1 = randomNumbers[globalId + (I + 1) * stride];

    return Vector2(RNGFunctions::ToFloat01<Float>(xi0),
                   RNGFunctions::ToFloat01<Float>(xi1));
}

MR_PF_DEF
uint32_t RNRequestList::DimensionOfRequest(uint32_t requestIndex) const
{
    uint32_t i = requestIndex;
    uint32_t result = (list >> (i * BIT_PER_REQUEST)) & MASK;
    return result;
}

MR_PF_DEF
uint32_t RNRequestList::TotalRNCount() const
{
    static_assert(BIT_PER_REQUEST == 2, "This implementation is "
                  "only for 2-bit dimension per request. "
                  "You need to change constants for that.");
    constexpr uint64_t LSB_MASK = 0x5555555555555555;
    constexpr uint64_t MSB_MASK = 0xAAAAAAAAAAAAAAAA;
    return uint32_t(Bit::PopC(MSB_MASK & list) * 2ull +
                    Bit::PopC(LSB_MASK & list));
}

MR_PF_DEF
uint32_t RNRequestList::TotalRequestCount() const
{
    assert((fillCounter % BIT_PER_REQUEST) == 0);
    return fillCounter / BIT_PER_REQUEST;
}

MR_PF_DEF
RNRequestList RNRequestList::Append(uint32_t dimOfRequest) const
{
    assert(fillCounter + 2 < BIT_LIMIT);
    assert(dimOfRequest <= MASK);

    RNRequestList result = (*this);
    if(dimOfRequest == 0) return result;

    result.list = (dimOfRequest << fillCounter) | list;
    result.fillCounter = fillCounter + 2u;
    return result;
}

MR_PF_DEF
RNRequestList RNRequestList::Append(const RNRequestList& other) const
{
    assert(other.fillCounter + fillCounter <= BIT_LIMIT);
    RNRequestList result = (*this);

    result.list = (other.list << fillCounter) | list;
    result.fillCounter = fillCounter + other.fillCounter;
    return result;
}

MR_PF_DEF_V
void RNRequestList::CombineMax(const RNRequestList& r)
{
    // Given "this" and another RN request list
    // find the maximum required random numbers that does not
    // alter the dimensionality of the samples.
    // Increase sample count if required
    uint32_t newFill = Math::Max(fillCounter, r.fillCounter);
    uint32_t reqCount = newFill / BIT_PER_REQUEST;

    RNRequestList newList;
    for(uint32_t i = 0; i < reqCount; i++)
    {
        uint32_t localMax = Math::Max(DimensionOfRequest(i),
                                      r.DimensionOfRequest(i));
        newList = newList.Append(localMax);
    }
    *this = newList;
}

template<uint32_t... Is>
MR_PF_DEF
RNRequestList GenRNRequestList()
{
    constexpr std::array UNPACK_LIST = {Is...};
    RNRequestList result;
    for(size_t i = 0; i < UNPACK_LIST.size(); i++)
        result = result.Append(UNPACK_LIST[i]);
    return result;
}

MR_HF_DEF
typename PermutedCG32::State
PermutedCG32::GenerateState(uint32_t seed)
{
    State r;
    r = Step(0);
    r += seed;
    r = Step(r);
    return r;
}

MR_HF_DEF
typename PermutedCG32::State
PermutedCG32::Step(State r)
{
    return r * Multiplier + Increment;
}

MR_HF_DEF
uint32_t PermutedCG32::Permute(State s)
{
    uint32_t result = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (result >> 22u) ^ result;
}

MR_HF_DEF
PermutedCG32::PermutedCG32(State& dState)
    : dState(dState)
    , rState(dState)
{}

MR_HF_DEF
PermutedCG32::~PermutedCG32()
{
    dState = rState;
}

MR_HF_DEF
uint32_t PermutedCG32::Next()
{
    State newState = Step(rState);
    uint32_t r = Permute(rState);
    rState = newState;
    return r;
}

MR_HF_DEF
Float PermutedCG32::NextFloat()
{
    uint32_t nextInt = Next();
    return RNGFunctions::ToFloat01<Float>(nextInt);
}

MR_HF_DEF
void PermutedCG32::Advance(uint32_t delta)
{
    // This is from the PCG32 code,
    // which is based on
    // Brown, "Random Number Generation
    // with Arbitrary Stride", Transactions of the American Nuclear
    // Society (Nov. 1994).
    //
    // Upon inspection, this may not be useful for advancing the state on a warp
    // local fashion since it is log(N) operations
    // We may want to advance for warp size
    // log(32) = 5 iterations, then do 32 i
    uint32_t accMult = 1u;
    uint32_t accPlus = 0u;
    uint32_t curMult = Multiplier;
    uint32_t curPlus = Increment;
    while(delta > 0)
    {
        if(delta & 0b1)
        {
            accMult *= curMult;
            accPlus = accPlus * curMult + curPlus;
        }
        curPlus = (curMult + 1u) * curPlus;
        curMult *= curMult;
        delta /= 2;
    }
    rState = accMult * rState + accPlus;
}