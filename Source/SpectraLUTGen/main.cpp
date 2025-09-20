// This exectuable more or less follows
// https://github.com/mitsuba-renderer/rgb2spec
// and paper
// https://rgl.epfl.ch/publications/Jakob2019Spectral
//
#include <span>
#include <latch>
#include <thread>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
// Native format c++20, we do not use it normally although
// our baseline cpp standard is 20. fmt lib is the baseline
// of the standard implementation and it is stil maintained.
//
#include <format>

// From CoreLib we can only use the header only portion
#include "Core/Definitions.h"
#include "Core/Vector.h"
#include "Core/Span.h"
#include "Core/Math.h"
#include "Core/LinearAlg.h"
// Color functions have some static data, we will compile it
// internally
#include "Core/ColorFunctions.h"

using ColorspaceList = Tuple
<
    Color::Colorspace<MRayColorSpaceEnum::MR_ACES2065_1>,
    Color::Colorspace<MRayColorSpaceEnum::MR_ACES_CG>,
    Color::Colorspace<MRayColorSpaceEnum::MR_REC_709>,
    Color::Colorspace<MRayColorSpaceEnum::MR_REC_2020>,
    Color::Colorspace<MRayColorSpaceEnum::MR_DCI_P3>,
    Color::Colorspace<MRayColorSpaceEnum::MR_ADOBE_RGB>
>;

using FloatType = double;
using Vec3F = Vector<3, FloatType>;
using Mat3F = Matrix<3, FloatType>;

inline constexpr uint32_t TABLE_COUNT = 3;
inline constexpr uint32_t CHANNEL_COUNT = 3;

struct Error
{
    bool hasError = false;
    std::string errorInfo;
};


struct RGBToXYZMatrixPair
{
    Mat3F rgbToXYZ;
    Mat3F xyzToRGB;
};

RGBToXYZMatrixPair
FetchConversionMatrices(MRayColorSpaceEnum globalColorSpace)
{
    uint32_t i = static_cast<uint32_t>(globalColorSpace);
    constexpr ColorspaceList CS = {};

    RGBToXYZMatrixPair result;
    InvokeAt(i, CS, [&result](auto CS)
    {
        static constexpr auto E = decltype(CS)::Enum;
        static constexpr auto Primaries = Color::Colorspace<E>::Prims;
        static constexpr Matrix3x3 toXYZ = Color::GenRGBToXYZ(Primaries);
        static constexpr Matrix3x3 fromXYZ = toXYZ.Inverse();
        result = RGBToXYZMatrixPair
        {
            .rgbToXYZ = toXYZ,
            .xyzToRGB = fromXYZ
        };
        return true;
    });
    return result;
}

Error PassGenSpectraToRGB(// Output
                          Span<Vec3F> spectraToRGB,
                          Vec3F& whitepoint,
                          // Input
                          Span<const Vector3> cie1931_XYZ,
                          Span<const Float> illuminantSPD,
                          // Constants
                          const RGBToXYZMatrixPair& convMatrices,
                          Float illuminantSPDNormFactor,
                          Vector2ui passRange)
{
     using namespace Color;
    // Data fetch etc is more direct which assumes
    // index i directly corresponds to a data point,
    // no interpolation needed.
    static_assert(CIE_1931_DELTA == Float(1), "Spectrum interval must be one!");
    static_assert(CIE_1931_N % 3 == uint32_t(0),
                  "While integrating we use Simpson's 3/8 rule, which requires "
                  "the data to be evenly divisible by 3");
    auto Weight = [](uint32_t i) -> FloatType
    {
        constexpr FloatType W = FloatType(3) / FloatType(8) * FloatType(CIE_1931_DELTA);
        // Integral weight pattern of Simpson's 3/8 rule is
        // 1 | 3 | 3 | 2 | 3 | 3 | 2 | ... | 2 | 3 | 3 | 1
        bool isEdge = (i == (CIE_1931_N - 1) || i == 0);
        //
             if(isEdge)           return W;
        else if((i - 1) % 3 == 2) return W * FloatType(2);
        else                      return W * FloatType(3);
    };

    // Sanity checks
    assert(cie1931_XYZ.size() == CIE_1931_N);
    assert(illuminantSPD.size() == CIE_1931_N);
    assert(spectraToRGB.size() == CIE_1931_N);
    // Colorspace "ToXYZ" function also converts to a common whitepoint
    // (D65), but here we don't want I guess? So we just generate the
    // matrix ourselves and multiply.
    const Mat3F& XYZToRGB = convMatrices.xyzToRGB;

    // Grid stride loop
    for(uint32_t i = passRange[0]; i < passRange[1]; i++)
    {
        Vec3F xyz = Vec3F(cie1931_XYZ[i]);
        FloatType weight = Weight(i);
        FloatType I = FloatType(illuminantSPD[i]) / FloatType(illuminantSPDNormFactor);
        spectraToRGB[i] = (XYZToRGB * xyz) * I * weight;

        Vec3F wpFraction = xyz * I * weight;
        std::atomic_ref(whitepoint[0]).fetch_add(wpFraction[0]);
        std::atomic_ref(whitepoint[1]).fetch_add(wpFraction[1]);
        std::atomic_ref(whitepoint[2]).fetch_add(wpFraction[2]);
    }
    return Error{.hasError = false};
}

auto OptimizePolynomial(const Vec3F& rgb,
                        const Vec3F& initialGuess,
                        Span<const Vec3F>& dSpectraToRGB,
                        const Vec3F& whitepoint,
                        const RGBToXYZMatrixPair& convMatrices,
                        uint32_t optimizePassCount)
{
    using namespace Color;
    struct Result
    {
        Vec3F     coeffs;
        FloatType error;
    };

    auto EvaluateResidual = [&](Vec3F coeffs)
    {
        Vec3F integralResult = Vec3F::Zero();
        for(uint32_t i = 0; i < CIE_1931_N; i++)
        {
            constexpr FloatType CIE_1931_NORM_FACTOR = FloatType(1) / FloatType(CIE_1931_N);
            // This
            //FloatType lambda = CIE_1931_RANGE[0] + CIE_1931_DELTA * FloatType(i);
            //FloatType lambdaN = (lambda - CIE_1931_RANGE[0]) * CIE_1931_NORM_FACTOR;
            // will reduce to this:
            FloatType lambdaN = FloatType(i) * CIE_1931_DELTA * CIE_1931_NORM_FACTOR;
            //
            FloatType x = coeffs[0];
            x = x * lambdaN + coeffs[1];
            x = x * lambdaN + coeffs[2];

            // Sigmoid
            FloatType s = FloatType(0.5) * x;
            s /= Math::Sqrt(FloatType(1) + x * x);
            s += FloatType(0.5);
            //
            integralResult += dSpectraToRGB[i] * s;
        }
        using Color::XYZToCIELab;
        // Colorspace "ToXYZ" function also converts to a common whitepoint
        // (D65), but here we don't want I guess? So we just generate the
        // matrix ourselves and multiply.
        const Mat3F& RGBToXYZ = convMatrices.rgbToXYZ;

        // TODO: We will lose precision here when FloatType is double.
        // We only have generic "Float" (which is "float" by default)
        // functions in Color namespace. Check if this reduction of accuracy
        // is unacceptable.
        Vec3F diff = XYZToCIELab<FloatType>(RGBToXYZ * integralResult, whitepoint);
        Vec3F residual = XYZToCIELab<FloatType>(RGBToXYZ * rgb, whitepoint);
        return residual - diff;
    };
    //
    Vec3F coeffs = initialGuess;
    FloatType error = std::numeric_limits<FloatType>::max();
    for(uint32_t _ = 0; _ < optimizePassCount; _++)
    {
        // ==================== //
        //  Evaluate Residual   //
        // ==================== //
        Vec3F residual = EvaluateResidual(coeffs);

        // ==================== //
        //  Evaluate Jacobian   //
        // ==================== //
        Matrix<3, FloatType> jacobian;
        for(uint32_t i = 0; i < 3; i++)
        {
            // The paper had static epsilon here,
            // Float precision was not enough to get plausible results.
            // I've pinpointed the problem that the static EPSILON is the culprit.
            // With double precision, 1e-4 gives similar/same results compared to
            // paper's implementation.
            // With single precision, it does not give proper results (NaNs galore)
            // if we reduce it to 1e-2, 90% results are not NaN (NaN ones are very
            // dark rgb values (i.e. 0.0002, 0.0000, 0.0001))
            //
            // So one precaution for this is to dynamically adjust the
            // finite difference delta value according to the Float type.
            constexpr FloatType EPSILON = []()
            {
                if constexpr(std::is_same_v<float, FloatType>)
                    return FloatType(1e-2);
                else
                    return FloatType(1e-4);
            }();

            Vec3F r0 = coeffs; r0[i] -= EPSILON;
            r0 = EvaluateResidual(r0);

            Vec3F r1 = coeffs; r0[i] += EPSILON;
            r1 = EvaluateResidual(r1);

            constexpr FloatType Factor = FloatType(0.5) / EPSILON;
            jacobian(0, i) = (r1[0] - r0[0]) * Factor;
            jacobian(1, i) = (r1[1] - r0[1]) * Factor;
            jacobian(2, i) = (r1[2] - r0[2]) * Factor;
        }

        // ===================== //
        // Solve Matrix Equation //
        // ===================== //
        // A * x = y, find y.
        {
            using namespace LinearAlg;
            LUPResult<3, FloatType> luResult = LUDecompose(jacobian);
            assert(luResult.success);
            Vec3F y = residual;
            Vec3F x = SolveWithLU(luResult, y);
            coeffs -= x;
        }
        //
        // TODO: This probably numerically unstable thus;
        // paper does some LU decomposition stuff.
        // Change to LU decomposition later.
        //{
        //    Vec3F y = residual;
        //    Vec3F x = jacobian.Inverse() * y;
        //    coeffs -= x;
        //}
        // TODO: What is this? Is this numeric adjustment?
        FloatType max = coeffs[coeffs.Maximum()];
        if(max > FloatType(200))
            coeffs *= FloatType(200) / max;

        // Early terminate if result is fine
        error = (residual * residual).Sum();
        if(error < MathConstants::SmallEpsilon<FloatType>())
            break;
    }
    return Result{coeffs, Math::Sqrt(error)};
}

Error PassGenerateSpectrumLUT(// Output
                              Span<Float> outputLUT,
                              // Input
                              Span<const Vec3F> spectraToRGB,
                              const Vec3F& whitepoint,
                              // Constants
                              const RGBToXYZMatrixPair& convMatrices,
                              const Vector3ui& resolution,
                              uint32_t optimizePassCount,
                              Vector2ui passRange)
{
    assert(spectraToRGB.size() == Color::CIE_1931_N);
    using namespace Color;
    // We define a special struct since std::array disregards alignment (why?)
    // thus; CUDA give misalined read exception
    struct LUTResult
    {
        Vec3F outCoeffs;
        Vec3F nextGuess;
        Error error;
    };
    //
    uint32_t xSize = resolution[0];
    uint32_t xySize = xSize * resolution[1];
    uint32_t xyzSize = xySize * resolution[2];

    auto ProcessLUTEntry = [&](Vector3ui ijk, uint32_t l,
                               const Vec3F& initialGuess) -> LUTResult
    {
        auto SmoothStep = [](FloatType t)
        {
            return t * t * (FloatType{3} - FloatType{2} * t);
        };

        // TODO: Paper divides as such to normalize, which is not correct
        // if we are going to sample this data via textures. We need to adjust it so that
        // direct normalized & interpolated access is possible.
        Vec3F xyz = Vec3F(ijk) / Vec3F(resolution - Vector3ui(1));
        // so called "max pixel" index (l0) other two is next indices in rotating order
        uint32_t l0 = l;
        uint32_t l1 = (l0 + 1) % 3;
        uint32_t l2 = (l1 + 1) % 3;
        // Actual calculation
        FloatType b = SmoothStep(SmoothStep(xyz[2]));
        Vec3F rgb;
        rgb[l0] = b;
        rgb[l1] = xyz[0] * b;
        rgb[l2] = xyz[1] * b;

        auto [c, e] = OptimizePolynomial
        (
            rgb, initialGuess,
            spectraToRGB,
            whitepoint,
            convMatrices,
            optimizePassCount
        );

        if(!Math::IsFinite(c))
        {
            using namespace std::string_view_literals;
            std::string errInfo = std::format
            (
                "Unable to optimize polynomial for l: {}, ijk: {} => "
                "Coeffs: {}, Residual: {}, Mode: {:s}",
                l, ijk.AsArray(), c.AsArray(), e,
                (std::is_same_v<FloatType, double>)
                ? "Double"sv : "Float"sv
            );
            return LUTResult
            {
                .error =
                {
                    .hasError = true,
                    .errorInfo = std::move(errInfo)
                }
            };
        }

        //printf("[%u|%u|%u|%u][%4f, %4f, %4f] ::: [%5f, %5f, %5f], %f\n",
        //       l, ijk[0], ijk[1], ijk[2],
        //       rgb[0], rgb[1], rgb[2],
        //       c[0], c[1], c[2], e);

        constexpr FloatType CIE_1931_START_LAMBDA = FloatType(CIE_1931_RANGE[0]);
        constexpr FloatType p0 = CIE_1931_START_LAMBDA;
        constexpr FloatType p1 = FloatType(1) / FloatType(CIE_1931_N - 1);
        Vec3F outCoeffs;
        outCoeffs[0] = c[0] * p1 * p1;
        outCoeffs[1] = c[1] * p1 - FloatType(2) * c[0] * p0 * p1 * p1;
        outCoeffs[2] = c[2] - c[1] * p0 * p1 + c[0] * p0 * p1 * p0 * p1;

        return LUTResult{outCoeffs, c};
    };
    auto WriteToOutputBuffer = [&](Vector3ui ijk, uint32_t l,
                                   const Vec3F& result) -> void
    {
        // We separate channels since we will load it into seperate textures
        // Float3 texture will have an extra float padding due to HW limitations.
        constexpr uint32_t CHANNELS = 3;
        uint32_t globalOffset = l      * xyzSize * CHANNELS + // LUT
                                ijk[2] * xySize             + // Z
                                ijk[1] * xSize              + // Y
                                ijk[0];                       // X

        uint32_t rOffset = globalOffset + 0 * xyzSize;
        uint32_t gOffset = globalOffset + 1 * xyzSize;
        uint32_t bOffset = globalOffset + 2 * xyzSize;
        outputLUT[rOffset] = Float(result[0]);
        outputLUT[gOffset] = Float(result[1]);
        outputLUT[bOffset] = Float(result[2]);
    };

    // ======================== //
    //     Actual Work Start    //
    // ======================== //
    // According to the paper, each k's initial guess depends on pevious calculation
    // Thus; we can only allocate single thread for each "row" of data.
    for(uint32_t tid = passRange[0]; tid < passRange[1]; tid++)
    {
        // Pixel(Voxel) indices of LUT
        // k will be generated later due to initial guess relation
        uint32_t i, j;
        j = (tid % xySize) / xSize;
        i = (tid % xSize);
        //
        uint32_t l = tid / xySize;
        // Moderately bright color (from the implementation)
        static constexpr uint32_t EXPAND_POINT = 5;
        uint32_t mid = resolution[2] / EXPAND_POINT;

        Vec3F curGuess = Vec3F::Zero();
        Vec3F middleGuess = Vec3F::Zero();
        for(uint32_t k = mid; k < resolution[2]; k++)
        {
            Vector3ui ijk = Vector3ui(i, j, k);
            auto rVal = ProcessLUTEntry(ijk, l, curGuess);
            //
            if(rVal.error.hasError) return rVal.error;
            //
            const auto& [result, nextGuess, _] = rVal;
            WriteToOutputBuffer(ijk, l, result);

            // Save middle for the middle to zero iteration
            if(k == mid) middleGuess = nextGuess;

            curGuess = nextGuess;
        }
        // Restart initial guess to the middle
        curGuess = middleGuess;
        for(int32_t k = (mid - 1); k >= 0; k--)
        {
            Vector3ui ijk = Vector3ui(i, j, k);
            auto rVal = ProcessLUTEntry(ijk, l, curGuess);
            //
            if(rVal.error.hasError) return rVal.error;
            //
            const auto& [result, nextGuess, _] = rVal;

            WriteToOutputBuffer(ijk, l, result);
            curGuess = nextGuess;
        }
    }
    return Error{};
}

void THRDEntryPoint(// Output
                    Error& potentialError,
                    Span<Float> outputLUT,
                    // Temp
                    Vec3F& whitepoint,
                    Span<Vec3F> spectraToRGB,
                    // Input
                    Span<const Vector3> cie1931_XYZ,
                    Span<const Float> illuminantSPD,
                    // Constants
                    const RGBToXYZMatrixPair& convMatrices,
                    Float illuminantSPDNormFactor,
                    Vector3ui resolution,
                    uint32_t optimizePassCount,
                    // Processing Related
                    std::latch& passBarrier,
                    Vector2ui spectraToRGBPassRange,
                    Vector2ui lutPassRange)
{
    potentialError = PassGenSpectraToRGB(spectraToRGB, whitepoint, cie1931_XYZ,
                                         illuminantSPD, convMatrices,
                                         illuminantSPDNormFactor, spectraToRGBPassRange);
    //
    passBarrier.arrive_and_wait();
    //
    potentialError = PassGenerateSpectrumLUT(outputLUT,
                                             ToConstSpan(spectraToRGB),
                                             whitepoint, convMatrices,
                                             resolution, optimizePassCount,
                                             lutPassRange);
}

Error GenerateSpectraLUT(Span<Float> outputLUT,
                         MRayColorSpaceEnum colorSpace,
                         Vector3ui lutResolution)
{
    assert(outputLUT.size() == lutResolution.Multiply() * TABLE_COUNT * CHANNEL_COUNT);
    // 32-bit float does not cut it here,
    // optimization problem + integration (piecewise function) does not give
    // satisfying results. So we instantiate kernels according to float type
    // to check / validate different float types.
    using FloatType = double;
    using Vec3F = Vector<3, FloatType>;

    // Inputs
    Span<const Vector3> hCIE1931_XYZ = Span<const Vector3>(Color::CIE_1931_XYZ);
    Span<const Float> hIlluminantSPD = Span<const Float>(Color::SelectIlluminantSPD(colorSpace));
    Float hIlluminantNormFactor = Color::SelectIlluminantSPDNormFactor(colorSpace);
    RGBToXYZMatrixPair convMatrices = FetchConversionMatrices(colorSpace);

    // Temp data
    std::vector<Vec3F> spectraToRGB(Color::CIE_1931_N);
    Span<Vec3F> hSpectraToRGB = spectraToRGB;
    Vec3F hWhitepoint = Vec3F::Zero();

    // Threads and Launch
    uint32_t threadCount = std::thread::hardware_concurrency();
    std::vector<Error> threadLocalErrors(threadCount);
    std::vector<std::jthread> threads;
    //
    uint32_t lutWorkCount = lutResolution[0] * lutResolution[1] * TABLE_COUNT;
    uint32_t spectraToRGBWPT = Math::DivideUp(Color::CIE_1931_N, threadCount);
    uint32_t lutWPT = Math::DivideUp(lutWorkCount, threadCount);
    auto passBarrier = std::latch(threadCount);
    for(uint32_t i = 0; i < threadCount; i++)
    {
        Vector2ui spectraToRGBRange = Vector2ui(i, i + 1) * spectraToRGBWPT;
        spectraToRGBRange[1] = std::min(spectraToRGBRange[1], Color::CIE_1931_N);

        Vector2ui lutRange = Vector2ui(i, i + 1) * lutWPT;
        lutRange[1] = std::min(lutRange[1], lutWorkCount);

        threads.emplace_back(THRDEntryPoint,
                             // Outputs
                             std::ref(threadLocalErrors[i]),
                             outputLUT,
                             // Temp
                             std::ref(hWhitepoint),
                             hSpectraToRGB,
                             // Input
                             hCIE1931_XYZ,
                             hIlluminantSPD,
                             // Constants
                             std::ref(convMatrices),
                             hIlluminantNormFactor,
                             lutResolution,
                             15u,
                             // Processing Related
                             std::ref(passBarrier),
                             spectraToRGBRange,
                             lutRange);
    }
    //
    threads.clear();

    Error result = {};
    for(const auto& e : threadLocalErrors)
    {
        if(e.hasError)
        {
            result.hasError = true;
            result.errorInfo += " || " + e.errorInfo;
        }
    }
    return result;
}

int main(int argc, const char* argv[])
{
    uint32_t argcUInt = static_cast<uint32_t>(argc);
    static constexpr uint32_t MAX_ARG_COUNT = 3;
    if(argcUInt != MAX_ARG_COUNT + 1)
    {
        std::cerr << std::format("Wrong Argument Count({})", argcUInt);
        return 1;
    }
    std::array<std::string_view, MAX_ARG_COUNT> args;
    for(uint32_t i = 0; i < std::min(argcUInt, MAX_ARG_COUNT); i++)
        args[i] = argv[i + 1];

    uint32_t resolution = 0;
    if(std::from_chars(args[0].data(), args[0].data() + args[0].size(),
                       resolution).ec != std::errc())
    {
        std::cerr << std::format("1st arg is not a number. ({})", args[1]);
        return 1;
    };
    std::string_view colorSpaceString = args[1];

    auto colorSpace = MRayColorSpaceStringifier::FromString(colorSpaceString);
    if(colorSpace == MRayColorSpaceEnum::MR_END)
    {
        std::cerr << std::format("Unknown color space name ({})", colorSpaceString);
        return 1;
    };

    namespace fs = std::filesystem;
    std::string_view outputFolder = args[2];
    if(fs::create_directories(outputFolder))
    {
        std::cerr << std::format("Unable to create path towards \"{}\"",
                                 outputFolder);
        return 1;
    }

    //
    Vector3ui lutSize = Vector3ui(resolution);
    std::vector<Float> outputLUTData(TABLE_COUNT * CHANNEL_COUNT *
                                     lutSize.Multiply());
    Span<Float> outputLUT = Span<Float>(outputLUTData);

    Error e = GenerateSpectraLUT(outputLUT, colorSpace, lutSize);
    if(e.hasError)
    {
        std::cerr << e.errorInfo;
        return 1;
    }

    // Finally write
    using namespace std::string_view_literals;
    auto outFilePath = (fs::path(outputFolder) /
                        fs::path(std::string(colorSpaceString) +
                                 std::string(Color::LUT_FILE_EXT)));
    outFilePath = fs::absolute(outFilePath);

    std::ofstream outFile(outFilePath, std::ios::binary);
    if(!outFile)
    {
        std::cerr << std::format("Unable to open {}", outFilePath.string());
        return 1;
    }

    outFile << Color::LUT_FILE_CC;
    outFile.write(reinterpret_cast<const char*>(&resolution), sizeof(uint32_t));
    // Reserved, for data  type, 0 means half, 1 means float, 2 means double
    uint32_t mode = 1;
    outFile.write(reinterpret_cast<const char*>(&mode), sizeof(uint32_t));
    // The actual data
    outFile.write(reinterpret_cast<const char*>(outputLUT.data()),  outputLUT.size_bytes());
    // All Done!

    //for(uint32_t l = 0; l < 3; l++)
    //for(uint32_t k = 0; k < 10; k++)
    //for(uint32_t j = 0; j < 10; j++)
    //for(uint32_t i = 0; i < 10; i++)
    //{
    //    int idx = ((l * resolution + k) * resolution + j) * resolution + i;

    //    auto r0 = outputLUT[3 * idx + 0];
    //    auto r1 = outputLUT[3 * idx + 1];
    //    auto r2 = outputLUT[3 * idx + 2];
    //    printf("[%u][%u, %u, %u] = %.9f, %.9f, %.9f\n",
    //           l, i, j, k, r0, r1, r2);
    //}


    return 0;
}