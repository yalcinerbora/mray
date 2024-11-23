#version 300 es
//
// Single Pass Path Tracing of
// Classic Cornel Box Scene
//
// Cornell Box is embedded into the shader
// as 3 unit (vec3(-1), vec3(1)) AABBs
//
//
precision highp float;
precision highp int;

// ============================= //
//           LITERALS            //
// ============================= //
const uint PCG32_Multiplier = 747796405u;
const uint PCG32_Increment = 2891336453u;

const mat4 TransformsAABB[4] = mat4[4]
(
    // Outer box
    mat4( 1,  0,  0, 0,
          0,  1,  0, 0,
          0,  0,  1, 0,
          0,  1,  0, 1),
    // Tall Box
    mat4( 0.2867760f, 0.0f, -0.099798f, 0,
          0.098229f,  0.0f,  0.282266f, 0,
               0.0f, -0.6f,       0.0f, 0,
         -0.335439f,  0.6f, -0.291415f, 1),

    // Short Box
    mat4(0.085164f,  0.0f, -0.284951f, 0,
         0.289542f,  0.0f,  0.086536f, 0,
              0.0f, -0.3f,       0.0f, 0,
         0.328631f,  0.3f,  0.374592f, 1),
    // Light
    mat4(0.25f, 0.000f, 0.00f, 0,
         0.00f, 0.010f, 0.00f, 0,
         0.00f, 0.000f, 0.25f, 0,
         0.00f, 1.998f, 0.00f, 1)
);

// TODO: Check if these are literals or every shader invocation calculates
// these (which will be slow).
const mat4 InvTransformsAABB[4] = mat4[4]
(
    // Outer box
    inverse(TransformsAABB[0]),
    // Tall Box
    inverse(TransformsAABB[1]),
    // Short Box
    inverse(TransformsAABB[2]),
    // Light
    inverse(TransformsAABB[3])
);

const vec3 Colors[3] = vec3[3]
(
    // Red
    vec3(0.6300, 0.0650, 0.0500),
    // Green
    vec3(0.14, 0.45, 0.091),
    // Gray
    vec3(0.725, 0.71, 0.68)
);

const vec3 LightRadiance = vec3(6.8, 4.8, 1.6);

// ============================= //
//            STRUCTS            //
// ============================= //
struct Hit
{
    vec3 albedoOrRadiance;
    vec3 N;
    bool isLight;
    bool isMissed;
    float tMax;
};

struct RayPack
{
    vec3 rDir;
    vec3 rPos;
    vec2 tMM;
};

// ============================= //
//           UNIFORMS            //
// ============================= //
uniform vec2 uResolution;
uniform uint uFrame;
uniform vec3 uBGColor;
uniform vec3 uCamPos;

// These are not required to be changed from CPU
const vec3 uCamGaze = vec3(0, 1, 0);
const vec3 uCamUp   = vec3(0, 1, 0);
const float fovX    = 22.5 * 3.1415 / 180.0; // Radians
const vec2 nearFar  = vec2(0.05, 200);

// ============================= //
//          FUNCTIONS            //
// ============================= //
void Swap(inout float a, inout float b)
{
    float t = a; a = b; b = t;
}

vec4 IntersectsUnitAABB(in vec3 rDir, in vec3 rPos, in vec2 tMM)
{
    vec3 invD = vec3(1) / rDir;
    vec3 t0 = (vec3(-1) - rPos) * invD;
    vec3 t1 = (vec3(+1) - rPos) * invD;
    vec2 tOut = tMM;

    vec2 n = vec2(-1);
    for(int i = 0; i < 3; i++)
    {
        if(invD[i] < 0.0f) Swap(t0[i], t1[i]);
        vec2 newT = vec2(max(tOut[0], min(t0[i], t1[i])),
                         min(tOut[1], max(t0[i], t1[i])));

        // Calcualte the axis of intersection
        if(newT[0] != tOut[0])
        {
            tOut[0] = newT[0];
            n[0] = float(i);
        }
        if(newT[1] != tOut[1])
        {
            tOut[1] = newT[1];
            n[1] = float(i);
        }
    }
    if(tOut[1] >= tOut[0])
        return vec4(tOut, n);
    else
        return vec4(vec2(1e20, 1e20), n);
}

vec3 CalculateNormal(in uint axis, in uint tIndex, in vec3 dir)
{
    vec3 normal = vec3(0);
    normal[axis] = 1.0f;
    // Normal matrix multiplication (Inv Transpose of the matrix t is
    // left multiplication of inv transform matrix
    normal = (vec4(normal, 0.0f) * InvTransformsAABB[tIndex]).xyz;
    normal = normalize(normal);

    if(dot(normal, dir) > 0.0f) normal = -normal;
    return normal;
}

Hit IntersectScene(in RayPack ray)
{
    vec4 tMM0 = IntersectsUnitAABB((InvTransformsAABB[0] * vec4(ray.rDir, 0)).xyz,
                                   (InvTransformsAABB[0] * vec4(ray.rPos, 1)).xyz,
                                   ray.tMM);
    vec4 tMM1 = IntersectsUnitAABB((InvTransformsAABB[1] * vec4(ray.rDir, 0)).xyz,
                                   (InvTransformsAABB[1] * vec4(ray.rPos, 1)).xyz,
                                   ray.tMM);
    vec4 tMM2 = IntersectsUnitAABB((InvTransformsAABB[2] * vec4(ray.rDir, 0)).xyz,
                                   (InvTransformsAABB[2] * vec4(ray.rPos, 1)).xyz,
                                   ray.tMM);
    vec4 tMM3 = IntersectsUnitAABB((InvTransformsAABB[3] * vec4(ray.rDir, 0)).xyz,
                                   (InvTransformsAABB[3] * vec4(ray.rPos, 1)).xyz,
                                   ray.tMM);
    float tMax = min(min(tMM0[1], tMM1[0]),
                     min(tMM2[0], tMM3[0]));
    tMax = min(tMax, ray.tMM[1]);

    Hit h;
    h.isLight = false;
    h.isMissed = false;
    h.tMax = tMax;
    if(tMax == tMM0[1])
    {
        h.N = CalculateNormal(uint(tMM0[3]), 0u, ray.rDir);

        if(tMM0[3] == 0.0f && h.N[0] == 1.0f)
            h.albedoOrRadiance = Colors[0];
        else if(tMM0[3] == 0.0f && h.N[0] == -1.0f)
            h.albedoOrRadiance = Colors[1];
        else
            h.albedoOrRadiance = Colors[2];
    }
    else if(tMax == tMM1[0])
    {
        h.N = CalculateNormal(uint(tMM1[2]), 1u, ray.rDir);
        h.albedoOrRadiance = Colors[2];
    }
    else if(tMax == tMM2[0])
    {
        h.N = CalculateNormal(uint(tMM2[2]), 2u, ray.rDir);
        h.albedoOrRadiance = Colors[2];
    }
    else if(tMax == tMM3[0])
    {
        h.N = CalculateNormal(uint(tMM3[2]), 3u, ray.rDir);
        h.albedoOrRadiance = LightRadiance;
        h.isLight = true;
    }
    else
    {
        h.albedoOrRadiance = uBGColor;
        //h.albedoOrRadiance = vec3(100);
        h.isMissed = true;
    }
    return h;
}

RayPack CamRayGen(in vec2 fragCoord, in vec2 xi)
{
    // Find world space window sizes
    float fovY = uResolution.y / uResolution.x * fovX;
    float widthHalf = tan(fovX * 0.5f) * nearFar[0];
    float heightHalf = tan(fovY * 0.5f) * nearFar[0];

    // Camera Vector Correction
    vec3 gazeDir = uCamGaze - uCamPos;
    vec3 right = normalize(cross(gazeDir, uCamUp));
    vec3 up = normalize(cross(right, gazeDir));
    gazeDir = normalize(cross(up, right));

    // Camera parameters
    vec3 bottomLeft = uCamPos;
    bottomLeft -= right   * widthHalf;
    bottomLeft -= up      * heightHalf;
    bottomLeft += gazeDir * nearFar[0];

    vec2 planeSize = vec2(widthHalf, heightHalf) * 2.0f;
    vec2 delta = planeSize / uResolution;
    vec2 jitter = xi;

    vec2 sampleDist = (fragCoord + jitter) * delta;
    vec3 samplePoint = bottomLeft;
    samplePoint += sampleDist[0] * right;
    samplePoint += sampleDist[1] * up;

    RayPack ray;
    ray.rDir = normalize(samplePoint - uCamPos);
    ray.rPos = uCamPos;
    ray.tMM = nearFar;
    return ray;
}

vec3 SampleCosDirection(in vec2 xi, vec3 N)
{
    // Generated direction is on unit space (+Z oriented hemisphere)
    float xi1Angle = 2.0 * 3.1415 * xi[1];
    float xi0Sqrt = sqrt(xi[0]);

    vec3 dir;
    dir[0] = xi0Sqrt * cos(xi1Angle);
    dir[1] = xi0Sqrt * sin(xi1Angle);
    dir[2] = max(0.0, sqrt(1.0 - dot(dir.xy, dir.xy)));

    // Align with the normal (dir is in tangent space)
    // Edge cases
    if(N[2] == 1.0)
        return dir;
    if(N[2] == -1.0)
        return -dir;

    //vec3 k = normalize(cross(dir, N));
    vec3 k = normalize(vec3(-N[1], N[0], 0.0));
    vec3 v = dir;
    float cosTheta = -N[2];
    float sinTheta = max(0.0, sqrt(1.0 - cosTheta * cosTheta));
    vec3 r = v * cosTheta + cross(k, v) * sinTheta;
    r += k * dot(k, v) * (1.0 - cosTheta);
    return normalize(r);
}

// To make the shader stateless, we create a random
// state for every invokation of every pixel.
//
// TODO: This may make the RNG a low quality one maybe?
//
uint PCG32_State(in vec2 fragCoord)
{
    // Hash a state from fragCoord, and Frame Index
    uint seed = uint(uResolution.x) * uint(uResolution.y) * uint(uFrame);
    seed += uint(uResolution.y) * uint(fragCoord.y);
    seed += uint(fragCoord.x);
    //
    uint state = seed + PCG32_Increment;
    state = PCG32_Multiplier * state + PCG32_Increment;
    return state;
}

uint PCG32_Next(inout uint s)
{
    // Churn the state
    uint newState = PCG32_Multiplier * s + PCG32_Increment;
    // Permutation
    uint r = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    r = (r >> 22u) ^ r;
    //
    s = newState;
    return r;
}

vec2 PCG32_Vec2(inout uint state)
{
    uint i0 = PCG32_Next(state);
    uint i1 = PCG32_Next(state);

    // https://marc-b-reynolds.github.io/distribution/2017/01/17/DenseFloat.html
    // GLSL does not accept hex decimals so "0x1p-24f = 5.9604645e-8f"
    float v0 = float(i0 >> 8u) * 5.9604645e-8f;
    float v1 = float(i1 >> 8u) * 5.9604645e-8f;
    return vec2(v0, v1);
}

vec3 mainImage(in vec2 fragCoord)
{
    uint state = PCG32_State(fragCoord);
    vec3 fragColor;

    // Initial ray
    RayPack ray = CamRayGen(fragCoord, PCG32_Vec2(state));

    // Four bounce
    vec3 radiance = vec3(0);
    vec3 throughput = vec3(1);
    for(int i = 0; i < 6; i++)
    {
        Hit hit = IntersectScene(ray);

        if(hit.isLight || hit.isMissed)
        {
            radiance = throughput * hit.albedoOrRadiance;
            break;
        }

        // Calculate w0
        vec3 w0 = SampleCosDirection(PCG32_Vec2(state), hit.N);
        ray.rPos += ray.rDir * hit.tMax;
        ray.rPos += hit.N * 1e-4f;
        ray.rDir = w0;
        ray.tMM = vec2(0, 200);

        throughput *= hit.albedoOrRadiance;
    }
    return radiance;
}

// ============================= //
//     FRAGMENT SHADER ENTRY     //
// ============================= //
uniform highp usampler2D tRadiance;
in vec2 fUV;

layout(location = 0)
out uvec4 fboColor;

void main(void)
{
    uvec4 radianceInt = texture(tRadiance, fUV);
    uint sampleCount = radianceInt.a;
    vec3 radiance = vec3(radianceInt.rgb) * vec3(1.525902189669642e-5);
    //
    vec2 fragCoord = fUV * vec2(uResolution);
    vec3 outColor = mainImage(fragCoord);

    // Combine samples
    vec3 outRadiance = radiance * float(sampleCount) + outColor;
    uint outSampleCount = sampleCount + uint(1);
    outRadiance /= float(outSampleCount);

    uvec4 outRadianceInt = uvec4(uvec3(outRadiance * vec3(65535.0)), outSampleCount);

    fboColor = outRadianceInt;
}