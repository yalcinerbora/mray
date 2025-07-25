#pragma once

#include "Core/Types.h"
#include "Tracer/Hit.h"
#include "Tracer/TracerTypes.h"
#include "Tracer/Random.h"
#include "Tracer/AcceleratorC.h"

#include <optix_device.h>

enum class RenderModeOptiX
{
	NORMAL,
	VISIBILITY,
	LOCAL
};

struct NormalRayCastArgPackOptiX
{
	OptixTraversableHandle 	baseAccelerator;
	// Outputs
	Span<HitKeyPack>        dHitKeys;
	Span<MetaHit>			dHits;
	// I-O
	Span<BackupRNGState>	dRNGStates;
	Span<RayGMem>           dRays;
	// Inputs
	Span<const RayIndex>    dRayIndices;
};

struct VisibilityCastArgPackOptiX
{
	OptixTraversableHandle 	baseAccelerator;
	// Input
	Bitspan<uint32_t>		dIsVisibleBuffer;
	// I-O
	Span<BackupRNGState>	dRNGStates;
	// Inputs
	Span<const RayGMem>     dRays;
	Span<const RayIndex>	dRayIndices;
};

struct LocalRayCastArgPackOptiX
{
	// Outputs
	Span<HitKeyPack>        dHitKeys;
	Span<MetaHit>			dHits;
	// I-O
	Span<BackupRNGState>	dRNGStates;
	Span<RayGMem>           dRays;
	// Inputs
	Span<const RayIndex>		dRayIndices;
	Span<const AcceleratorKey>	dAcceleratorKeys;
	// Constants
	Span<const OptixTraversableHandle>	dGlobalInstanceTraversables;
	Span<const Matrix4x4>				dGlobalInstanceInvTransforms;
	uint32_t							batchStartOffset;
};

struct ArgumentPackOptiX
{
	RenderModeOptiX mode;
	union
	{
		NormalRayCastArgPackOptiX nParams;
		VisibilityCastArgPackOptiX vParams;
		LocalRayCastArgPackOptiX lParams;
	};
};

// Hit records (as far as I understand) can be defined multiple times
// per-GAS / per build input
//
// We only utilize a subset of the SBT system.
// Each each build input will have a single record.
template <class PrimSoA = void, class TransSoA = void>
struct GenericHitRecordData
{
	// This pointer is can be accessed by optixGetPrimitiveIndex() from Optix
	Span<const PrimitiveKey> dPrimKeys;
	// Transform key will be duplicated for each Record in a GAS
	// (MRay can have up to 8 records (PrimBatch in MRay terms))
	TransformKey		transformKey;
	// Each batch can have a single alpha map (This %100 fail for OptiX compilation but check)
	Optional<AlphaMap>	alphaMap;
	// Material key of the array
	LightOrMatKey 		lightOrMatKey;
	// MRay identifier of the GAS (may be used to individually
	// trace the GAS for SSS)
	AcceleratorKey		acceleratorKey;
	// Cull face is instance level operation and is a software flag
	// for triangles, for other primitives this will be valid
	bool				cullBackFaceNonTri;
	//
	// SoA Access
	// Triangle types do not need SoA since
	// everything handled by the HW.
	//
	// For custom types, we need to access the actual data to
	// do intersection check.
	const PrimSoA* primSoA;
	// Also for PER_PRIM_TRANSFORM typed primitives, we need to have an access to
	// the Transform' SoA.
	const TransSoA* transSoA;
};

// Meta Hit Record
// This Record is kept for each accelerator in an accelerator group
template <class PrimSoA = void, class TransSoA = void>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) GenericHitRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

	GenericHitRecordData<PrimSoA, TransSoA> data;
};

// Other SBT Records are Empty (use this for those API calls)
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) EmptyHitRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	// Optix 7 Course had dummy pointer here so i will leave it as well
	void* data;
};
