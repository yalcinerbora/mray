#pragma once

#include "Vector.h"
#include "Core/Types.h"

// Stratified Discrete Alias Table
//
// Given ordered ranges "k", find n
// where n is between [k_i, k_(i+1))
// n is in range of [min(k), max(k))
//
// Classical approach will use binary search
// Maximum memory usage (including storage of k) is
// size(k). Computation time is log(n)
//
// Stratified alias table increases storage cost but makes
// the computation O(1). Worst case it will have a size of
// O(max(k)) insead of O(size(k)) which can be quite large.
//
// The idea comes from inverse CDF sampling vs. alias table
// sampling, but this time it is utilized for range finding.
//
// Since this data structure utilize stratification, similar
// approach can be utilized to fast and **stratified** sampling
// of discrete probabilities. But for loating points gcd does not
// makes sense so different approach should be employed.
//
// ***Use-cases***
// Finding primitive batch id from primitiveId. In common terms,
// pirmitive batch is a mesh (series of triangles).
// A scene probably consists of multiple batches. Assuming all the
// mesh primitives are in linearly laid out in memory (this is true
// for mray). When computation wants to acquire per-primitive variables
// It can do so using the id directly (it is just an index).
//
// However; when mesh-common properties are required to be accessed
// we need to either binary search a range (memory efficient) or hold
// a indirection item per-primitive.
//
// This data structure enables in-between memory cost given there is no
// single triangled mesh in the scene. Alias table is generated as follows:
//
// - Find GCD of all mesh's triangle counts.
// - Allocate (totalTriangleCount / GCD) amount of entries
// - Partition mesh ids to the table.
//
// Worst case GCD is one (so all counts are relatively).
//
// To find the range, find the table entry index (primitiveId / GCD)
// and thats it. With this, we only do single memory fetch from global memory
// to find the primitive id. So worst case corresponds to holding a primitive-batch
// id on the primitive itself.
//
// Unfortunately due to "2-triangle walls" being commong GCD probably be 2.
// In order to save more space, this approach can co-jointly utilized by the scene loader
// which "pads" meshes to a common alignment (i.e 128) this means GCD will guareanteed
// to be a large number (again 128) and memory can be saved.
//
// By this protocol, the explained class is very simple
template <std::unsigned_integral T>
class StratifiedIntegerAliasTable
{
    private:
    T               gcd;
    const T*        gAliasRanges;

    public:
    MRAY_HYBRID     StratifiedIntegerAliasTable(const T* aliasRanges, T gcd);
    MRAY_HYBRID
    uint32_t        FindIndex(T id) const;
};

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
StratifiedIntegerAliasTable<T>::StratifiedIntegerAliasTable(const T* dAliasRanges, T gcd)
    : gcd(gcd)
    , gAliasRanges(dAliasRanges)
{}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t StratifiedIntegerAliasTable<T>::FindIndex(T id) const
{
    // TODO: Enforce GCD to be power of two to change this to shift
    uint32_t tableIndex = id / gcd;
    return gAliasRanges[tableIndex];
}




// A simple hash table with linear probing.
template <class K, class V>
class HashMapView
{
    private:
    // Keys and values are seperate,
    // keys are small, values may be large
    Span<K> keys;
    Span<V> values;

    public:
};



//IssueQueue stream,
//Get queue

//template<class Function, class... Args>
//MRAY_HOST
//void KC_X(uint32_t sharedMemSize,
//          size_t workCount,
//          //
//          Function&& f, Args&&... args)
//{
//    //CUDA_CHECK(cudaSetDevice(deviceId));
//    uint32_t blockCount = static_cast<uint32_t>((workCount + (StaticThreadPerBlock1D - 1)) / StaticThreadPerBlock1D);
//    uint32_t blockSize = StaticThreadPerBlock1D;
//
//    //std::forward<Kernel>(f)<<<blockCount, blockSize, sharedMemSize, stream>>>(std::forward<Args>(args)...);
//    CUDA_KERNEL_CHECK();
//}

