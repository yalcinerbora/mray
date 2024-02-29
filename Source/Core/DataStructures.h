#pragma once

#include "Vector.h"
#include "Core/Types.h"
#include <bit>

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
    T               gcdShift;
    const T*        gAliasRanges;

    public:
    MRAY_HYBRID     StratifiedIntegerAliasTable(const T* aliasRanges, T gcd);
    MRAY_HYBRID
    uint32_t        FindIndex(T id) const;
};


// A simple lookup table with linear probing.
// It is a hash table but it is specifically not called hash table
// since it is not generic, you can not remove data from it.
//
// Data Management is outside of the scope of the hash table
// since it may reside on GPU

template <class LookupStrategy, class H, class Key>
concept LookupStrategyC = requires()
{
    { LookupStrategy::Hash(Key{}, size_t{}) } -> std::same_as<H>;
    { LookupStrategy::IsSentinel(H{}) } -> std::same_as<bool>;
    { LookupStrategy::IsEmpty(H{}) } -> std::same_as<bool>;
};

template <class K, class V, std::unsigned_integral H,
          uint32_t VEC_LENGTH, LookupStrategyC<H, K> Strategy>
class LookupTable
{
    static constexpr uint32_t VL = VEC_LENGTH;
    // Vec can be 2-3-4 so check if VL is 3, rest will be caught by
    // Vector class
    static_assert(std::has_single_bit(VL), "Lookup table hash chunk size"
                  " must be power of two");
    static constexpr uint32_t VEC_SHIFT = (VL == 2) ? 1 : 2;

    private:
    // Keys and values are seperate,
    // keys are small, values may be large
    Span<Vector<4, H>>  hashes;
    Span<K>             keys;
    Span<V>             values;

    public:
    MRAY_HYBRID         LookupTable(const Span<Vector<VL, H>>& hashes,
                                    const Span<K>& keys,
                                    const Span<V>& values);

    MRAY_HYBRID
    Optional<const V&>  Search(const K&) const;
};

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
StratifiedIntegerAliasTable<T>::StratifiedIntegerAliasTable(const T* dAliasRanges, T gcd)
    : gcdShift(static_cast<T>(std::popcount(gcd - 1)))
    , gAliasRanges(dAliasRanges)
{
    assert(std::has_single_bit(gcd));
}

template <std::unsigned_integral T>
MRAY_HYBRID MRAY_CGPU_INLINE
uint32_t StratifiedIntegerAliasTable<T>::FindIndex(T id) const
{
    uint32_t tableIndex = id >> gcdShift;
    return gAliasRanges[tableIndex];
}

template <class K, class V, std::unsigned_integral H,
          uint32_t VECL, LookupStrategyC<H, K> S>
MRAY_HYBRID MRAY_CGPU_INLINE
LookupTable<K, V, H, VECL, S>::LookupTable(const Span<Vector<VL, H>>& hashes,
                                           const Span<K>& keys,
                                           const Span<V>& values)
    : hashes(hashes)
    , keys(keys)
    , values(values)
{
    assert(keys.size() == values.size());
    assert(keys.size() * VECL <= hashes.size());
}

template <class K, class V, std::unsigned_integral H,
          uint32_t VECL, LookupStrategyC<H, K> S>
MRAY_HYBRID MRAY_CGPU_INLINE
Optional<const V&> LookupTable<K, V, H, VECL, S>::Search(const K& k) const
{
    uint32_t tableSize = static_cast<uint32_t>(keys.size());
    H hashVal = S::Hash(k);
    H index = hashVal % tableSize;

    while(true)
    {
        uint32_t vectorIndex = index >> VEC_SHIFT;
        Vector<4, H> hashChunk = hashes[vectorIndex];
        UNROLL_LOOP
        for(uint32_t i = 0; i < VL; i++)
        {
            // Roll to start of the case special case
            // (since we are bulk reading)
            if(vectorIndex + i >= tableSize) break;
            // If empty, this means linear probe chain is iterated
            // and we did not find the value return null
            if(S::IsEmpty(hashChunk[i])) return std::nullopt;

            // Actual comparison case, if hash is equal it does not mean
            // keys are equal, check them only if the hashes are equal
            uint32_t globalIndex = vectorIndex + i;
            if(hashVal == hashChunk[i] && keys[globalIndex] == k)
                return values[globalIndex];
        }
        index = (index >= tableSize) ? 0 : (index + VL);
        assert(index != hashVal % tableSize);
    }
    return std::nullopt;
}