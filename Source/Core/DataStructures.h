#pragma once

#include "Vector.h"
#include "Core/Types.h"
#include <bit>
#include <type_traits>
#include <numeric>
#include <algorithm>
#include <iterator>

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

// Simple host-only, static-sized array
// This is not a replacement of std::array but it is statically
// allocated but dynamically-sized std::vector like data structure.
//
// So it is between std::array and std::vector,
// underlying type does not have to be default constructible unlike std::array
// Probably work on device as well, but currently not needed there so annotations
// are skipped purposefully.
//
// Tries to be API compatible (but not really) with std::vector
// All is marked "constexpr" however it is hard/impossible (maybe?) to
// create compile-time version of it for all types trivial/non-trivial
// (copy/move)-constrcutible types, etc.
//
// Lets not make the mistake of old std::vector impl
enum class StaticVecSize : size_t {};

template<class T, size_t N>
class alignas(std::max(alignof(T), size_t(8))) StaticVector
{
    static constexpr size_t JUMP_SIZE = std::max(sizeof(T), alignof(T));
    static constexpr size_t ALLOCATION_SIZE = sizeof(T) * JUMP_SIZE;

    private:
    Byte        storage[ALLOCATION_SIZE];
    size_t      count;

    constexpr Byte*         ItemLocation(size_t);
    constexpr const Byte*   ItemLocation(size_t) const;
    constexpr T*            ItemAt(size_t);
    constexpr const T*      ItemAt(size_t) const;
    constexpr void          DestructObjectAt(size_t);
    template<class... Args>
    constexpr T&            ConstructObjectAt(size_t, Args...);

    public:
    // Constructors & Destructor
    constexpr               StaticVector();
    constexpr               StaticVector(std::initializer_list<T>);
    constexpr explicit      StaticVector(StaticVecSize count)   requires std::is_default_constructible_v<T>;
    constexpr explicit      StaticVector(StaticVecSize count,
                                         const T& initialValue) requires std::is_copy_constructible_v<T>;
    constexpr               StaticVector(const StaticVector&)   = delete;
    constexpr               StaticVector(const StaticVector&)   requires std::is_copy_constructible_v<T>;
    constexpr               StaticVector(StaticVector&&)        = delete;
    constexpr               StaticVector(StaticVector&&)        requires std::is_move_constructible_v<T>;

    constexpr StaticVector& operator=(const StaticVector&)      = delete;
    constexpr StaticVector& operator=(const StaticVector&)      requires std::is_copy_assignable_v<T>;
    constexpr StaticVector& operator=(StaticVector&&)           = delete;
    constexpr StaticVector& operator=(StaticVector&&)           requires std::is_move_assignable_v<T>;
    constexpr               ~StaticVector()                     = default;
    constexpr               ~StaticVector()                     requires(!std::is_trivially_destructible_v<T>);

    //
    constexpr T&            operator[](size_t);
    constexpr const T&      operator[](size_t) const;
    constexpr T*            data();
    constexpr const T*      data() const;
    constexpr T&            back();
    constexpr const T&      back() const;
    constexpr T&            front();
    constexpr const T&      front() const;

    // TODO: Don't bother implementing actual iterators pointer should suffice?
    // Tap in the backwards compatibility of C++ with C arrays
    constexpr T*            begin();
    constexpr const T*      begin() const;
    constexpr T*            end();
    constexpr const T*      end() const;

    constexpr const T*      cbegin() const;
    constexpr const T*      cend() const;

    constexpr size_t        size() const;
    constexpr size_t        isEmpty() const;
    constexpr size_t        capacity() const;

    constexpr void          resize(size_t size);
    constexpr void          clear();

    constexpr void          push_back(const T&);
    constexpr void          push_back(T&&);
    template<class... Args>
    constexpr T&            emplace_back(Args&&...);
    constexpr void          pop_back();

    constexpr void          remove(T*);
};


// Flat set implementation (only MSVC has c++23 flat_map/set currently)
// This is a basic impl prob not std compliant
enum class IsSorted : size_t
{};

template<class Container>
concept RandomAccessContainerC = requires()
{
    // TODO: Should we check const_iterator as well?
    requires std::random_access_iterator<typename Container::iterator>;
};

template<class T,
         class Compare = std::less<T>,
         RandomAccessContainerC Container = std::vector<T>>
class FlatSet
{
    public:
    using iterator                  = typename Container::iterator;
    using const_iterator            = typename Container::const_iterator;
    using reverse_iterator          = typename Container::reverse_iterator;
    using const_reverse_iterator    = typename Container::const_reverse_iterator;

    private:
    Container           container;
    Compare             compare;

    public:
    // Constructors & Destructor
                    FlatSet();
                    FlatSet(Container);
    template <class Alloc>
                    FlatSet(Container, const Alloc&);
    explicit        FlatSet(const Compare&);
    template <class Alloc>
                    FlatSet(const Compare&, const Alloc&);
    template <class Alloc>
                    FlatSet(const Alloc&);
    template <class InputIterator>
                    FlatSet(InputIterator, InputIterator,
                            const Compare& = Compare());
    template <class InputIterator, class Alloc>
                    FlatSet(InputIterator, InputIterator,
                            const Compare&, const Alloc&);
    template <class InputIterator, class Alloc>
                    FlatSet(InputIterator, InputIterator,
                            const Alloc&);
    template <class Alloc>
                    FlatSet(std::initializer_list<T>,
                            const Compare&, const Alloc&);
    template <class Alloc>
                    FlatSet(std::initializer_list<T>, const Alloc&);

    // Sorted Variants
                    FlatSet(IsSorted, Container);
    template <class Alloc>
                    FlatSet(IsSorted, Container, const Alloc&);
    template <class InputIterator>
                    FlatSet(IsSorted, InputIterator, InputIterator,
                            const Compare& = Compare());
    template <class InputIterator, class Alloc>
                    FlatSet(IsSorted, InputIterator, InputIterator,
                            const Compare&, const Alloc&);
    template <class InputIterator, class Alloc>
                    FlatSet(IsSorted, InputIterator, InputIterator,
                            const Alloc&);
                    FlatSet(std::initializer_list<T>,
                            const Compare& = Compare());
                    FlatSet(IsSorted, std::initializer_list<T>,
                            const Compare& = Compare());
    template <class Alloc>
                    FlatSet(IsSorted, std::initializer_list<T>,
                            const Compare&, const Alloc&);
    template <class Alloc>
                    FlatSet(IsSorted, std::initializer_list<T>,
                            const Alloc&);
    // Iterators
    iterator        begin();
    const_iterator  begin() const;
    const_iterator  cbegin() const;

    iterator        end();
    const_iterator  end() const;
    const_iterator  cend() const;

    iterator        rbegin();
    const_iterator  rbegin() const;
    const_iterator  crbegin() const;

    iterator        rend();
    const_iterator  rend() const;
    const_iterator  crend() const;

    // Capacity
    bool            empty() const;
    size_t          size() const;
    size_t          max_size() const;

    // Modifiers
    template<class... Args>
    std::pair<iterator, bool>   emplace(Args&&...);
    // TODO: emplace_hint

    std::pair<iterator, bool>   insert(const T&);
    std::pair<iterator, bool>   insert(T&&);
    // TODO: other inserts

    // Container Swap
    Container       extract() &&;
    void            replace(Container&&);

    // TODO: erease
    // TODO: swap
    void            clear();

    // Lookup
    iterator        find(const T&);
    const_iterator  find(const T&) const;
    size_t          count(const T&) const;
    bool            contains(const T&) const;
    iterator        lower_bound(const T&);
    const_iterator  lower_bound(const T&) const;
    iterator        upper_bound(const T&);
    const_iterator  upper_bound(const T&) const;
    iterator        equal_range(const T&);
    const_iterator  equal_range(const T&) const;

    // Indexed access **extra** from standard
    // Only non-modifying version is provided
    const T&        operator[](size_t) const;
};

#include "DataStructures.hpp"