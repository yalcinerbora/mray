#pragma once

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


template<class T, size_t N>
constexpr Byte* StaticVector<T, N>::ItemLocation(size_t i)
{
    return storage + i * JUMP_SIZE;
}

template<class T, size_t N>
constexpr const Byte* StaticVector<T, N>::ItemLocation(size_t i) const
{
    return storage + i * JUMP_SIZE;
}

template<class T, size_t N>
constexpr T* StaticVector<T, N>::ItemAt(size_t i )
{
    assert(count > i);
    return std::launder(reinterpret_cast<T*>(ItemLocation(i)));
}

template<class T, size_t N>
constexpr const T* StaticVector<T, N>::ItemAt(size_t i) const
{
    assert(count > i);
    return std::launder(reinterpret_cast<const T*>(ItemLocation(i)));
}

template<class T, size_t N>
constexpr void StaticVector<T, N>::DestructObjectAt(size_t i)
{
    assert(count > i);
    T* item = std::launder(reinterpret_cast<T*>(ItemLocation(i)));
    item->~T();
}

template<class T, size_t N>
constexpr StaticVector<T, N>::StaticVector()
    : count(0)
{}

template<class T, size_t N>
constexpr StaticVector<T, N>::StaticVector(StaticVecSize countIn)
requires std::is_default_constructible_v<T>
    : count(static_cast<size_t>(countIn))
{
    assert(count <= N);
    for(size_t i = 0; i < count; i++)
    {
        new(ItemLocation(i)) T();
    }
}

template<class T, size_t N>
constexpr StaticVector<T, N>::StaticVector(StaticVecSize countIn,
                                           const T& initialValue)
requires std::is_copy_constructible_v<T>
    : count(static_cast<size_t>(countIn))
{
    assert(count <= N);
    for(size_t i = 0; i < count; i++)
    {
        new(ItemLocation(i)) T(initialValue);
    }
}

template<class T, size_t N>
constexpr StaticVector<T, N>::StaticVector(const StaticVector& other)
requires std::is_copy_constructible_v<T>
    : count(other.count)
{
    assert(count <= N);
    for(size_t i = 0; i < count; i++)
    {
        new(ItemLocation(i)) T(other[i]);
    }
}

template<class T, size_t N>
constexpr StaticVector<T, N>::StaticVector(StaticVector&& other)
requires std::is_move_constructible_v<T>
{
    assert(count <= N);
    for(size_t i = 0; i < count; i++)
    {
        new(ItemLocation(i)) T(std::move(other[i]));
    }
}

template<class T, size_t N>
constexpr StaticVector<T, N>& StaticVector<T, N>::operator=(const StaticVector& other)
requires std::is_copy_assignable_v<T>
{
    assert(count <= N);
    count = other.count;
    for(size_t i = 0; i < count; i++)
    {
        if constexpr(!std::is_trivially_default_constructible_v<T>)
            DestructObjectAt(i);
        new(ItemLocation(i)) T(other[i]);
    }
}

template<class T, size_t N>
constexpr StaticVector<T, N>& StaticVector<T, N>::operator=(StaticVector&& other)
requires std::is_move_assignable_v<T>
{
    assert(count <= N);
    assert(this != &other);
    count = other.count;
    for(size_t i = 0; i < count; i++)
    {
        if constexpr(!std::is_trivially_default_constructible_v<T>)
            DestructObjectAt(i);
        new(ItemLocation(i)) T(other[i]);
    }
}

template<class T, size_t N>
constexpr StaticVector<T, N>::~StaticVector()
requires(!std::is_trivially_destructible_v<T>)
{
    clear();
}

template<class T, size_t N>
constexpr T& StaticVector<T, N>::operator[](size_t i)
{
    return *ItemAt(i);
}

template<class T, size_t N>
constexpr const T& StaticVector<T, N>::operator[](size_t i) const
{
    return *ItemAt(i);
}

template<class T, size_t N>
constexpr T* StaticVector<T, N>::data()
{
    return *ItemAt(0);
}

template<class T, size_t N>
constexpr const T* StaticVector<T, N>::data() const
{
    return ItemAt(0);
}

template<class T, size_t N>
constexpr T& StaticVector<T, N>::back()
{
    return ItemAt(count - 1);
}

template<class T, size_t N>
constexpr const T& StaticVector<T, N>::back() const
{
    return ItemAt(count - 1);
}

template<class T, size_t N>
constexpr T& StaticVector<T, N>::front()
{
    return *ItemAt(0);
}

template<class T, size_t N>
constexpr const T& StaticVector<T, N>::front() const
{
    return *ItemAt(0);
}

template<class T, size_t N>
constexpr size_t StaticVector<T, N>::size() const
{
    return count;
}

template<class T, size_t N>
constexpr size_t StaticVector<T, N>::isEmpty() const
{
    return count == 0;
}

template<class T, size_t N>
constexpr size_t StaticVector<T, N>::capacity() const
{
    return N;
}

template<class T, size_t N>
constexpr void StaticVector<T, N>::clear()
{
    if constexpr(!std::is_trivially_destructible_v<T>)
    {
        for(size_t i = 0; i < count; i++)
        {
            DestructObjectAt(i);
        }
    }
    count = 0;
}

template<class T, size_t N>
constexpr void StaticVector<T, N>::push_back(const T& t)
{
    assert(count < N);
    new(ItemLocation(count)) T(std::forward<T>(t));
    count++;
}

template<class T, size_t N>
constexpr void StaticVector<T, N>::push_back(T&& t)
{
    assert(count < N);
    new(ItemLocation(count)) T(std::forward<T>(t));
    count++;
}

template<class T, size_t N>
template<class... Args>
constexpr T& StaticVector<T, N>::emplace_back(Args&&... args)
{
    assert(count < N);
    T* ptr = new(ItemLocation(count)) T(std::forward<Args>(args)...);
    count++;
    return *ptr;
}

template<class T, size_t N>
constexpr void StaticVector<T, N>::pop_back()
{
    assert(count > 0);
    if constexpr(!std::is_trivially_destructible_v<T>)
    {
        DestructObjectAt(count - 1);
    }
    count--;
}