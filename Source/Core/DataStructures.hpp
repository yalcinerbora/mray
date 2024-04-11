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
    assert(i < N);
    return storage + i * JUMP_SIZE;
}

template<class T, size_t N>
constexpr const Byte* StaticVector<T, N>::ItemLocation(size_t i) const
{
    assert(i < N);
    return storage + i * JUMP_SIZE;
}

template<class T, size_t N>
constexpr T* StaticVector<T, N>::ItemAt(size_t i)
{
    return std::launder(reinterpret_cast<T*>(ItemLocation(i)));
}

template<class T, size_t N>
constexpr const T* StaticVector<T, N>::ItemAt(size_t i) const
{
    return std::launder(reinterpret_cast<const T*>(ItemLocation(i)));
}

template<class T, size_t N>
template<class... Args>
constexpr T& StaticVector<T, N>::ConstructObjectAt(size_t i, Args... args)
{
    return *std::construct_at(reinterpret_cast<T*>(ItemLocation(i)),
                              std::forward<Args>(args)...);
}

template<class T, size_t N>
constexpr void StaticVector<T, N>::DestructObjectAt(size_t i)
{
    assert(i < count);
    T* item = std::launder(reinterpret_cast<T*>(ItemLocation(i)));
    item->~T();
}

template<class T, size_t N>
constexpr StaticVector<T, N>::StaticVector()
    : storage()
    , count(0)
{}

template<class T, size_t N>
constexpr StaticVector<T, N>::StaticVector(std::initializer_list<T> init)
    : count(init.size())
{
    assert(init.size() <= N);
    size_t i = 0;
    for(const T& t : init)
    {
        ConstructObjectAt(i, t);
        i++;
    }
}

template<class T, size_t N>
constexpr StaticVector<T, N>::StaticVector(StaticVecSize countIn)
requires std::is_default_constructible_v<T>
    : storage()
    , count(static_cast<size_t>(countIn))
{
    assert(count <= N);
    for(size_t i = 0; i < count; i++)
    {
        ConstructObjectAt(i);
    }
}

template<class T, size_t N>
constexpr StaticVector<T, N>::StaticVector(StaticVecSize countIn,
                                           const T& initialValue)
requires std::is_copy_constructible_v<T>
    : storage()
    , count(static_cast<size_t>(countIn))
{
    assert(count <= N);
    for(size_t i = 0; i < count; i++)
    {
        ConstructObjectAt(i, initialValue);
    }
}

template<class T, size_t N>
constexpr StaticVector<T, N>::StaticVector(const StaticVector& other)
requires std::is_copy_constructible_v<T>
    : storage()
    , count(other.count)
{
    assert(count <= N);
    for(size_t i = 0; i < count; i++)
    {
        ConstructObjectAt(i, other[i]);
    }
}

template<class T, size_t N>
constexpr StaticVector<T, N>::StaticVector(StaticVector&& other)
requires std::is_move_constructible_v<T>
    : storage()
    , count(other.count)
{
    assert(count <= N);
    for(size_t i = 0; i < count; i++)
    {
        // TODO: This is %99 wrong, check it
        ConstructObjectAt(i, std::move(other[i]));
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
        ConstructObjectAt(i, other[i]);
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
        ConstructObjectAt(i, std::move(other[i]));
    }
    other.count = 0;
    return *this;
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
    assert(i < count);
    return *ItemAt(i);
}

template<class T, size_t N>
constexpr const T& StaticVector<T, N>::operator[](size_t i) const
{
    assert(i < count);
    return *ItemAt(i);
}

template<class T, size_t N>
constexpr T* StaticVector<T, N>::data()
{
    assert(count != 0);
    return ItemAt(0);
}

template<class T, size_t N>
constexpr const T* StaticVector<T, N>::data() const
{
    assert(count != 0);
    return ItemAt(0);
}

template<class T, size_t N>
constexpr T& StaticVector<T, N>::back()
{
    assert(count != 0);
    return ItemAt(count - 1);
}

template<class T, size_t N>
constexpr const T& StaticVector<T, N>::back() const
{
    assert(count != 0);
    return ItemAt(count - 1);
}

template<class T, size_t N>
constexpr T& StaticVector<T, N>::front()
{
    assert(count > 0);
    return *ItemAt(0);
}

template<class T, size_t N>
constexpr const T& StaticVector<T, N>::front() const
{
    assert(count > 0);
    return *ItemAt(0);
}

template<class T, size_t N>
constexpr T* StaticVector<T, N>::begin()
{
    return ItemAt(0);
}

template<class T, size_t N>
constexpr const T* StaticVector<T, N>::begin() const
{
    return ItemAt(0);
}

template<class T, size_t N>
constexpr T* StaticVector<T, N>::end()
{
    return ItemAt(0) + count;
}

template<class T, size_t N>
constexpr const T* StaticVector<T, N>::end() const
{
    return ItemAt(0) + count;
}

template<class T, size_t N>
constexpr const T* StaticVector<T, N>::cbegin() const
{
    return begin();
}

template<class T, size_t N>
constexpr const T* StaticVector<T, N>::cend() const
{
    return end();
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
constexpr void StaticVector<T, N>::resize(size_t size)
{
    assert(size <= N);
    if(size == count) return;

    size_t start = count, end = size;
    if(size < count) std::swap(start, end);

    for(size_t i = start; i < end; i++)
    {
        if(size < count) DestructObjectAt(i);
        else ConstructObjectAt(i);
    }
    count = size;
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
    ConstructObjectAt(count, t);
    count++;
}

template<class T, size_t N>
constexpr void StaticVector<T, N>::push_back(T&& t)
{
    assert(count < N);
    ConstructObjectAt(count, std::forward<T>(t));
    count++;
}

template<class T, size_t N>
template<class... Args>
constexpr T& StaticVector<T, N>::emplace_back(Args&&... args)
{
    assert(count < N);
    T& ref = ConstructObjectAt(count, std::forward<Args>(args)...);
    count++;
    return ref;
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

template<class T, size_t N>
constexpr void StaticVector<T, N>::remove(T* loc)
{
    assert(loc >= begin() && loc < end());
    for(T* i = loc; i != (end() - 1); i++)
    {
        std::swap(*i, *(i + 1));
    }
    pop_back();
}

template <class T, class Comp, RandomAccessContainerC Cont>
FlatSet<T, Comp, Cont>::FlatSet()
{}

template <class T, class Comp, RandomAccessContainerC Cont>
FlatSet<T, Comp, Cont>::FlatSet(Cont c)
    : FlatSet(IsSorted(), c)
{
    std::sort(container.begin(), container.end(), compare);
}

template <class T, class Comp, RandomAccessContainerC Cont>
template <class Alloc>
FlatSet<T, Comp, Cont>::FlatSet(Cont c, const Alloc& a)
    : FlatSet(IsSorted(), c, a)
{
    std::sort(container.begin(), container.end(), compare);
}

template <class T, class Comp, RandomAccessContainerC Cont>
FlatSet<T, Comp, Cont>::FlatSet(const Comp& comp)
    : container()
    , compare(comp)
{}

template <class T, class Comp, RandomAccessContainerC Cont>
template <class Alloc>
FlatSet<T, Comp, Cont>::FlatSet(const Comp& comp, const Alloc& a)
    : container(a)
    , compare(comp)
{}

template <class T, class Comp, RandomAccessContainerC Cont>
template <class Alloc>
FlatSet<T, Comp, Cont>::FlatSet(const Alloc& a)
    : container(a)
{}

template <class T, class Comp, RandomAccessContainerC Cont>
template <class InputIterator>
FlatSet<T, Comp, Cont>::FlatSet(InputIterator first, InputIterator last,
                                const Comp& comp)
    : FlatSet(IsSorted(), first, last, comp)
{
    std::sort(container.begin(), container.end(), compare);
}

template <class T, class Comp, RandomAccessContainerC Cont>
template <class InputIterator, class Alloc>
FlatSet<T, Comp, Cont>::FlatSet(InputIterator first, InputIterator last,
                                const Comp& comp, const Alloc& a)
    : FlatSet(IsSorted(), first, last, comp, a)
{
    std::sort(container.begin(), container.end(), compare);
}

template <class T, class Comp, RandomAccessContainerC Cont>
template <class InputIterator, class Alloc>
FlatSet<T, Comp, Cont>::FlatSet(InputIterator first, InputIterator last,
                                const Alloc& a)
    : FlatSet(IsSorted(), first, last, a)
{
    std::sort(container.begin(), container.end(), compare);
}

template <class T, class Comp, RandomAccessContainerC Cont>
FlatSet<T, Comp, Cont>::FlatSet(std::initializer_list<T> il,
                                const Comp& comp)
    : FlatSet(IsSorted(), il, comp)
{
    std::sort(container.begin(), container.end(), compare);
}

template <class T, class Comp, RandomAccessContainerC Cont>
template <class Alloc>
FlatSet<T, Comp, Cont>::FlatSet(std::initializer_list<T> il,
                                const Comp& comp, const Alloc& a)
    : FlatSet(IsSorted(), il, comp, a)
{
    std::sort(container.begin(), container.end(), compare);
}

template <class T, class Comp, RandomAccessContainerC Cont>
template <class Alloc>
FlatSet<T, Comp, Cont>::FlatSet(std::initializer_list<T> il, const Alloc& a)
    : FlatSet(IsSorted(), il, a)
{
    std::sort(container.begin(), container.end(), compare);
}

template <class T, class Comp, RandomAccessContainerC Cont>
FlatSet<T, Comp, Cont>::FlatSet(IsSorted, Cont c)
    : container(c)
{}

template <class T, class Comp, RandomAccessContainerC Cont>
template <class Alloc>
FlatSet<T, Comp, Cont>::FlatSet(IsSorted, Cont c, const Alloc& a)
    : container(c, a)
{}

template <class T, class Comp, RandomAccessContainerC Cont>
template <class InputIterator>
FlatSet<T, Comp, Cont>::FlatSet(IsSorted, InputIterator first,
                                InputIterator last,
                                const Comp& comp)
    : container(first, last)
    , compare(comp)
{}

template <class T, class Comp, RandomAccessContainerC Cont>
template <class InputIterator, class Alloc>
FlatSet<T, Comp, Cont>::FlatSet(IsSorted, InputIterator first,
                                InputIterator last,
                                const Comp& comp, const Alloc& a)
    : container(first, last, a)
    , compare(comp)
{}

template <class T, class Comp, RandomAccessContainerC Cont>
template <class InputIterator, class Alloc>
FlatSet<T, Comp, Cont>::FlatSet(IsSorted, InputIterator first,
                                InputIterator last,
                                const Alloc& a)
    : container(first, last, a)
    , compare(Comp())
{}

template <class T, class Comp, RandomAccessContainerC Cont>
FlatSet<T, Comp, Cont>::FlatSet(IsSorted, std::initializer_list<T> il,
                                const Comp& comp)
    : container(il)
    , compare(comp)
{}

template <class T, class Comp, RandomAccessContainerC Cont>
template <class Alloc>
FlatSet<T, Comp, Cont>::FlatSet(IsSorted, std::initializer_list<T> il,
                                const Comp& comp, const Alloc& a)
    : container(il, a)
    , compare(comp)

{}

template <class T, class Comp, RandomAccessContainerC Cont>
template <class Alloc>
FlatSet<T, Comp, Cont>::FlatSet(IsSorted, std::initializer_list<T> il,
                                const Alloc& a)
    : container(il, a)
    , compare(Comp())
{}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::iterator
FlatSet<T, Comp, Cont>::begin()
{
    return container.begin();
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::const_iterator
FlatSet<T, Comp, Cont>::begin() const
{
    return container.begin();
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::const_iterator
FlatSet<T, Comp, Cont>::cbegin() const
{
    return container.cbegin();
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::iterator
FlatSet<T, Comp, Cont>::end()
{
    return container.end();
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::const_iterator
FlatSet<T, Comp, Cont>::end() const
{
    return container.end();
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::const_iterator
FlatSet<T, Comp, Cont>::cend() const
{
    return container.cend();
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::iterator
FlatSet<T, Comp, Cont>::rbegin()
{
    return container.rbegin();
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::const_iterator
FlatSet<T, Comp, Cont>::rbegin() const
{
    return container.rbegin();
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::const_iterator
FlatSet<T, Comp, Cont>::crbegin() const
{
    return container.crbegin();
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::iterator
FlatSet<T, Comp, Cont>::rend()
{
    return container.rend();
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::const_iterator
FlatSet<T, Comp, Cont>::rend() const
{
    return container.rend();
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::const_iterator
FlatSet<T, Comp, Cont>::crend() const
{
    return container.crend();
}

template <class T, class Comp, RandomAccessContainerC Cont>
bool FlatSet<T, Comp, Cont>::empty() const
{
    return container.empty();
}

template <class T, class Comp, RandomAccessContainerC Cont>
size_t FlatSet<T, Comp, Cont>::size() const
{
    return container.size();
}

template <class T, class Comp, RandomAccessContainerC Cont>
size_t FlatSet<T, Comp, Cont>::max_size() const
{
    return std::numeric_limits<size_t>::max();
}

template <class T, class Comp, RandomAccessContainerC Cont>
template<class... Args>
std::pair<typename FlatSet<T, Comp, Cont>::iterator, bool>
FlatSet<T, Comp, Cont>::emplace(Args&&... args)
{
    // delegate to move insert
    return insert(T(std::forward<Args>(args)...));
}

template <class T, class Comp, RandomAccessContainerC Cont>
std::pair<typename FlatSet<T, Comp, Cont>::iterator, bool>
FlatSet<T, Comp, Cont>::insert(const T& t)
{
    // delegate to move version
    return insert(T(t));
}

template <class T, class Comp, RandomAccessContainerC Cont>
std::pair<typename FlatSet<T, Comp, Cont>::iterator, bool>
FlatSet<T, Comp, Cont>::insert(T&& t)
{
    auto loc = std::lower_bound(container.begin(), container.end(),
                                t, compare);
    bool shouldInsert = (loc == container.end());
    if(!shouldInsert)
    {
        // Utilize less than for equavilency check
        // user may not define other operators (such as '==')
        shouldInsert = !(!compare(t, *loc) &&
                         !compare(*loc, t));
    }
    if(shouldInsert)
        loc = container.insert(loc, std::forward<T>(t));

    return std::pair(loc, shouldInsert);
}

template <class T, class Comp, RandomAccessContainerC Cont>
Cont FlatSet<T, Comp, Cont>::extract() &&
{
    return container;
}

template <class T, class Comp, RandomAccessContainerC Cont>
void FlatSet<T, Comp, Cont>::replace(Cont&& c)
{
    container = std::move(c);
}

template <class T, class Comp, RandomAccessContainerC Cont>
void FlatSet<T, Comp, Cont>::clear()
{
    container.clear();
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::iterator
FlatSet<T, Comp, Cont>::find(const T& t)
{
    auto loc = std::lower_bound(container.begin(), container.end(),
                                t, compare);
    if(loc == container.end()) return;
    if(!compare(t, *loc) && !compare(*loc, t))
        return loc;
    else return container.end();
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::const_iterator
FlatSet<T, Comp, Cont>::find(const T& t) const
{
    auto loc = std::lower_bound(container.begin(), container.end(),
                                t, compare);
    if(loc == container.end()) return;
    if(!compare(t, *loc) && !compare(*loc, t))
        return loc;
    else return container.end();
}

template <class T, class Comp, RandomAccessContainerC Cont>
size_t FlatSet<T, Comp, Cont>::count(const T& t) const
{
    auto loc = find(t);
    return (loc != container.end()) ? 1 : 0;
}

template <class T, class Comp, RandomAccessContainerC Cont>
bool FlatSet<T, Comp, Cont>::contains(const T& t) const
{
    auto loc = find(t);
    return (loc != container.end()) ? 1 : 0;
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::iterator
FlatSet<T, Comp, Cont>::lower_bound(const T& t)
{
    return std::lower_bound(container.begin(), container.end(),
                            t, compare);
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::const_iterator
FlatSet<T, Comp, Cont>::lower_bound(const T& t) const
{
    return std::lower_bound(container.cbegin(), container.cend(),
                            t, compare);
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::iterator
FlatSet<T, Comp, Cont>::upper_bound(const T& t)
{
    return std::upper_bound(container.begin(), container.end(),
                            t, compare);
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::const_iterator
FlatSet<T, Comp, Cont>::upper_bound(const T& t) const
{
    return std::upper_bound(container.cbegin(), container.cend(),
                            t, compare);
}

template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::iterator
FlatSet<T, Comp, Cont>::equal_range(const T& t)
{
    return std::equal_range(container.begin(), container.end(),
                            t, compare);
}


template <class T, class Comp, RandomAccessContainerC Cont>
typename FlatSet<T, Comp, Cont>::const_iterator
FlatSet<T, Comp, Cont>::equal_range(const T& t) const
{
    return std::equal_range(container.cbegin(), container.cend(),
                            t, compare);
}

template <class T, class Comp, RandomAccessContainerC Cont>
const T& FlatSet<T, Comp, Cont>::operator[](size_t i) const
{
    assert(i < container.size());
    return container[i];
}