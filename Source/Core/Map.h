#pragma once

#include <concepts>
#include <map>
#include <shared_mutex>

#include "Types.h"

// Map class, wraps std::map and eliminates some quirks of the map
// (operator[] being non-const for example and issues an insert if
// key is not found). Adding optional fetching instead of exceptions to "at"
// Also a simple mutex wrapped version for thread safety
//
// This also alias the std::map so we can change it later
//
// Currently map is used due to not validating and references to this class
// and "fast" lookups. So only "emplace" and "at" is used.
// we do not remove elements currently (we may require it in the future)

template<class Key, class T,
         class Compare = std::less<Key>,
         class Allocator = std::allocator<std::pair<const Key, T>>>
class Map : private std::map<Key, T, Compare, Allocator>
{
    public:
    using Base = std::map<Key, T, Compare, Allocator>;
    using iterator = typename Base::iterator;
    public:
    // Constructors & Destructor
    using Base::Base;
    using Base::operator=;
    // Iterators
    using Base::cbegin;
    using Base::begin;
    using Base::cend;
    using Base::end;
    // Capacity
    using Base::empty;
    using Base::size;
    // Modifiers
    using Base::clear;
    using Base::emplace;
    using Base::try_emplace;
    using Base::erase;
    // Lookup
    using Base::find;

    template<class KConv>
    requires std::convertible_to<KConv, Key>
    Optional<std::reference_wrapper<T>>         at(const KConv&);

    template<class KConv>
    requires std::convertible_to<KConv, Key>
    Optional<std::reference_wrapper<const T>>   at(const KConv&) const;
};

// Simple thread safe map wrapper
template<class K, class V>
class ThreadSafeMap
{
    public:
    using MapType = Map<K, V>;
    using Iterator = typename MapType::iterator;
    private:
    MapType                     map;
    mutable std::shared_mutex   mutex;

    public:
    template<class... Args>
    std::pair<Iterator, bool>   try_emplace(const K& k, Args&&... args);
    void                        remove_at(const K&);
    void                        clear();

    Optional<std::reference_wrapper<const V>>
    at(const K& k) const;
    Optional<std::reference_wrapper<V>>
    at(const K& k);

    const MapType& Map() const;
    MapType& Map();
};

// Simple thread safe vector wrapper
template<class T>
class ThreadSafeVector
{
    public:
    using VecType = std::vector<T>;
    using Iterator = typename VecType::iterator;
    private:
    VecType                     vector;
    mutable std::shared_mutex   mutex;

    public:
    template<class... Args>
    T&          emplace_back(Args&&... args);
    const T&    at(size_t i) const;
    void        clear();

    const VecType&  Vec() const;
    VecType&        Vec();
};

template<class K, class T, class C, class A>
template<class KC>
requires std::convertible_to<KC, K>
inline Optional<std::reference_wrapper<T>> Map<K, T, C, A>::at(const KC& k)
{
    auto loc = find(k);
    if(loc == cend())
        return std::nullopt;
    return loc->second;
}

template<class K, class T, class C, class A>
template<class KC>
requires std::convertible_to<KC, K>
inline Optional<std::reference_wrapper<const T>> Map<K, T, C, A>::at(const KC& k) const
{
    auto loc = find(k);
    if(loc == cend())
        return std::nullopt;
    return loc->second;
}

template<class K, class V>
template<class... Args>
std::pair<typename ThreadSafeMap<K, V>::Iterator, bool>
ThreadSafeMap<K, V>::try_emplace(const K& k, Args&&... args)
{
    std::unique_lock<std::shared_mutex> l(mutex);
    return map.try_emplace(k, std::forward<Args>(args)...);
}

template<class K, class V>
Optional<std::reference_wrapper<const V>> ThreadSafeMap<K, V>::at(const K& k) const
{
    std::shared_lock<std::shared_mutex> l(mutex);
    auto loc = map.find(k);
    if(loc == map.cend())
        return std::nullopt;
    return loc->second;
}

template<class K, class V>
Optional<std::reference_wrapper<V>> ThreadSafeMap<K, V>::at(const K& k)
{
    std::shared_lock<std::shared_mutex> l(mutex);
    auto loc = map.find(k);
    if(loc == map.cend())
        return std::nullopt;
    return loc->second;
}

template<class K, class V>
void ThreadSafeMap<K, V>::remove_at(const K& k)
{
    std::unique_lock<std::shared_mutex> l(mutex);
    map.erase(map.find(k));
}

template<class K, class V>
void ThreadSafeMap<K, V>::clear()
{
    std::unique_lock<std::shared_mutex> l(mutex);
    map.clear();
}

template<class K, class V>
const typename ThreadSafeMap<K, V>::MapType&
ThreadSafeMap<K, V>::Map() const
{
    return map;
}

template<class K, class V>
typename ThreadSafeMap<K, V>::MapType&
ThreadSafeMap<K, V>::Map()
{
    return map;
}

template<class T>
template<class... Args>
T& ThreadSafeVector<T>::emplace_back(Args&&... args)
{
    std::shared_lock<std::shared_mutex> l(mutex);
    return vector.emplace_back(std::forward<Args>(args)...);
}

template <class T>
const T& ThreadSafeVector<T>::at(size_t i) const
{
    std::shared_lock<std::shared_mutex> l(mutex);
    return vector.at(i);
}

template <class T>
void ThreadSafeVector<T>::clear()
{
    std::unique_lock<std::shared_mutex> l(mutex);
    vector.clear();
}

template <class T>
const typename ThreadSafeVector<T>::VecType&
ThreadSafeVector<T>::Vec() const
{
    return vector;
}

template <class T>
typename ThreadSafeVector<T>::VecType&
ThreadSafeVector<T>::Vec()
{
    return vector;
}