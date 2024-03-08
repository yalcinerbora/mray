#pragma once

namespace TransientPoolDetail
{

// TypeId hash may generate false positives, but it should
// catch some error most of the time
template<ImplicitLifetimeC T>
inline TransientData::TransientData(std::in_place_type_t<T>, size_t count)
    : typeHash(typeid(T).hash_code())
    , usedBytes(0)
    , alignment(alignof(T))
{
    Byte* ptr = reinterpret_cast<Byte*>(mainR.allocate(count * sizeof(T), alignof(T)));
    // TODO: add start_lifetime_as ?
    ownedMem = Span<Byte>(ptr, count * sizeof(T));
}

inline TransientData::TransientData(TransientData&& other)
    : typeHash(other.typeHash)
    , ownedMem(std::move(other.ownedMem))
    , usedBytes(other.usedBytes)
    , alignment(other.alignment)
{
    other.ownedMem = Span<Byte>();
}

inline TransientData& TransientData::operator=(TransientData&& other)
{
    assert(this != &other);
    if(ownedMem.data() != nullptr)
        mainR.deallocate(ownedMem.data(), ownedMem.size(), alignment);
    ownedMem = other.ownedMem;
    typeHash = other.typeHash;
    usedBytes = other.usedBytes;
    alignment = other.alignment;

    other.ownedMem = Span<Byte>();
    return *this;
}

inline TransientData::~TransientData()
{
    if(ownedMem.data() != nullptr)
        mainR.deallocate(ownedMem.data(), ownedMem.size(), alignment);
}

template<ImplicitLifetimeC T>
inline void TransientData::Push(Span<const T> data)
{
    if(typeHash != typeid(T).hash_code())
       throw MRayError("TransientData(Push): Object did constructed with this type!");

    size_t writeSize = data.size_bytes();
    if((writeSize + usedBytes) > ownedMem.size())
        throw MRayError("TransientData(Push): Out of bounds push over the buffer");

    const Byte* writeBytes = reinterpret_cast<const Byte*>(data.data());
    std::memcpy(ownedMem.data() + usedBytes, writeBytes, writeSize);
    usedBytes += writeSize;
}

template<ImplicitLifetimeC T>
inline Span<const T> TransientData::AccessAs() const
{
    if(typeHash != typeid(T).hash_code())
       throw MRayError("TransientData(AccessAs): Object did constructed with this type!");

    return Span<const T>(std::launder(reinterpret_cast<T*>(ownedMem.data())),
                         usedBytes / sizeof(T));
}

template<ImplicitLifetimeC T>
inline Span<T> TransientData::AccessAs()
{
    if(typeHash != typeid(T).hash_code())
       throw MRayError("TransientData(AccessAs): Object did constructed with this type!");

    return Span<T>(std::launder(reinterpret_cast<T*>(ownedMem.data())),
                    usedBytes / sizeof(T));
}

template<>
inline Span<const Byte> TransientData::AccessAs() const
{
    return ToConstSpan(ownedMem);
}

template<>
inline Span<Byte> TransientData::AccessAs()
{
    return ownedMem;
}

// =========================== //
//    String Specialization    //
// =========================== //

// Strings are special, assume as char array
// count should be the count of
template<StringC T>
inline TransientData::TransientData(std::in_place_type_t<T>, size_t charCount)
    : typeHash(typeid(std::string).hash_code())
    , usedBytes(0)
    , alignment(alignof(typename std::string::value_type))
{
    using CharType = typename std::string::value_type;
    Byte* ptr = reinterpret_cast<Byte*>(mainR.allocate(charCount * sizeof(CharType), alignof(CharType)));
    ownedMem = Span<Byte>(ptr, charCount * sizeof(CharType));
}

template<StringC T>
inline void TransientData::Push(Span<const T, 1> str)
{
    if(typeHash != typeid(std::string).hash_code() &&
       typeHash != typeid(std::string_view).hash_code())
        throw MRayError("TransientData(Push): Object did constructed with "
                        "either std::string or std::string_view!");

    size_t writeSize = str[0].size();
    if((writeSize + usedBytes) > ownedMem.size())
        throw MRayError("TransientData(Push): Out of bounds push over the buffer");

    const Byte* writeBytes = reinterpret_cast<const Byte*>(str.data());
    std::memcpy(ownedMem.data() + usedBytes, writeBytes, writeSize);
    usedBytes += writeSize;
}

inline std::string_view TransientData::AccessAsString() const
{
    if(typeHash != typeid(std::string).hash_code() &&
       typeHash != typeid(std::string_view).hash_code())
       throw MRayError("TransientData(AccessAsString): Object did constructed with "
                       "either std::string or std::string_view!");

    return std::string_view(reinterpret_cast<char*>(ownedMem.data()), usedBytes);
}

inline std::string_view TransientData::AccessAsString()
{
    if(typeHash != typeid(std::string).hash_code() &&
       typeHash != typeid(std::string_view).hash_code())
        throw MRayError("TransientData(AccessAsString): Object did constructed with "
                        "either std::string or std::string_view!");

    return std::string_view(reinterpret_cast<char*>(ownedMem.data()), usedBytes);
}

inline FreeListNode* FreeList::GetALocation(TransientData buffer)
{
    FreeListNode* n = nullptr;
    std::lock_guard<std::mutex> lock(m);
    if(freeHead)
    {
        n = freeHead;
        freeHead = freeHead->next;
        n->prev = nullptr;
        n->next = nullptr;
        n->input = std::move(buffer);
    }
    else
    {
        void* allocatedPtr = monoBuffer.allocate(sizeof(FreeListNode),
                                                 alignof(FreeListNode));
        n = new(allocatedPtr)FreeListNode{std::move(buffer)};
    }
    return n;
}

inline void FreeList::GiveTheLocation(FreeListNode* node)
{
    std::lock_guard<std::mutex> lock(m);
    node->next = freeHead;
    freeHead = node;
}

}