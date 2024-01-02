#pragma once

#include <tuple>
#include <concepts>

inline size_t operator ""_MiB(size_t s)
{
    return 1024 * 1024 * s;
}

inline size_t operator ""_KiB(size_t s)
{
    return 1024 * s;
}

//template <class MemT>
//concept MemoryC = requires(MemT)
//{
//
//    true
//};

//namespace MemAlloc
//{
//
//}

//template<typename T>
//const T* start_lifetime_as(const void* p) noexcept {
//    const auto mutable_pointer = const_cast<void*>(p);
//    const auto bytes = new(mutable_pointer) std::byte[sizeof(T)];
//    const auto pointer = reinterpret_cast<T*>(bytes);
//    (void)*pointer;
//    return pointer;
//}

//template<class T>
//requires (std::is_trivially_copyable_v<T>)
//T* start_lifetime_as(void* p) noexcept
//{
//    return std::launder(static_cast<T*>(std::memmove(p, p, sizeof(T))));
//}

//template<class T>
//requires (std::is_trivially_copyable_v<T>)
//T* start_lifetime_as_array(void* p, size_t size) noexcept
//{
//    return std::launder(static_cast<T*>(std::memmove(p, p, sizeof(T) * size)));
//}

//template <MemoryC GPUMem>
//inline void GPUMemFuncs::EnlargeBuffer(GPUMem& mem, size_t s)
//{
//    if(s > mem.Size())
//    {
//        mem = std::move(GPUMem(mem.Device()));
//        mem = std::move(GPUMem(mem.Device(), s));
//    }
//}
//
//template <>
//inline void GPUMemFuncs::EnlargeBuffer(DeviceMemory& mem, size_t s)
//{
//    if(s > mem.Size())
//    {
//        mem = DeviceMemory();
//        mem = DeviceMemory(s);
//    }
//}
//
//namespace DeviceMemDetail
//{
//    template<size_t I = 0, class... Tp>
//    inline typename std::enable_if<I == sizeof...(Tp), size_t>::type
//        AcquireTotalSize(std::array<size_t, sizeof...(Tp)>&,
//                         const std::array<size_t, sizeof...(Tp)>&,
//                         size_t)
//    {
//        return 0;
//    }
//
//    template<std::size_t I = 0, class... Tp>
//    inline typename std::enable_if<(I < sizeof...(Tp)), size_t>::type
//        AcquireTotalSize(std::array<size_t, sizeof...(Tp)>& alignedSizeList,
//                         const std::array<size_t, sizeof...(Tp)>& countList,
//                         size_t alignment)
//    {
//        using CurrentType = typename std::tuple_element_t<I, std::tuple<Tp...>>;
//
//        size_t alignedSize = Memory::AlignSize(sizeof(CurrentType) * countList[I],
//                                               alignment);
//
//        alignedSizeList[I] = alignedSize;
//
//        return alignedSize + AcquireTotalSize<I + 1, Tp...>(alignedSizeList, countList, alignment);
//    }
//
//    template<std::size_t I = 0, class... Tp>
//    inline typename std::enable_if<I == sizeof...(Tp), void>::type
//        CalculatePointers(std::tuple<Tp*&...>&, size_t&, Byte*,
//                          const std::array<size_t, sizeof...(Tp)>&)
//    {}
//
//    template<std::size_t I = 0, class... Tp>
//    inline typename std::enable_if<(I < sizeof...(Tp)), void>::type
//        CalculatePointers(std::tuple<Tp*&...>& t, size_t& offset, Byte* memory,
//                          const std::array<size_t, sizeof...(Tp)>& alignedSizeList)
//    {
//        using CurrentType = typename std::tuple_element_t<I, std::tuple<Tp...>>;
//        // Set Pointer
//        size_t size = alignedSizeList[I];
//        std::get<I>(t) = (size == 0) ? nullptr : reinterpret_cast<CurrentType*>(memory + offset);
//        // Increment Offset
//        offset += size;
//        // Statically Recurse
//        CalculatePointers<I + 1, Tp...>(t, offset, memory, alignedSizeList);
//    }
//}
//
//template <class GPUMem, class... Args>
//void GPUMemFuncs::AllocateMultiData(std::tuple<Args*&...> pointers, GPUMem& memory,
//                                    const std::array<size_t, sizeof...(Args)>& countList,
//                                    size_t alignment)
//{
//    std::array<size_t, sizeof...(Args)> alignedSizeList;
//    // Acquire total size & allocation size of each array
//    size_t totalSize = DeviceMemDetail::AcquireTotalSize<0, Args...>(alignedSizeList,
//                                                                     countList,
//                                                                     alignment);
//    // Allocate Memory
//    GPUMemFuncs::EnlargeBuffer(memory, totalSize);
//    Byte* ptr = static_cast<Byte*>(memory);
//    // Populate pointers
//    size_t offset = 0;
//    DeviceMemDetail::CalculatePointers(pointers, offset, ptr, alignedSizeList);
//
//    assert(totalSize == offset);
//}