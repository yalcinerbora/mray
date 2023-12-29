//#pragma once
//
///**
//
//CUDA Device Memory RAII principle classes
//
//New unified memory classes are used where applicable
//These are wrapper of cuda functions and their most important responsibility is
//to delete allocated memory
//
//All of the operations (except allocation) are asynchronous.
//
//TODO: should we interface these?
//
//*/
//#include <cuda_runtime.h>
//
//#include <limits>
//#include <fstream>
//#include <iostream>
//#include <tuple>
//#include <cassert>
//
//#include "Core/Definitions.h"
//#include "GPUSystem.h"
//
//namespace mray::cuda
//{
//
//class DeviceMemoryI
//{
//    public:
//        virtual             ~DeviceMemoryI() = default;
//
//        // Interface
//        virtual size_t      Size() const = 0;
//};
//
//// Basic semi-interface for memories that are static for each GPU
//// Textures are one example
//class DeviceLocalMemoryI : public DeviceMemoryI
//{
//    private:
//    protected:
//        const GPUDevice&        currentDevice;
//
//    public:
//                                DeviceLocalMemoryI(const GPUDevice& gpu) : currentDevice(gpu) {}
//        virtual                 ~DeviceLocalMemoryI() = default;
//
//        // Interface
//        const GPUDevice&        Device() const;
//        virtual void            MigrateToOtherDevice(const GPUDevice* deviceTo, cudaStream_t stream = (cudaStream_t)0) = 0;
//};
//
//// Device Local Memory
//class DeviceLocalMemory : public DeviceLocalMemoryI
//{
//    private:
//        void*                       d_ptr;
//        size_t                      size;
//
//    protected:
//    public:
//        // Constructors & Destructor
//                                    DeviceLocalMemory(const GPUDevice& gpu);
//                                    DeviceLocalMemory(const GPUDevice& gpu, size_t sizeInBytes);
//                                    DeviceLocalMemory(const DeviceLocalMemory&);
//                                    DeviceLocalMemory(DeviceLocalMemory&&) noexcept;
//                                    ~DeviceLocalMemory();
//        DeviceLocalMemory&          operator=(const DeviceLocalMemory&);
//        DeviceLocalMemory&          operator=(DeviceLocalMemory&&) noexcept;
//
//        // Access
//        template<class T>
//        constexpr explicit          operator T* ();
//        template<class T>
//        constexpr explicit          operator const T* () const;
//        constexpr                   operator void* ();
//        constexpr                   operator const void* () const;
//
//        // Misc
//        size_t                      Size() const override;
//
//        void                        MigrateToOtherDevice(const GPUDevice* deviceTo, cudaStream_t stream = (cudaStream_t)0) override;
//};
//
//// Generic Device Memory (most of the cases this should be used)
//// Automatic multi-device seperation etc.
//class DeviceMemory : public DeviceMemoryI
//{
//    private:
//        void*                       m_ptr;
//        size_t                      size;
//        const GPUSystem&            system;
//
//
//    protected:
//    public:
//        // Constructors & Destructor
//                                    DeviceMemory(GPUSystem&);
//                                    DeviceMemory(GPUSystem&, size_t sizeInBytes);
//                                    DeviceMemory(const DeviceMemory&);
//                                    DeviceMemory(DeviceMemory&&);
//                                    ~DeviceMemory();
//        DeviceMemory&               operator=(const DeviceMemory&);
//        DeviceMemory&               operator=(DeviceMemory&&);
//
//        // Access
//        template<class T>
//        constexpr explicit          operator T*();
//        template<class T>
//        constexpr explicit          operator const T*() const;
//        constexpr                   operator void*();
//        constexpr                   operator const void*() const;
//
//        // Misc
//        size_t                      Size() const override;
//};
//
//namespace GPUMemFuncs
//{
//    template <class GPUMem>
//    void    EnlargeBuffer(GPUMem&, size_t);
//
//    template <class GPUMem, class... Args>
//    void     AllocateMultiData(std::tuple<Args*&...> pointers, GPUMem& memory,
//                               const std::array<size_t, sizeof...(Args)>& sizeList,
//                               size_t alignment = 256);
//}
//
//inline const GPUDeviceCUDA& DeviceLocalMemoryI::Device() const
//{
//    return currentDevice;
//}
//
//template<class T>
//inline constexpr DeviceLocalMemory::operator T* ()
//{
//    return reinterpret_cast<T*>(d_ptr);
//}
//
//template<class T>
//inline constexpr DeviceLocalMemory::operator const T* () const
//{
//    return reinterpret_cast<T*>(d_ptr);
//}
//
//inline constexpr DeviceLocalMemory::operator void* ()
//{
//    return d_ptr;
//}
//
//inline constexpr DeviceLocalMemory::operator const void* () const
//{
//    return d_ptr;
//}
//
//template<class T>
//inline constexpr DeviceMemory::operator T*()
//{
//    return reinterpret_cast<T*>(m_ptr);
//}
//
//template<class T>
//inline constexpr DeviceMemory::operator const T*() const
//{
//    return reinterpret_cast<T*>(m_ptr);
//}
//
//inline constexpr DeviceMemory::operator void*()
//{
//    return m_ptr;
//}
//
//inline constexpr DeviceMemory::operator const void*() const
//{
//    return m_ptr;
//}
//
//template <class GPUMem>
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
//    AcquireTotalSize(std::array<size_t, sizeof...(Tp)>&,
//                     const std::array<size_t, sizeof...(Tp)>&,
//                     size_t)
//    {
//        return 0;
//    }
//
//    template<std::size_t I = 0, class... Tp>
//    inline typename std::enable_if<(I < sizeof...(Tp)), size_t>::type
//    AcquireTotalSize(std::array<size_t, sizeof...(Tp)>& alignedSizeList,
//                     const std::array<size_t, sizeof...(Tp)>& countList,
//                     size_t alignment)
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
//    CalculatePointers(std::tuple<Tp*&...>&, size_t&, Byte*,
//                      const std::array<size_t, sizeof...(Tp)>&)
//    {}
//
//    template<std::size_t I = 0, class... Tp>
//    inline typename std::enable_if<(I < sizeof...(Tp)), void>::type
//    CalculatePointers(std::tuple<Tp*&...>& t, size_t& offset, Byte* memory,
//                      const std::array<size_t, sizeof...(Tp)>& alignedSizeList)
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
//
//}