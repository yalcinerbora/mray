#pragma once

#include "Core/TracerI.h"
#include "Core/Vector.h"

#include "TransientPool/TransientPool.h"

#include "TextureView.h"
#include "ParamVaryingData.h"

using AttributeRanges = StaticVector<Vector<2, size_t>,
                                     TracerConstants::MaxAttributePerGroup>;

template <class GenericGroupType>
concept GenericGroupC = requires(GenericGroupType gg,
                                 TransientData input,
                                 typename GenericGroupType::Id id,
                                 const GPUQueue& q)
{
    typename GenericGroupType::Id;
    typename GenericGroupType::IdInt;
    typename GenericGroupType::IdList;
    typename GenericGroupType::AttribInfoList;

    {gg.Reserve(std::vector<AttributeCountList>{})
    } -> std::same_as<typename GenericGroupType::IdList>;
    {gg.CommitReservations()} -> std::same_as<void>;
    {gg.IsInCommitState()} -> std::same_as<bool>;
    {gg.AttributeInfo()
    } -> std::same_as<typename GenericGroupType::AttribInfoList>;
    {gg.PushAttribute(id, uint32_t{}, std::move(input), q)
    } -> std::same_as<void>;
    {gg.PushAttribute(id, uint32_t{}, Vector2ui{}, std::move(input), q)
    } ->std::same_as<void>;
    {gg.PushAttribute(id, id, uint32_t{}, std::move(input), q)
    } ->std::same_as<void>;
    {gg.GPUMemoryUsage()} -> std::same_as<size_t>;

    // Can query the type
    {gg.Name()} -> std::same_as<std::string_view>;
    {GenericGroupType::TypeName()} -> std::same_as<std::string_view>;
};

template <class IdTypeT, class AttribInfoT>
class GenericGroupI
{
    public:
    using Id                = IdTypeT;
    using IdInt             = typename Id::Type;
    using IdList            = std::vector<Id>;
    using AttribInfo        = AttribInfoT;
    using AttribInfoList    = StaticVector<AttribInfo, TracerConstants::MaxAttributePerGroup>;

    public:
    virtual                 ~GenericGroupI() = default;
    //
    virtual IdList          Reserve(const std::vector<AttributeCountList>&) = 0;
    virtual void            CommitReservations() = 0;
    virtual bool            IsInCommitState() const = 0;
    virtual void            PushAttribute(Id id, uint32_t attributeIndex,
                                          TransientData data,
                                          const GPUQueue& queue) = 0;
    virtual void            PushAttribute(Id id, uint32_t attributeIndex,
                                          const Vector2ui& subRange,
                                          TransientData data,
                                          const GPUQueue& queue) = 0;
    virtual void            PushAttribute(Id idStart, Id idEnd,
                                          uint32_t attributeIndex,
                                          TransientData data,
                                          const GPUQueue& queue) = 0;
    // Finalize functionality,
    // This will be called just before commit surfaces, once.
    virtual void                Finalize(const GPUQueue&) = 0;
    //
    virtual size_t              GPUMemoryUsage() const = 0;
    virtual AttribInfoList      AttributeInfo() const = 0;
    virtual std::string_view    Name() const = 0;
    virtual IdInt               GroupId() const = 0;
    virtual size_t              TotalItemCount() const = 0;
};

// Implementation of the common parts
template<class IdTypeT, class AttribInfoT>
class GenericGroupT : public GenericGroupI<IdTypeT, AttribInfoT>
{
    static constexpr size_t MapReserveSize = 512;

    public:
    using InterfaceType = GenericGroupI<IdTypeT, AttribInfoT>;
    using BaseType      = GenericGroupT<IdTypeT, AttribInfoT>;
    using typename InterfaceType::Id;
    using typename InterfaceType::IdInt;
    using typename InterfaceType::IdList;
    using typename InterfaceType::AttribInfo;
    using typename InterfaceType::AttribInfoList;

    using ItemRangeMap  = Map<IdInt, AttributeRanges>;
    using ItemCountMap  = Map<IdInt, AttributeCountList>;

    protected:
    ItemRangeMap        itemRanges;
    ItemCountMap        itemCounts;

    const GPUSystem&    gpuSystem;
    std::atomic_bool    isCommitted;
    std::mutex          mutex;
    IdInt               groupId;
    DeviceMemory        deviceMem;

    const AttributeRanges&      FindRange(IdInt) const;

    template <class... Args>
    void GenericCommit(Tuple<Span<Args>&...> output,
                       std::array<int32_t, sizeof...(Args)> countLookup);

    template <class T>
    void GenericPushData(const Span<T>& dAttributeRegion,
                         //
                         IdInt id, uint32_t attribIndex,
                         TransientData data,
                         const GPUQueue& deviceQueue) const;
    template <class T>
    void GenericPushData(const Span<T>& dAttributeRegion,
                         //
                         Vector<2, IdInt> idRange,
                         uint32_t attribIndex,
                         TransientData data,
                         const GPUQueue& deviceQueue) const;
    template <class T>
    void GenericPushData(const Span<T>& dAttributeRegion,
                         //
                         IdInt id, uint32_t attribIndex,
                         const Vector2ui& subRange,
                         TransientData data,
                         const GPUQueue& deviceQueue) const;

    public:
            GenericGroupT(uint32_t groupId, const GPUSystem&,
                          size_t allocationGranularity = 2_MiB,
                          size_t initialReservationSize = 4_MiB);
    IdList  Reserve(const std::vector<AttributeCountList>&) override;
    bool    IsInCommitState() const override;
    size_t  GPUMemoryUsage() const override;
    IdInt   GroupId() const override;
    size_t  TotalItemCount() const override;
    // Finalize is useful on rare occasions so we default it to empty
    void    Finalize(const GPUQueue&) override {};
};

template<class IdType, class AttributeInfoType>
class GenericTexturedGroupT : public GenericGroupT<IdType, AttributeInfoType>
{
    using Parent = GenericGroupT<IdType, AttributeInfoType>;

    template<uint32_t D, class T>
    std::vector<TracerTexView<D, T>>
    ConvertToView(std::vector<TextureId> texIds,
                  uint32_t attributeIndex) const;

    template<uint32_t D, class T>
    std::vector<Optional<TracerTexView<D, T>>>
    ConvertToView(std::vector<Optional<TextureId>> texIds,
                  uint32_t attributeIndex) const;

    protected:
    const TextureViewMap& globalTextureViews;

    template<uint32_t D, class T>
    void            GenericPushTexAttribute(Span<ParamVaryingData<D, T>>,
                                            //
                                            IdType idStart, IdType idEnd,
                                            uint32_t attributeIndex,
                                            TransientData,
                                            std::vector<Optional<TextureId>>,
                                            const GPUQueue& queue);
    template<uint32_t D, class T>
    void            GenericPushTexAttribute(Span<Optional<TracerTexView<D, T>>>,
                                            //
                                            IdType idStart, IdType idEnd,
                                            uint32_t attributeIndex,
                                            std::vector<Optional<TextureId>>,
                                            const GPUQueue& queue);
    template<uint32_t D, class T>
    void            GenericPushTexAttribute(Span<TracerTexView<D, T>>,
                                            //
                                            IdType idStart, IdType idEnd,
                                            uint32_t attributeIndex,
                                            std::vector<TextureId>,
                                            const GPUQueue& queue);

    public:
    // Constructors & Destructor
    GenericTexturedGroupT(uint32_t groupId, const GPUSystem&,
                          const TextureViewMap&,
                          size_t allocationGranularity = 2_MiB,
                          size_t initialReservationSize = 4_MiB);

    // Extra textured functionality
    virtual void    PushTexAttribute(IdType idStart, IdType idEnd,
                                     uint32_t attributeIndex,
                                     TransientData,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) = 0;
    virtual void    PushTexAttribute(IdType idStart, IdType idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<Optional<TextureId>>,
                                     const GPUQueue& queue) = 0;
    virtual void    PushTexAttribute(IdType idStart, IdType idEnd,
                                     uint32_t attributeIndex,
                                     std::vector<TextureId>,
                                     const GPUQueue& queue) = 0;
};

// Due to high amount of instantiations we pre-generate
// these classes.
//
// TODO: This somewhat of a bullshit impl.
// This should be type-ereased etc. to minimize templates
//
// ========================== //
//     GENERIC PUSH NORMAL    //
// ========================== //
#define MRAY_GENERIC_PUSH_ATTRIB_INST_0(I, A, T) \
    template void GenericGroupT<I, A>::          \
    GenericPushData                              \
    (                                            \
        const Span<T>&, typename I::Type,        \
        uint32_t, TransientData, const GPUQueue& \
    ) const

#define MRAY_GENERIC_PUSH_ATTRIB_INST_1(I, A, T) \
    template void GenericGroupT<I, A>::          \
    GenericPushData                              \
    (                                            \
        const Span<T>&,                          \
        Vector<2, typename I::Type>,             \
        uint32_t, TransientData,                 \
        const GPUQueue&                          \
    ) const

#define MRAY_GENERIC_PUSH_ATTRIB_INST_2(I, A, T) \
    template void GenericGroupT<I, A>::          \
    GenericPushData                              \
    (                                            \
        const Span<T>&, typename I::Type,        \
        uint32_t, const Vector2ui&,              \
        TransientData, const GPUQueue&           \
    ) const

#define MRAY_GENERIC_PUSH_ATTRIB_INST(E, I, A, T) \
    E MRAY_GENERIC_PUSH_ATTRIB_INST_0(I, A, T);   \
    E MRAY_GENERIC_PUSH_ATTRIB_INST_1(I, A, T);   \
    E MRAY_GENERIC_PUSH_ATTRIB_INST_2(I, A, T)


// ========================== //
//     GENERIC PUSH TEXTURE   //
// ========================== //
#define MRAY_GENERIC_TEX_PUSH_ATTRIB_INST_0(I, A, D, T)     \
    template void GenericTexturedGroupT<I, A>::             \
    GenericPushTexAttribute                                 \
    (                                                       \
        Span<ParamVaryingData<D, T>>,                       \
        I, I, uint32_t, TransientData,                      \
        std::vector<Optional<TextureId>>,                   \
        const GPUQueue&                                     \
    );

#define MRAY_GENERIC_TEX_PUSH_ATTRIB_INST_1(I, A, D, T)     \
    template void GenericTexturedGroupT<I, A>::             \
    GenericPushTexAttribute                                 \
    (                                                       \
        Span<Optional<TracerTexView<D, T>>>,                \
        I, I, uint32_t,                                     \
        std::vector<Optional<TextureId>>,                   \
        const GPUQueue&                                     \
    );

#define MRAY_GENERIC_TEX_PUSH_ATTRIB_INST_2(I, A, D, T)     \
    template void GenericTexturedGroupT<I, A>::             \
    GenericPushTexAttribute                                 \
    (                                                       \
        Span<TracerTexView<D, T>>,                          \
        I, I, uint32_t,                                     \
        std::vector<TextureId>,const GPUQueue&              \
    )

// We only instantiate Float / Vector2 / Vector3 since for
// textures attributes it should be enough. We can add more later
#define MRAY_GENERIC_TEX_PUSH_ATTRIB_INST(E, I, A, D)        \
    E MRAY_GENERIC_TEX_PUSH_ATTRIB_INST_0(I, A, D, Float);   \
    E MRAY_GENERIC_TEX_PUSH_ATTRIB_INST_1(I, A, D, Float);   \
    E MRAY_GENERIC_TEX_PUSH_ATTRIB_INST_2(I, A, D, Float);   \
    E MRAY_GENERIC_TEX_PUSH_ATTRIB_INST_0(I, A, D, Vector2); \
    E MRAY_GENERIC_TEX_PUSH_ATTRIB_INST_1(I, A, D, Vector2); \
    E MRAY_GENERIC_TEX_PUSH_ATTRIB_INST_2(I, A, D, Vector2); \
    E MRAY_GENERIC_TEX_PUSH_ATTRIB_INST_0(I, A, D, Vector3); \
    E MRAY_GENERIC_TEX_PUSH_ATTRIB_INST_1(I, A, D, Vector3); \
    E MRAY_GENERIC_TEX_PUSH_ATTRIB_INST_2(I, A, D, Vector3)

// ================== //
//   INSTANTIATIONS   //
// ================== //
//     PRIMITIVE      //
// ================== //
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, PrimBatchKey, PrimAttributeInfo, Float);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, PrimBatchKey, PrimAttributeInfo, Vector2);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, PrimBatchKey, PrimAttributeInfo, Vector3);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, PrimBatchKey, PrimAttributeInfo, Vector3ui);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, PrimBatchKey, PrimAttributeInfo, Vector4);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, PrimBatchKey, PrimAttributeInfo, Quaternion);
// Skeleton-related
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, PrimBatchKey, PrimAttributeInfo, Vector4uc);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, PrimBatchKey, PrimAttributeInfo, UNorm4x8);

// ================== //
//     TRANSFORM      //
// ================== //
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, TransformKey, TransAttributeInfo, Matrix4x4);

// ================== //
//       CAMERA       //
// ================== //
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, CameraKey, CamAttributeInfo, Float);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, CameraKey, CamAttributeInfo, Vector2);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, CameraKey, CamAttributeInfo, Vector3);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, CameraKey, CamAttributeInfo, Vector4);

// ================== //
//       MEDIUM       //
// ================== //
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, MediumKey, MediumAttributeInfo, Float);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, MediumKey, MediumAttributeInfo, Vector2);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, MediumKey, MediumAttributeInfo, Vector3);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, MediumKey, MediumAttributeInfo, Vector4);
//
MRAY_GENERIC_TEX_PUSH_ATTRIB_INST(extern, MediumKey, MediumAttributeInfo, 3);

// ================== //
//       LIGHT        //
// ================== //
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, LightKey, LightAttributeInfo, Float);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, LightKey, LightAttributeInfo, Vector2);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, LightKey, LightAttributeInfo, Vector3);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, LightKey, LightAttributeInfo, Vector4);
//
MRAY_GENERIC_TEX_PUSH_ATTRIB_INST(extern, LightKey, LightAttributeInfo, 2);

// ================== //
//      MATERIAL      //
// ================== //
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, MaterialKey, MatAttributeInfo, Float);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, MaterialKey, MatAttributeInfo, Vector2);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, MaterialKey, MatAttributeInfo, Vector3);
MRAY_GENERIC_PUSH_ATTRIB_INST(extern, MaterialKey, MatAttributeInfo, Vector4);
//
MRAY_GENERIC_TEX_PUSH_ATTRIB_INST(extern, MaterialKey, MatAttributeInfo, 2);

// ================== //
//      CLASSES       //
// ================== //
// Non-Textured
extern template class GenericGroupT<PrimBatchKey, PrimAttributeInfo>;
extern template class GenericGroupT<CameraKey, CamAttributeInfo>;
extern template class GenericGroupT<TransformKey, TransAttributeInfo>;
// Textured
extern template class GenericTexturedGroupT<MediumKey, MediumAttributeInfo>;
extern template class GenericTexturedGroupT<MaterialKey, MatAttributeInfo>;
extern template class GenericTexturedGroupT<LightKey, LightAttributeInfo>;
