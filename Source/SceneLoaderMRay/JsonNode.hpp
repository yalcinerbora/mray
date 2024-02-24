#pragma once

inline TextureAccessLayout LoadTextureAccessLayout(const nlohmann::json& node)
{
    using namespace std::literals;
    std::string_view l = node;
    if(l == "r"sv)  return TextureAccessLayout::R;
    if(l == "g"sv)  return TextureAccessLayout::G;
    if(l == "b"sv)  return TextureAccessLayout::B;
    if(l == "a"sv)  return TextureAccessLayout::A;
    //
    if(l == "rg"sv)  return TextureAccessLayout::RG;
    if(l == "gb"sv)  return TextureAccessLayout::GB;
    if(l == "ba"sv)  return TextureAccessLayout::BA;
    //
    if(l == "rgb"sv)  return TextureAccessLayout::RGB;
    if(l == "gba"sv)  return TextureAccessLayout::GBA;
    //
    if(l == "rgba"sv)  return TextureAccessLayout::RGBA;

    else throw MRayError("Unknown texture access layout");
}

inline void from_json(const nlohmann::json& n, NodeTexStruct& ts)
{
    using namespace NodeNames;
    ts.texId = n.at(TEXTURE_NAME);
    ts.channelLayout = LoadTextureAccessLayout(n.at(TEXTURE_CHANNEL));
}

inline void from_json(const nlohmann::json& n, SurfaceStruct& s)
{
    auto it = n.find(NodeNames::TRANSFORM);
    s.transformId = (it == n.end())
                        ? EMPTY_TRANSFORM
                        : it->get<uint32_t>();

    const auto& matArray = n.at(NodeNames::MATERIAL);
    const auto& primArray = n.at(NodeNames::PRIMITIVE);
    if(matArray.size() != primArray.size())
        throw MRayError("Material/Primitive pair lists does not match on a surface!");

    // As a default we do back face culling (it is helpful for self occlusions)
    s.doCullBackFace.fill(true);
    // and no alpha maps (all prims are opaque)
    s.alphaMaps.fill(std::nullopt);

    if(matArray.is_number_integer() &&
       primArray.is_number_integer())
    {
        typename SurfaceStruct::IdPair p;
        std::get<SurfaceStruct::MATERIAL_INDEX>(p) = matArray;
        std::get<SurfaceStruct::PRIM_INDEX>(p) = primArray;
        s.pairCount = 1;
        s.matPrimBatchPairs[0] = p;

        auto alphaIt = n.find(NodeNames::ALPHA_MAP);
        if(alphaIt != n.cend())
            s.alphaMaps[0] = (*alphaIt).get<NodeTexStruct>();

        auto cullIt = n.find(NodeNames::CULL_FACE);
        if(cullIt != n.cend())
            s.doCullBackFace[0] = (*cullIt).get<bool>();
    }
    else
    {
        for(size_t i = 0; i < matArray.size(); i++)
        {
            typename SurfaceStruct::IdPair p;
            std::get<SurfaceStruct::MATERIAL_INDEX>(p) = matArray[i];
            std::get<SurfaceStruct::PRIM_INDEX>(p) = primArray[i];
            s.matPrimBatchPairs[i] = p;

            auto alphaIt = n.find(NodeNames::ALPHA_MAP);
            // Technically "-" should be supported only but
            // check if it is string here actual type is object.
            if(alphaIt != n.cend() && !alphaIt[i].is_string())
                s.alphaMaps[i] = (*alphaIt)[i].get<NodeTexStruct>();

            auto cullIt = n.find(NodeNames::CULL_FACE);
            if(cullIt != n.cend() && !cullIt[i].is_string())
                s.doCullBackFace[i] = (*cullIt)[i].get<bool>();

        }
        s.pairCount = static_cast<uint8_t>(matArray.size());
    }
}

inline void from_json(const nlohmann::json& n, LightSurfaceStruct& s)
{
    auto itT = n.find(NodeNames::TRANSFORM);
    s.transformId = (itT == n.end())
                        ? EMPTY_TRANSFORM
                        : itT->get<uint32_t>();
    auto itM = n.find(NodeNames::MEDIUM);
    s.mediumId = (itM == n.end())
                    ? EMPTY_MEDIUM
                    : itM->get<uint32_t>();

    s.lightId = n.at(NodeNames::LIGHT);
}

inline void from_json(const nlohmann::json& n, CameraSurfaceStruct& s)
{
    auto itT = n.find(NodeNames::TRANSFORM);
    s.transformId = (itT == n.end())
                        ? EMPTY_TRANSFORM
                        : itT->get<uint32_t>();
    auto itM = n.find(NodeNames::MEDIUM);
    s.mediumId = (itM == n.end())
                    ? EMPTY_MEDIUM
                    : itM->get<uint32_t>();

    s.cameraId = n.at(NodeNames::CAMERA);
}

template<ArrayLikeC T>
void from_json(const nlohmann::json& n, T& out)
{
    using IT = typename T::InnerType;
    using S = Span<IT, T::Dims>;
    std::array<IT, T::Dims> a = n;
    out = T(ToConstSpan(S(a)));
}

template<std::floating_point T>
void from_json(const nlohmann::json& n, Quat<T>& out)
{
    using V = Vector<4, T>;
    using S = Span<T, 4>;
    std::array<T, 4> a = n;
    out = Quat<T>(V(ToConstSpan(S(a))));
}

template<std::floating_point T>
void from_json(const nlohmann::json& n, AABB<3, T>& out)
{
    using V = Vector<3, T>;
    using S = Span<T, 3>;
    std::array<T, 3> v0 = n.at(0);
    std::array<T, 3> v1 = n.at(1);
    out = AABB<3, T>(V(ToConstSpan(S(v0))),
                     V(ToConstSpan(S(v1))));
}

template<std::floating_point T>
void from_json(const nlohmann::json& n, RayT<T>& out)
{
    using V = Vector<3, T>;
    using S = Span<T, 3>;
    std::array<T, 3> v0 = n.at(0);
    std::array<T, 3> v1 = n.at(1);
    out = RayT<T>(V(ToConstSpan(S(v0))),
                  V(ToConstSpan(S(v1))));
}

inline JsonNode::JsonNode(const nlohmann::json& node, uint32_t innerIndex)
    : node(node)
    , innerIndex(innerIndex)
{
    // Check the multi-nodeness
    auto n = node.at(NodeNames::ID);
    isMultiNode = n.is_array();
}

inline const nlohmann::json& JsonNode::RawNode() const
{
    return node;
}

inline std::string_view JsonNode::Type() const
{
    return node.at(NodeNames::TYPE);
}

inline std::string_view JsonNode::Tag() const
{
    return node.at(NodeNames::TAG);
}

inline uint32_t JsonNode::Id() const
{
    const auto& nodeArray = (isMultiNode) ? node.at(NodeNames::ID).at(innerIndex)
                                          : node.at(NodeNames::ID);
    return nodeArray.get<uint32_t>();
}

// Inner node unspecific data access
template<class T>
T JsonNode::CommonData(std::string_view name) const
{
    return node.at(name).get<T>();
}

template<class T>
MRayInput JsonNode::CommonDataArray(std::string_view name) const
{
    const auto& nodeArray = node.at(name);
    MRayInput input(std::in_place_type_t<T>{}, nodeArray.size());
    for(const auto n : nodeArray)
    {
        input.Push(n.get<T>());
    }
    return std::move(input);
}

// Inner index related data loading
template<class T>
T JsonNode::AccessData(std::string_view name) const
{
    const auto& n = (isMultiNode) ? node.at(name).at(innerIndex)
                                  : node.at(name);
    return n.get<T>();
}

template<class T>
MRayInput JsonNode::AccessDataArray(std::string_view name) const
{
    const auto& nodeArray = (isMultiNode) ? node.at(name).at(innerIndex)
                                          : node.at(name);

    MRayInput input(std::in_place_type_t<T>{}, nodeArray.size());
    for(const auto n : nodeArray)
    {
        input.Push(n.get<T>());
    }
    return std::move(input);
}
// Optional Data
template<class T>
Optional<T> JsonNode::AccessOptionalData(std::string_view name) const
{
    // Entire entry is missing (which is not defined all items on this node)
    if(node.find(name) == node.cend()) return std::nullopt;

    // Elements are available fetch the inner entry
    const auto& n = (isMultiNode) ? node.at(name).at(innerIndex)
                                  : node.at(name);

    // Here user may provide "-" string which means data is
    // not present for this item in node
    using namespace std::literals;
    if(n.is_string() && n.get<std::string_view>() == "-"sv)
        return std::nullopt;
    else
        return n.get<T>();
}

template<class T>
Optional<MRayInput> JsonNode::AccessOptionalDataArray(std::string_view name) const
{
    // Entire entry is missing (which is not defined all items on this node)
    if(node.find(name) == node.cend()) return std::nullopt;

    // Elements are available fetch the inner entry
    const auto& nodeArray = (isMultiNode) ? node.at(name).at(innerIndex)
                                          : node.at(name);

    // Here user may provide "-" string which means data is
    // not present for this item in node
    MRayInput input(std::in_place_type_t<T>{}, nodeArray.size());
    using namespace std::literals;
    if(nodeArray.is_string() &&
       nodeArray.get<std::string_view>() == "-"sv)
        return std::nullopt;
    else
    {
        for(const auto n : nodeArray)
        {
            input.Push(n.get<T>());
        }
        return std::move(input);
    }
}

// Texturable (either data T, or texture struct)
template<class T>
Variant<NodeTexStruct, T> JsonNode::AccessTexturableData(std::string_view name) const
{
    using V = Variant<NodeTexStruct, T>;
    const auto& n = (isMultiNode) ? node.at(name).at(innerIndex)
                                  : node.at(name);
    return (n.is_object()) ? V(n.get<NodeTexStruct>())
                           : V(n.get<T>());
}

template<class T>
std::vector<Variant<NodeTexStruct, T>> JsonNode::AccessTexturableDataArray(std::string_view name) const
{
    using V = Variant<NodeTexStruct, T>;
    const auto& nArray = (isMultiNode) ? node.at(name).at(innerIndex)
                                  : node.at(name);

    std::vector<Variant<NodeTexStruct, T>> output; output.reserve(nArray.size());
    for(const auto& n : nArray)
    {
        Variant<NodeTexStruct, T> v = (n.is_object()) ? V(n.get<NodeTexStruct>())
                                                      : V(n.get<T>());
        output.push_back(v);
    }
}
