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

inline void from_json(const nlohmann::json&, SurfaceStruct&)
{}

inline void from_json(const nlohmann::json&, LightSurfaceStruct&)
{}

inline void from_json(const nlohmann::json&, CameraSurfaceStruct&)
{}

template<ArrayLikeC T>
void from_json(const nlohmann::json& n, T& out)
{
    out = T(Span<const typename T::InnerType, T::DIMS>(n.cbegin(), n.cend()));
}

inline MRayJsonNode::MRayJsonNode(nlohmann::json& node, uint32_t innerIndex)
    : node(node)
    , innerIndex(innerIndex)
{
    // Check the multi-nodeness
    auto n = node.at(NodeNames::ID);
    isMultiNode = n.is_array();
}

inline std::string_view MRayJsonNode::Type() const
{
    return node.at(NodeNames::TYPE);
}

inline std::string_view MRayJsonNode::Tag() const
{
    return node.at(NodeNames::TAG);
}

inline uint32_t MRayJsonNode::Id() const
{
    const auto& nodeArray = (isMultiNode) ? node.at(NodeNames::ID).at(innerIndex)
                                          : node.at(NodeNames::ID);
    return nodeArray.get<uint32_t>();
}

// Inner node unspecific data access
template<class T>
T MRayJsonNode::CommonData(std::string_view name) const
{
    return node.at(name).get<T>();
}

template<class T>
MRayInput MRayJsonNode::CommonDataArray(std::string_view name) const
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
T MRayJsonNode::AccessData(std::string_view name) const
{
    const auto& n = (isMultiNode) ? node.at(name).at(innerIndex)
                                  : node.at(name);
    return n.get<T>();
}

template<class T>
MRayInput MRayJsonNode::AccessDataArray(std::string_view name) const
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
Optional<T> MRayJsonNode::AccessOptionalData(std::string_view name) const
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
Optional<MRayInput> MRayJsonNode::AccessOptionalDataArray(std::string_view name) const
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
Variant<NodeTexStruct, T> MRayJsonNode::AccessTexturableData(std::string_view name)
{
    const auto& n = (isMultiNode) ? node.at(name).at(innerIndex)
                                  : node.at(name);
    return (n.is_object()) ? n.get<NodeTexStruct>()
                           : n.get<T>();
}

template<class T>
std::vector<Variant<NodeTexStruct, T>> MRayJsonNode::AccessTexturableDataArray(std::string_view name)
{
    const auto& nArray = (isMultiNode) ? node.at(name).at(innerIndex)
                                  : node.at(name);

    std::vector<Variant<NodeTexStruct, T>> output; output.reserve(nArray.size());
    for(const auto& n : nArray)
    {
        Variant<NodeTexStruct, T> v = (n.is_object()) ? n.get<NodeTexStruct>()
                                                      : n.get<T>();
        output.push_back(v);
    }
}
