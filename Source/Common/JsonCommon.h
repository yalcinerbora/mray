#pragma once

#include "Core/Vector.h"
#include "Core/AABB.h"
#include "Core/Quaternion.h"
#include "Core/Ray.h"

#include <nlohmann/json.hpp>

// Json converters
template<ArrayLikeC T>
void from_json(const nlohmann::json&, T&);
template<std::floating_point T>
void from_json(const nlohmann::json&, Quat<T>&);
template<std::floating_point T>
void from_json(const nlohmann::json&, AABB<3, T>&);
template<std::floating_point T>
void from_json(const nlohmann::json&, RayT<T>&);

template<ArrayLikeC T>
inline
void from_json(const nlohmann::json& n, T& out)
{
    using IT = typename T::InnerType;
    using S = Span<IT, T::Dims>;
    std::array<IT, T::Dims> a = n;
    out = T(ToConstSpan(S(a)));
}

template<std::floating_point T>
inline
void from_json(const nlohmann::json& n, Quat<T>& out)
{
    using V = Vector<4, T>;
    using S = Span<T, 4>;
    std::array<T, 4> a = n;
    out = Quat<T>(V(ToConstSpan(S(a))));
}

template<std::floating_point T>
inline
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
inline
void from_json(const nlohmann::json& n, RayT<T>& out)
{
    using V = Vector<3, T>;
    using S = Span<T, 3>;
    std::array<T, 3> v0 = n.at(0);
    std::array<T, 3> v1 = n.at(1);
    out = RayT<T>(V(ToConstSpan(S(v0))),
                  V(ToConstSpan(S(v1))));
}