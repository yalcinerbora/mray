#pragma once

#include "Core/TracerI.h"
#include "TracerTypes.h"

template<class KeyType>
struct GroupIdFetcher
{
    typename KeyType::Type operator()(auto id)
    {
        return std::bit_cast<KeyType>(id).FetchBatchPortion();
    }
};

using PrimGroupIdFetcher = GroupIdFetcher<PrimBatchKey>;
using TransGroupIdFetcher = GroupIdFetcher<TransformKey>;
using LightGroupIdFetcher = GroupIdFetcher<LightKey>;

// Comparison Routines
inline bool SurfaceLessThan(const Pair<SurfaceId, SurfaceParams>& left,
                            const Pair<SurfaceId, SurfaceParams>& right)
{
    PrimBatchKey lpk = std::bit_cast<PrimBatchKey>(left.second.primBatches.front());
    TransformKey ltk = std::bit_cast<TransformKey>(left.second.transformId);
    //
    PrimBatchKey rpk = std::bit_cast<PrimBatchKey>(right.second.primBatches.front());
    TransformKey rtk = std::bit_cast<TransformKey>(right.second.transformId);
    //
    using T = std::tuple<CommonKey, CommonKey>;
    return (T(lpk.FetchBatchPortion(), ltk.FetchBatchPortion()) <
            T(rpk.FetchBatchPortion(), rtk.FetchBatchPortion()));
}

inline bool LightSurfaceLessThan(const Pair<LightSurfaceId, LightSurfaceParams>& left,
                                 const Pair<LightSurfaceId, LightSurfaceParams>& right)
{
    LightKey llk = std::bit_cast<LightKey>(left.second.lightId);
    TransformKey ltk = std::bit_cast<TransformKey>(left.second.transformId);
    //
    LightKey rlk = std::bit_cast<LightKey>(right.second.lightId);
    TransformKey rtk = std::bit_cast<TransformKey>(right.second.transformId);
    //
    using T = std::tuple<CommonKey, CommonKey>;
    return (T(llk.FetchBatchPortion(), ltk.FetchBatchPortion()) <
            T(rlk.FetchBatchPortion(), rtk.FetchBatchPortion()));
}

inline bool CamSurfaceLessThan(const Pair<CamSurfaceId, CameraSurfaceParams>& left,
                               const Pair<CamSurfaceId, CameraSurfaceParams>& right)
{
    CameraKey lck = std::bit_cast<CameraKey>(left.second.cameraId);
    TransformKey ltk = std::bit_cast<TransformKey>(left.second.transformId);
    //
    CameraKey rck = std::bit_cast<CameraKey>(right.second.cameraId);
    TransformKey rtk = std::bit_cast<TransformKey>(right.second.transformId);
    //
    using T = std::tuple<CommonKey, CommonKey>;
    return (T(lck.FetchBatchPortion(), ltk.FetchBatchPortion()) <
            T(rck.FetchBatchPortion(), rtk.FetchBatchPortion()));
}
