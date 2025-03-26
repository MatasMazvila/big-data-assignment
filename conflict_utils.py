# This file contains the function to detect conflicts between vessels in a chunk of data

import numpy as np
from scipy.spatial import cKDTree

def detect_conflicts_chunk(df_chunk, max_distance_km, max_time_diff_sec):
    df_chunk = df_chunk[
        (df_chunk["Latitude"].between(-90, 90)) &
        (df_chunk["Longitude"].between(-180, 180))
    ].copy()

    coords_rad = np.radians(df_chunk[["Latitude", "Longitude"]].values)
    timestamps = df_chunk["# Timestamp"].astype("int64").values // 1_000_000_000
    mmsis = df_chunk["MMSI"].values

    radius_rad = max_distance_km / 6371
    tree = cKDTree(coords_rad)
    neighbors = tree.query_ball_point(coords_rad, r=radius_rad)

    conflict_flags = set()
    for i, idx_list in enumerate(neighbors):
        for j in idx_list:
            if i == j or mmsis[i] == mmsis[j]:
                continue
            if abs(timestamps[i] - timestamps[j]) <= max_time_diff_sec:
                conflict_flags.add(i)
                break

    return list(conflict_flags)