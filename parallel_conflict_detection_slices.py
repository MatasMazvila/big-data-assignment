# Define the parallel conflict detection function that doesn't use sequential_time as an argument

from concurrent.futures import ProcessPoolExecutor
import psutil
import time
import numpy as np
import pandas as pd
from conflict_utils import detect_conflicts_chunk

def parallel_conflict_detection_without_speedup(df, max_distance_km=0.02, max_time_diff_sec=10, num_workers=6):
    start = time.perf_counter()
    chunk_size = int(np.ceil(len(df) / num_workers))
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(
            detect_conflicts_chunk,
            chunks,
            [max_distance_km] * len(chunks),
            [max_time_diff_sec] * len(chunks)
        )

    all_conflict_indices = set()
    for chunk, idxs in zip(chunks, results):
        all_conflict_indices.update(chunk.index[idx] for idx in idxs)

    df = df.copy()
    df["position_conflict"] = False
    df.loc[list(all_conflict_indices), "position_conflict"] = True

    total_time = time.perf_counter() - start
    conflict_count = df["position_conflict"].sum()

    process = psutil.Process()
    mem_usage = process.memory_info().rss / (1024 ** 2)
    cpu_usage = psutil.cpu_percent(interval=1)

    print(f"Parallel Time ({num_workers} workers): {total_time:.2f} seconds")
    print(f"Conflicts Found (Parallel):   {conflict_count}")
    print(f"Memory Usage:                 {mem_usage:.2f} MB")
    print(f"CPU Usage:                    {cpu_usage:.2f}%")

    return total_time, conflict_count, mem_usage, cpu_usage, df


