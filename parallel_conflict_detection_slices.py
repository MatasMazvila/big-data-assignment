# Define the parallel conflict detection function that doesn't use sequential_time as an argument

from concurrent.futures import ProcessPoolExecutor
import psutil
import time
import numpy as np
import pandas as pd
from conflict_utils import detect_conflicts_chunk
import threading
import queue

def monitor_resources(interval=0.5, stop_event=None, usage_log=None):
    process = psutil.Process()
    while not stop_event.is_set():
        mem = process.memory_info().rss / (1024 ** 2)  # MB
        cpu = psutil.cpu_percent(interval=None)
        usage_log.put((mem, cpu))
        time.sleep(interval)

def parallel_conflict_detection_without_speedup(df, max_distance_km=0.02, max_time_diff_sec=10, num_workers=6):
    chunk_size = int(np.ceil(len(df) / num_workers))
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

    # Start resource monitoring in a separate thread
    usage_log = queue.Queue()
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, kwargs={"stop_event": stop_event, "usage_log": usage_log})
    monitor_thread.start()

    # Run the detection
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(
            detect_conflicts_chunk,
            chunks,
            [max_distance_km] * len(chunks),
            [max_time_diff_sec] * len(chunks)
        )
    total_time = time.perf_counter() - start

    # Stop the monitor
    stop_event.set()
    monitor_thread.join()

    # Collect usage data
    mem_values, cpu_values = [], []
    while not usage_log.empty():
        mem, cpu = usage_log.get()
        mem_values.append(mem)
        cpu_values.append(cpu)

    avg_mem = round(np.mean(mem_values), 2) if mem_values else None
    avg_cpu = round(np.mean(cpu_values), 2) if cpu_values else None

    # Compile conflicts
    all_conflict_indices = set()
    for chunk, idxs in zip(chunks, results):
        all_conflict_indices.update(chunk.index[idx] for idx in idxs)

    df = df.copy()
    df["position_conflict"] = False
    df.loc[list(all_conflict_indices), "position_conflict"] = True
    conflict_count = df["position_conflict"].sum()

    # Output
    print(f"Parallel Time ({num_workers} workers): {total_time:.2f} seconds")
    print(f"Conflicts Found (Parallel):   {conflict_count}")
    print(f"Avg Memory Usage:             {avg_mem} MB")
    print(f"Avg CPU Usage:                {avg_cpu}%")

    return total_time, conflict_count, avg_mem, avg_cpu, df



