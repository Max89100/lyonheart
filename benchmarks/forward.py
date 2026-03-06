import psutil
import time
import os
import torch
import lyonheart as lh
import gc
import numpy as np

# MONOCORE 
p = psutil.Process()
p.cpu_affinity([0])
torch.set_num_threads(1)
os.environ["RAYON_NUM_THREADS"] = "1"


def bench_lyonheart(batch_size, iters):
    model = lh.nn.Sequential([lh.nn.Linear(784,128),lh.nn.ReLU(),lh.nn.Linear(128,10)])
    x = lh.randn((batch_size,784))
    # Warmup
    for _ in range(1000): _ = model(x)

    start = time.perf_counter_ns()
    for _ in range(iters):   
        y = model(x)
    return (time.perf_counter_ns() - start) / iters

def bench_pytorch(batch_size, iters):
    model = torch.nn.Sequential(torch.nn.Linear(784,128), torch.nn.ReLU(), torch.nn.Linear(128,10)).to(torch.float32)
    x = torch.randn(batch_size,784, dtype=torch.float32)
    #Warmup
    with torch.no_grad(): 
        for _ in range(1000): _ = model(x)

    with torch.no_grad():
        start = time.perf_counter_ns()
        for _ in range(iters):
            y = model(x)
    return (time.perf_counter_ns() - start) / iters

def run_full_benchmark(batch_size, n_trials=100, iters_per_trial=1000):
    lh_throughput = []
    pt_throughput = []
    print(f"--- Lancement du Benchmark (Batch Size: {batch_size}) ---")
    for i in range(n_trials):
        # LyonHeart
        t_lh = bench_lyonheart(batch_size, iters_per_trial) / 1e9
        lh_throughput.append(batch_size / t_lh)
        # PyTorch
        t_pt = bench_pytorch(batch_size, iters_per_trial) / 1e9
        pt_throughput.append(batch_size / t_pt)
        if (i + 1) % 10 == 0:
            print(f"Essai {i+1}/{n_trials} terminé...")

    stats = {
        "LyonHeart": {
            "max": np.max(lh_throughput),    # Ta vitesse de pointe (le "Min" temps)
            "median": np.median(lh_throughput),
            "p95": np.percentile(lh_throughput, 5), # seuil de 5% des pires perfs (lent)
            "std": np.std(lh_throughput)     # Écart-type (stabilité)
        },
        "PyTorch": {
            "max": np.max(pt_throughput),
            "median": np.median(pt_throughput),
            "p95": np.percentile(pt_throughput, 5),
            "std": np.std(pt_throughput)
        }
    }
    
    return stats

def print_report(stats):
    print("\n" + "="*40)
    print(f"{'METRIQUE':<15} | {'LYONHEART':<12} | {'PYTORCH':<12}")
    print("-"*40)
    for m in ["max", "median", "p95", "std"]:
        lh_v = stats["LyonHeart"][m]
        pt_v = stats["PyTorch"][m]
        print(f"{m.upper():<15} | {lh_v:>12.2f} | {pt_v:>12.2f}")
    
    ratio = stats["PyTorch"]["median"] / stats["LyonHeart"]["median"]
    print("-"*40)
    print(f"RATIO MEDIAN (PT/LH): {ratio:.2f}x")
    print("="*40)


if __name__ == "__main__":
    batch_size = 1
    n_trials = 100
    stats = run_full_benchmark(batch_size,n_trials,1000)
    print_report(stats)