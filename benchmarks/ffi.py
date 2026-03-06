import psutil
import time
import os
import torch
import lyonheart as lh
import numpy as np

# MONOCORE 
p = psutil.Process()
p.cpu_affinity([0])
torch.set_num_threads(1)
os.environ["RAYON_NUM_THREADS"] = "1"


def bench_lyonheart(iters):
    # Warmup
    for _ in range(100):
        lh.ffi()
    
    start = time.perf_counter_ns()
    for _ in range(iters):
        lh.ffi()
    end = time.perf_counter_ns()
    return ((end - start)/ iters)


def bench_pytorch(iters):
    # Warmup
    for _ in range(100):
        torch.is_grad_enabled()
   
    start = time.perf_counter_ns()
    for _ in range(iters):
        torch.is_grad_enabled()
    end = time.perf_counter_ns()
    return (end - start) / iters


def run_full_benchmark(n_trials=100, iters_per_trial=1000):
    lh_throughput = []
    pt_throughput = []
    print(f"--- Lancement du Benchmark ---")
    for i in range(n_trials):
        # LyonHeart
        t_lh = bench_lyonheart(iters_per_trial)
        lh_throughput.append(t_lh)
        # PyTorch
        t_pt = bench_pytorch(iters_per_trial)
        pt_throughput.append(t_pt)
        if (i + 1) % 10 == 0:
            print(f"Essai {i+1}/{n_trials} terminé...")

    stats = {
        "LyonHeart": {
            "min": np.min(lh_throughput),    # Ta vitesse de pointe (le "Min" temps)
            "median": np.median(lh_throughput),
            "p95": np.percentile(lh_throughput, 95), # seuil de 5% des pires perfs (lent)
            "std": np.std(lh_throughput)     # Écart-type (stabilité)
        },
        "PyTorch": {
            "min": np.min(pt_throughput),
            "median": np.median(pt_throughput),
            "p95": np.percentile(pt_throughput, 95),
            "std": np.std(pt_throughput)
        }
    }
    
    return stats

def print_report(stats):
    print("\n" + "="*40)
    print(f"{'METRIQUE':<15} | {'LYONHEART':<12} | {'PYTORCH':<12}")
    print("-"*40)
    for m in ["min", "median", "p95", "std"]:
        lh_v = stats["LyonHeart"][m]
        pt_v = stats["PyTorch"][m]
        print(f"{m.upper():<15} | {lh_v:>12.2f} | {pt_v:>12.2f}")
    
    ratio = stats["LyonHeart"]["median"] / stats["PyTorch"]["median"]
    diff = stats["LyonHeart"]["median"] - stats["PyTorch"]["median"] 
    print("-"*40)
    print(f"RATIO MEDIAN (PT/LH): {ratio:.2f}x")
    print(f"DIFF MEDIANE (PT/LH): {diff:.2f}ns")
    print("="*40)



if __name__ == "__main__":
    n_trials = 100
    stats = run_full_benchmark(n_trials,iters_per_trial=10000000)
    print_report(stats)