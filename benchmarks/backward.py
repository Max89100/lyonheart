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
    model = lh.nn.Sequential([lh.nn.Linear(784,128),lh.nn.ReLU(),lh.nn.Linear(128,1), lh.nn.Sigmoid()])
    x = lh.randn((batch_size,784))
    y = lh.randn((batch_size,1))
    criterion = lh.MSELoss()
    total_forward = 0
    total_backward = 0

    # Warmup
    for _ in range(100):
        l = criterion(model(x), y)
        model.backward(l)

    for _ in range(iters):
        # --- FORWARD ---
        t0 = time.perf_counter_ns()
        y_ = model(x)
        t1 = time.perf_counter_ns()

        loss = criterion(y_, y)
        total_forward += (t1 - t0)

        # --- BACKWARD ---
        t2 = time.perf_counter_ns()
        model.backward(loss)
        t3 = time.perf_counter_ns()
        total_backward += (t3 - t2)
    return (total_forward / iters) / 1e9, (total_backward / iters) / 1e9


def bench_pytorch(batch_size, iters):
    model = torch.nn.Sequential(torch.nn.Linear(784,128), torch.nn.ReLU(), torch.nn.Linear(128,1), torch.nn.Sigmoid()).to(torch.float32)
    x = torch.randn(batch_size,784, dtype=torch.float32)
    y = torch.randn(batch_size,1, dtype=torch.float32)
    criterion = torch.nn.MSELoss()
    total_forward = 0
    total_backward = 0

    # Warmup
    for _ in range(100):
        l = criterion(model(x), y)
        l.backward()

    for _ in range(iters):
        # --- FORWARD ---
        t0 = time.perf_counter_ns()
        y_ = model(x)
        t1 = time.perf_counter_ns()

        loss = criterion(y_, y)
        total_forward += (t1 - t0)

        # --- BACKWARD ---
        t2 = time.perf_counter_ns()
        loss.backward()
        t3 = time.perf_counter_ns()
        total_backward += (t3 - t2)
    return (total_forward / iters) / 1e9, (total_backward / iters) / 1e9


def run_full_benchmark(batch_size, n_trials=50, iters_per_trial=100):
    # Diminue un peu n_trials si c'est trop long, 50 c'est déjà très robuste statistiquement
    f_lh_throughput = []
    b_lh_throughput = []
    f_pt_throughput = []
    b_pt_throughput = []

    print(f"--- Lancement du Benchmark (Batch Size: {batch_size}) ---")
    for i in range(n_trials):
        # LyonHeart
        t_f_lh, t_b_lh = bench_lyonheart(batch_size, iters_per_trial)
        f_lh_throughput.append(batch_size / t_f_lh)
        b_lh_throughput.append(batch_size / t_b_lh)

        # PyTorch
        t_f_pt, t_b_pt = bench_pytorch(batch_size, iters_per_trial)
        f_pt_throughput.append(batch_size / t_f_pt)
        b_pt_throughput.append(batch_size / t_b_pt)

        if (i + 1) % 5 == 0:
            print(f"Essai {i+1}/{n_trials} terminé...")

    stats = {
        "LyonHeart": {
            "max_f": np.max(f_lh_throughput),    # Ta vitesse de pointe (le "Min" temps)
            "median_f": np.median(f_lh_throughput),
            "p95_f": np.percentile(f_lh_throughput, 5), # seuil de 5% des pires perfs (lent)
            "std_f": np.std(f_lh_throughput),
            "max_b": np.max(b_lh_throughput),
            "median_b": np.median(b_lh_throughput),
            "p95_b": np.percentile(b_lh_throughput, 5), 
            "std_b": np.std(b_lh_throughput), 
            "ratio": np.median(f_lh_throughput) / np.median(b_lh_throughput)
        },
        "PyTorch": {
            "max_f": np.max(f_pt_throughput),    
            "median_f": np.median(f_pt_throughput),
            "p95_f": np.percentile(f_pt_throughput, 5),
            "std_f": np.std(f_pt_throughput),
            "max_b": np.max(b_pt_throughput),
            "median_b": np.median(b_pt_throughput),
            "p95_b": np.percentile(b_pt_throughput, 5), 
            "std_b": np.std(b_pt_throughput), 
            "ratio": np.median(f_pt_throughput) / np.median(b_pt_throughput)
        }
    }
    
    return stats

def print_report(stats):
    print("\n" + "="*40)
    print(f"{'METRIQUE':<15} | {'LYONHEART':<12} | {'PYTORCH':<12}")
    print("-"*40)
    for m in ["max_f", "median_f", "p95_f", "std_f","max_b", "median_b", "p95_b", "std_b","ratio"]:
        lh_v = stats["LyonHeart"][m]
        pt_v = stats["PyTorch"][m]
        print(f"{m.upper():<15} | {lh_v:>12.2f} | {pt_v:>12.2f}")
    
    ratio_forward = stats["PyTorch"]["median_f"] / stats["LyonHeart"]["median_f"]
    ratio_backward = stats["PyTorch"]["median_b"] / stats["LyonHeart"]["median_b"]
    print("-"*40)
    print(f"RATIO MEDIAN FORWARD (PT/LH): {ratio_forward:.2f}x")
    print("="*40)
    print("-"*40)
    print(f"RATIO MEDIAN BACKWARD (PT/LH): {ratio_backward:.2f}x")
    print("="*40)


if __name__ == "__main__":
    batch_size = 128
    n_trials = 50
    stats = run_full_benchmark(batch_size,n_trials,iters_per_trial=1000)
    print_report(stats)