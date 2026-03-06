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


def bench_lyonheart(batch_size, iters):
    model = lh.nn.Sequential([lh.nn.Linear(784,128),lh.nn.ReLU(),lh.nn.Linear(128,1), lh.nn.Sigmoid()])
    x = lh.randn((batch_size,784))
    y = lh.randn((batch_size,1))
    criterion = lh.MSELoss()
    optimizer = lh.optim.SGD(model.parameters(),0.1)
    total_step = 0

    # Warmup
    for _ in range(100):
        l = criterion(model(x), y)
        model.backward(l)
        optimizer.step()

    x = lh.randn((batch_size,784))
    for _ in range(iters):
        y_ = model(x)
        loss = criterion(y_, y)
        model.backward(loss)
        t0 = time.perf_counter_ns()
        optimizer.step()
        t1 = time.perf_counter_ns()
        total_step+= (t1 - t0)
    return (total_step / iters)


def bench_pytorch(batch_size, iters):
    model = torch.nn.Sequential(torch.nn.Linear(784,128), torch.nn.ReLU(), torch.nn.Linear(128,1), torch.nn.Sigmoid()).to(torch.float32)
    x = torch.randn(batch_size,784, dtype=torch.float32)
    y = torch.randn(batch_size,1, dtype=torch.float32)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),0.1)
    total_step = 0

    # Warmup
    for _ in range(100):
        l = criterion(model(x), y)
        l.backward()
        optimizer.step()

    x = torch.randn((batch_size,784))
    for _ in range(iters):
        y_ = model(x)
        loss = criterion(y_, y)
        loss.backward()
        t0 = time.perf_counter_ns()
        optimizer.step()
        t1 = time.perf_counter_ns()
        total_step += (t1 - t0)
    return (total_step / iters)


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
    batch_size = 128
    n_trials = 50
    stats = run_full_benchmark(batch_size,n_trials,iters_per_trial=1000)
    print_report(stats)