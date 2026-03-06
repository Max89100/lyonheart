import psutil
import os
import gc
import time
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

    # Warmup
    for _ in range(100):
        l = criterion(model(x), y)
        model.backward(l)
        optimizer.step()

    x = lh.randn((batch_size,784))
    y = lh.randn((batch_size,1))
    gc.collect()
    for _ in range(iters):
        y_ = model(x)
        loss = criterion(y_, y)
        model.backward(loss)
        optimizer.step()


def bench_pytorch(batch_size, iters):
    model = torch.nn.Sequential(torch.nn.Linear(784,128), torch.nn.ReLU(), torch.nn.Linear(128,1), torch.nn.Sigmoid()).to(torch.float32)
    x = torch.randn(batch_size,784, dtype=torch.float32)
    y = torch.randn(batch_size,1, dtype=torch.float32)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),0.1)

    # Warmup
    for _ in range(100):
        l = criterion(model(x), y)
        l.backward()
        optimizer.step()

    x = torch.randn((batch_size,784))
    y = torch.randn((batch_size,1))
    gc.collect()
    for _ in range(iters):
        y_ = model(x)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()


def get_memory_peak():
    # Sur Windows, peak_wset est le "Peak Working Set" en octets
    process = psutil.Process(os.getpid())
    return process.memory_info().peak_wset / (1024 * 1024) # Retourne en MB

def bench_memory_usage(framework_name, task_fn):
    # 1. Nettoyage initial
    gc.collect()
    time.sleep(1) # Laisse le temps à l'OS de respirer
    
    start_mem = get_memory_peak()
    
    # 2. Exécution de la tâche (ex: Forward + Backward sur un gros Batch)
    task_fn()
    
    peak_mem = get_memory_peak()
    
    print(f"[{framework_name}] Peak RAM: {peak_mem:.2f} MB (Delta: {peak_mem - start_mem:.2f} MB)")
    return peak_mem


if __name__ == "__main__":
    bench_memory_usage("LyonHeart", lambda: bench_lyonheart(512,1000))
    bench_memory_usage("PyTorch", lambda: bench_pytorch(512,1000))