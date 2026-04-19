import time
import torch
from torch.utils.data import DataLoader, TensorDataset


def benchmark_dataloader(num_workers_list=[0, 2, 4, 8],
                          num_samples=2048,
                          batch_size=32,
                          num_batches=50):
    """
    Benchmarks DataLoader speed across different num_workers settings.
    Uses a synthetic TensorDataset so no real data is needed.

    Key concepts demonstrated:
      - num_workers=0: single-process, CPU loads data on the main thread.
        GPU sits idle waiting. Causes spiky GPU utilisation.
      - num_workers>0: parallel workers prefetch batches in background.
        GPU stays busy. Smooth utilisation.
      - pin_memory=True: data loaded into page-locked memory.
        Faster CPU->GPU transfer. Always True when training on GPU.
      - prefetch_factor: each worker preloads this many batches ahead.
    """

    # Synthetic dataset: 2048 random images of size 3x224x224
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size} | Batches timed: {num_batches}\n")
    print(f"{'num_workers':<15} {'pin_memory':<14} {'time (s)':<12} {'batches/s':<12}")
    print("-" * 55)

    results = []

    for nw in num_workers_list:
        for pin in ([True, False] if str(device) != "cpu" else [False]):
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=nw,
                pin_memory=pin,
                prefetch_factor=2 if nw > 0 else None,
                persistent_workers=True if nw > 0 else False,
            )

            # Warm up
            for i, (x, y) in enumerate(loader):
                x = x.to(device)
                if i == 2:
                    break

            # Time it
            start = time.perf_counter()
            for i, (x, y) in enumerate(loader):
                x = x.to(device)
                if i == num_batches:
                    break
            elapsed = time.perf_counter() - start
            bps = num_batches / elapsed

            print(f"{nw:<15} {str(pin):<14} {elapsed:<12.3f} {bps:<12.1f}")
            results.append((nw, pin, elapsed, bps))

    print("\n✓ Conclusion:")
    fastest = min(results, key=lambda r: r[2])
    slowest = max(results, key=lambda r: r[2])
    speedup = slowest[2] / fastest[2]
    print(f"  Fastest: num_workers={fastest[0]}, pin_memory={fastest[1]}")
    print(f"  Slowest: num_workers={slowest[0]}, pin_memory={slowest[1]}")
    print(f"  Speedup: {speedup:.1f}x faster with optimal settings\n")

    return results


if __name__ == "__main__":
    benchmark_dataloader()