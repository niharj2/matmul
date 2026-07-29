import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

NAMES = {0: "cublas", 1: "native", 2: "coalesce", 3: "shared", 4: "1d_blocktiling", 5: "2d_blocktiling"}
COLORS = {0: "k", 1: "r", 2: "g", 3: "b", 4: "m", 5: "c"}


def parse_file(file):
    with open(file, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    data = []
    pattern = r"Average elasped time: \((.*?)\) second, performance: \((.*?)\) GFLOPS. size: \((.*?)\)."
    for line in lines:
        r = re.match(pattern, line)
        if r:
            data.append(float(r.group(2)))
    return data


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    fig = plt.figure(figsize=(12, 10))
    for num, name in NAMES.items():
        path = os.path.join(root, f"test/test_kernel_{num}.txt")
        if not os.path.exists(path):
            continue
        y = parse_file(path)
        x = [(i + 1) * 256 for i in range(len(y))]
        plt.plot(x, y, c=COLORS[num], linewidth=2, label=name)
        plt.scatter(x, y, marker="o", s=30, c=COLORS[num])

    plt.legend(fontsize=12)
    plt.tick_params(labelsize=10)
    plt.xlabel("Matrix size (M=N=K)", fontsize=12, fontweight="bold")
    plt.ylabel("Performance (GFLOPS)", fontsize=12, fontweight="bold")
    plt.title("SGEMM kernel comparison", fontsize=16, fontweight="bold")
    plt.gca().xaxis.set_major_locator(MultipleLocator(512))

    save_dir = os.path.join(root, "images")
    os.makedirs(save_dir, exist_ok=True)
    # Optional output filename: `python plot.py <name>` (default all_kernels.png).
    import sys
    fname = sys.argv[1] if len(sys.argv) > 1 else "all_kernels.png"
    if not fname.endswith(".png"):
        fname += ".png"
    out = os.path.join(save_dir, fname)
    plt.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
