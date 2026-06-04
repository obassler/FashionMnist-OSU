import numpy as np

d = np.load('../outputs/barrier_matrix_data.npz', allow_pickle=True)
B = d['barriers']
names = d['names'].tolist()
endpoints = d['endpoint_loss']
n = len(names)

triu = np.triu_indices(n, k=1)
off = B[triu]

print("endpoint losses (after BN recal):")
for nm, lo in zip(names, endpoints):
    print(f"  {nm:14s} {lo:.4f}")

print(f"\nAll {len(off)} off-diagonal barriers:")
print(f"  min    {off.min():.3f}")
print(f"  max    {off.max():.3f}")
print(f"  mean   {off.mean():.3f}")
print(f"  median {np.median(off):.3f}")

same_opt = []
cross_opt = []
for i in range(n):
    for j in range(i + 1, n):
        oi, oj = names[i].split('/')[0], names[j].split('/')[0]
        (same_opt if oi == oj else cross_opt).append((names[i], names[j], B[i, j]))

print(f"\nSame-optimizer pairs (n={len(same_opt)}):")
for a, b, v in same_opt:
    print(f"  {a:12s} <-> {b:12s}  {v:.3f}")
print(f"  mean: {np.mean([v for *_, v in same_opt]):.3f}")

print(f"\nCross-optimizer pairs (n={len(cross_opt)}): mean={np.mean([v for *_, v in cross_opt]):.3f}")
