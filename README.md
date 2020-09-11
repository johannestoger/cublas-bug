# cublas-bug

Output on my machine when Ar = 24000, Ac = 100000.

Bug is triggered Bc > 1 and N_A > 2^31

```
Init cuBLAS...
N_A = 2400000000 (2.4e+09)
Setting values for A and B
Call cuBLAS
cuBLAS success
cudaDeviceSynchronize()
C[0] = 24000
C[299999] = 0
```

C[0] is correct, but C[299999] is not.
