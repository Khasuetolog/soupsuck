# Optimization backlog (not implemented in the correctness phase)

Measured on 0c58998 / e0c50a2 (Kaggle T4):
- Fixed overhead outside the training loop ≈ 24 s (model load, Soup setup, save); loop ≈ 46 s.
- First training step ≈ 9.9 s (Triton/cuBLAS compile and init) vs median 0.51 s.
- GPU time per step (profile): GEMM 36%, H2D/D2H offload copies 28% (mostly overlapped),
  NF4 decode 7.5%, attention 5.4%.

Ideas: cut first-step compile cost (fewer kernel specializations); kernel launch configs from
`tools/kaggle/bench_kernels.py` (S-029, not run yet); cheaper offload copies.
- solution/patch.py `_patch_lazy_mlp`: the trampoline only installs the fused MLP when
  `cls.forward is forward`; any external wrapper of Qwen2MLP.forward recurses (found by S-043 probe). Jury path unaffected; harden by materializing on install or tolerating wrappers.
